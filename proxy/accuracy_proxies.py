"""Accuracy proxies for GPT-2-class transformer NAS.

Based on LPZero (arXiv:2410.04808) GPT-2/WikiText-103 benchmark rankings:
  - decoder_params: Spearman 0.967, Kendall 0.847 (fastest, no data needed)
  - SNIP:          Spearman 0.970, Kendall 0.858 (one forward+backward pass)
  - grad_norm:     Spearman 0.958, Kendall 0.834 (one backward pass)
  - LPZero:        Spearman 0.980, Kendall 0.886 (attention-aware, best known)
  - NEAR:          Not validated for LMs -- ablation only
  - SynFlow:       Spearman 0.730 -- too weak for primary use
  - jacob_cov:     Spearman 0.362 -- poor for transformers
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from proxy.features import ProxyFeatures, TransformerConfig

logger = logging.getLogger(__name__)

# Default composite weights (must sum to 1.0; normalised at call time)
_DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "decoder_params": 0.6,
    "grad_norm": 0.3,
    "near": 0.1,
}


# ---------------------------------------------------------------------------
# Proxy: decoder_params_score
# ---------------------------------------------------------------------------


def decoder_params_score(config: TransformerConfig) -> float:
    """Non-embedding parameter count expressed as log-scale score.

    Fastest possible proxy: requires only the config, no model, no data.
    Validated on GPT-2/WikiText-103: Spearman 0.967, Kendall 0.847.

    decoder_params = total_params
                   - vocab_size * d_model          (input embedding)
                   - (0 if tie_embeddings else vocab_size * d_model) (output)
                   - (0 if use_rotary else seq_len * d_model)  (positional)

    Returns
    -------
    float
        log(decoder_params).  Higher = more capacity = better predicted accuracy.
    """
    features = ProxyFeatures.from_config(config)
    total_params: int = features.total_params

    embed_params: int = config.vocab_size * config.d_model
    pos_embed_params: int = (
        0 if config.use_rotary else config.seq_len * config.d_model
    )
    lm_head_params: int = (
        0 if config.tie_embeddings else config.vocab_size * config.d_model
    )

    decoder_params: int = total_params - embed_params - pos_embed_params - lm_head_params

    if decoder_params <= 0:
        # Guard: return log of total params as fallback (should not happen for
        # valid configs, but protects against edge cases).
        logger.warning(
            "decoder_params <= 0 (%d) for config %s; falling back to log(total_params)",
            decoder_params,
            config,
        )
        return math.log(max(total_params, 1))

    return math.log(float(decoder_params))


# ---------------------------------------------------------------------------
# Proxy: grad_norm_score
# ---------------------------------------------------------------------------


def grad_norm_score(
    model: nn.Module,
    data: torch.Tensor,
    device: str = "cpu",
    n_batches: int = 1,
) -> float:
    """Sum of gradient L2 norms after one backward pass on language modeling loss.

    Validated on GPT-2/WikiText-103: Spearman 0.958, Kendall 0.834.

    LM loss = CrossEntropyLoss(logits[:, :-1], tokens[:, 1:])  (next-token prediction)
    Score    = sum over params of ||grad||_2

    One forward + backward pass only.  Does NOT update parameters.

    Parameters
    ----------
    model:
        GPT-2-class transformer (must accept integer token IDs as input).
    data:
        Integer token IDs, shape ``(batch, seq_len)``, on *device*.
    device:
        Device string passed to ``model.to(device)`` and ``data.to(device)``.
    n_batches:
        Number of batches to average the gradient signal over.  Defaults to 1
        (one forward+backward pass is sufficient for zero-cost ranking).

    Returns
    -------
    float
        Sum of per-parameter gradient L2 norms.  Non-negative.
        Returns 0.0 on any exception.
    """
    try:
        model = model.to(device)
        data = data.to(device)
        model.train()
        model.zero_grad()

        total_norm: float = 0.0
        for i in range(n_batches):
            batch = data[i : i + 1] if n_batches > 1 else data
            logits = model(batch)  # (batch, seq_len, vocab_size)
            vocab_size = logits.shape[-1]

            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                total_norm += float(param.grad.data.norm(2))

        model.zero_grad()
        return total_norm

    except Exception:  # pylint: disable=broad-except
        logger.debug("grad_norm_score failed", exc_info=True)
        try:
            model.zero_grad()
        except Exception:  # pylint: disable=broad-except
            pass
        return 0.0


# ---------------------------------------------------------------------------
# Proxy: snip_score
# ---------------------------------------------------------------------------


def snip_score(
    model: nn.Module,
    data: torch.Tensor,
    device: str = "cpu",
    n_batches: int = 1,
) -> float:
    """SNIP (Lee et al. 2019): sum of |grad * weight| across all parameters.

    Measures connection sensitivity to the loss.
    Validated on GPT-2/WikiText-103: Spearman 0.970, Kendall 0.858.

    LM loss = CrossEntropyLoss(logits[:, :-1], tokens[:, 1:])
    Score    = sum over params of |grad_p * param_p|

    One forward + backward pass.  Does NOT update parameters.

    Parameters
    ----------
    model:
        GPT-2-class transformer (must accept integer token IDs as input).
    data:
        Integer token IDs, shape ``(batch, seq_len)``, on *device*.
    device:
        Device string.
    n_batches:
        Number of batches to average over.

    Returns
    -------
    float
        SNIP sensitivity score.  Non-negative.
        Returns 0.0 on any exception.
    """
    try:
        model = model.to(device)
        data = data.to(device)
        model.train()
        model.zero_grad()

        for i in range(n_batches):
            batch = data[i : i + 1] if n_batches > 1 else data
            logits = model(batch)  # (batch, seq_len, vocab_size)
            vocab_size = logits.shape[-1]

            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            loss.backward()

        sensitivity: float = 0.0
        for param in model.parameters():
            if param.grad is not None:
                sensitivity += float(
                    (param.grad.data * param.data).abs().sum()
                )

        model.zero_grad()
        return sensitivity

    except Exception:  # pylint: disable=broad-except
        logger.debug("snip_score failed", exc_info=True)
        try:
            model.zero_grad()
        except Exception:  # pylint: disable=broad-except
            pass
        return 0.0


# ---------------------------------------------------------------------------
# Proxy: near_score_lm
# ---------------------------------------------------------------------------


def near_score_lm(
    model: nn.Module,
    data: torch.Tensor,
    device: str = "cpu",
) -> float:
    """NEAR score adapted for language model transformers.

    Reuses ``zkml_bench.nas.zero_cost_proxies.near_score`` but wraps it
    for the integer-token input format used by GPT-2-class LMs.

    **NOT validated for GPT-2/WikiText ranking** — use for ablation only.

    Parameters
    ----------
    model:
        GPT-2-class transformer that accepts integer token IDs.
    data:
        Integer token IDs, shape ``(batch, seq_len)``.
    device:
        Device string.

    Returns
    -------
    float
        NEAR score (higher = more performant predicted).
        Returns 0.0 on ``ImportError`` or any other exception.
    """
    try:
        from zkml_bench.nas.zero_cost_proxies import (  # pylint: disable=import-outside-toplevel
            near_score,
        )

        model = model.to(device)
        data = data.to(device)
        return float(near_score(model, data))

    except Exception:  # pylint: disable=broad-except
        logger.debug("near_score_lm failed (import or runtime error)", exc_info=True)
        return 0.0


# ---------------------------------------------------------------------------
# Composite proxy
# ---------------------------------------------------------------------------


def composite_accuracy_score(
    config: TransformerConfig,
    model: Optional[nn.Module] = None,
    data: Optional[torch.Tensor] = None,
    device: str = "cpu",
    weights: Optional[dict[str, float]] = None,
) -> dict[str, object]:
    """Compute all accuracy proxies and return a weighted composite score.

    If *model* / *data* are not provided, only ``decoder_params`` is computed
    (pure zero-cost, no forward pass required).  When both are supplied,
    ``grad_norm``, ``snip``, and ``near`` are also computed.

    Higher composite = better predicted accuracy.

    Parameters
    ----------
    config:
        Transformer architecture to evaluate.
    model:
        Optional instantiated PyTorch model.  Required for gradient-based proxies.
    data:
        Optional integer token ID tensor, shape ``(batch, seq_len)``.  Required
        for gradient-based proxies.
    device:
        Device string (e.g., ``'cpu'``, ``'cuda'``).
    weights:
        Override composite weights.  Recognised keys: ``'decoder_params'``,
        ``'grad_norm'``, ``'near'``.  Missing keys take their default value;
        weights are renormalised to sum to 1.

    Returns
    -------
    dict
        Keys:

        * ``"decoder_params"`` — log(decoder_params) score (always present).
        * ``"grad_norm"`` — gradient norm score (present when model+data given).
        * ``"snip"`` — SNIP sensitivity score (present when model+data given).
        * ``"near"`` — NEAR score (present when model+data given).
        * ``"composite"`` — weighted sum of available proxy scores.
        * ``"proxies_used"`` — list of proxy names included in the composite.
    """
    effective_weights: dict[str, float] = dict(_DEFAULT_COMPOSITE_WEIGHTS)
    if weights is not None:
        effective_weights.update(weights)

    result: dict[str, object] = {}
    proxies_used: list[str] = []

    # --- Always compute decoder_params (no model needed) ---
    dp_score = decoder_params_score(config)
    result["decoder_params"] = dp_score
    proxies_used.append("decoder_params")

    if model is not None and data is not None:
        # --- Gradient-based proxies ---
        gn = grad_norm_score(model, data, device=device)
        result["grad_norm"] = gn
        proxies_used.append("grad_norm")

        sn = snip_score(model, data, device=device)
        result["snip"] = sn
        proxies_used.append("snip")

        nr = near_score_lm(model, data, device=device)
        result["near"] = nr
        proxies_used.append("near")

    # --- Build composite ---
    # Collect weights for available proxies only (snip is computed but not
    # included in the composite by default; caller can override via *weights*).
    composite_keys = [k for k in proxies_used if k in effective_weights]
    raw_weights = {k: effective_weights[k] for k in composite_keys}
    total_w = sum(raw_weights.values())

    if total_w <= 0.0 or not composite_keys:
        # Fallback: equal-weight average of available scores.
        scores_list = [float(result[k]) for k in proxies_used]  # type: ignore[arg-type]
        composite = sum(scores_list) / max(len(scores_list), 1)
    else:
        composite = sum(
            (raw_weights[k] / total_w) * float(result[k])  # type: ignore[arg-type]
            for k in composite_keys
        )

    result["composite"] = composite
    result["proxies_used"] = proxies_used

    return result

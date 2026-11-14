"""Proxy validation experiment: Table 1 — zero-cost accuracy proxies vs. calibration PPL.

Validates zero-cost proxies (decoder_params, grad_norm, SNIP, NEAR) against
short-training perplexity on WikiText-2 for GPT-2-class transformers.

Usage
-----
    python experiments/proxy_validation.py --n-configs 50 --steps 500

The script generates:
- ``<output_dir>/results.json``        — raw results
- ``<output_dir>/proxy_correlation.png`` — 4-panel scatter plot
- ``<output_dir>/table1.md``           — LaTeX-ready markdown table
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from proxy.accuracy_proxies import (
    decoder_params_score,
    grad_norm_score,
    near_score_lm,
    snip_score,
)
from proxy.features import TransformerConfig
from search.lm_trainer import LMTrainer, LMTrainingConfig
from search.transformer_space import TransformerSearchSpace, ZKTransformer

# Use non-interactive Agg backend (must be set before any figure is created).
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROXY_NAMES: list[str] = ["decoder_params", "grad_norm", "snip", "near"]

_PROXY_NOTES: dict[str, str] = {
    "decoder_params": "No model needed",
    "grad_norm": "1 backward pass",
    "snip": "1 backward pass",
    "near": "Not recommended",
}

_PROXY_DISPLAY: dict[str, str] = {
    "decoder_params": "Decoder params",
    "grad_norm": "GradNorm",
    "snip": "SNIP",
    "near": "NEAR (ablation)",
}

# Number of gradient batches used for gradient-based proxies.
_N_PROXY_BATCHES: int = 4

# Batch size used when building the calibration data tensor for proxy scoring.
_PROXY_BATCH_SIZE: int = 4


# ---------------------------------------------------------------------------
# Config serialisation helpers (mirrors search/nas_loop.py conventions)
# ---------------------------------------------------------------------------


def _config_to_dict(config: TransformerConfig) -> dict[str, Any]:
    """Serialize a TransformerConfig to a plain JSON-compatible dict."""
    return {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "seq_len": config.seq_len,
        "ff_mult": config.ff_mult,
        "activation": config.activation,
        "use_rotary": config.use_rotary,
        "use_layernorm": config.use_layernorm,
        "vocab_size": config.vocab_size,
        "tie_embeddings": config.tie_embeddings,
    }


def _config_from_dict(d: dict[str, Any]) -> TransformerConfig:
    """Deserialize a TransformerConfig from a plain dict."""
    return TransformerConfig(
        d_model=int(d["d_model"]),
        n_heads=int(d["n_heads"]),
        n_layers=int(d["n_layers"]),
        seq_len=int(d["seq_len"]),
        ff_mult=float(d["ff_mult"]),
        activation=str(d["activation"]),
        use_rotary=bool(d["use_rotary"]),
        use_layernorm=bool(d["use_layernorm"]),
        vocab_size=int(d.get("vocab_size", 50257)),
        tie_embeddings=bool(d.get("tie_embeddings", False)),
    )


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(  # pylint: disable=too-many-locals
    space: TransformerSearchSpace,
    n: int,
    seed: int = 42,
) -> list[TransformerConfig]:
    """Sample *n* configs stratified by ZK cost tier (d_model thresholds).

    Tiers:
    - small:  d_model <= 384  (~20% of sample)
    - medium: 384 < d_model <= 768  (~60%)
    - large:  d_model > 768  (~20%)

    At least 10 configs will have ZK-friendly activations (poly2/poly3/linear).
    When the stratification cannot be satisfied exactly (e.g. n is very small),
    the remaining slots are filled with random configs.

    Parameters
    ----------
    space:
        The search space to sample from.
    n:
        Total number of configs to return.
    seed:
        Random seed.

    Returns
    -------
    list[TransformerConfig]
        List of *n* sampled configs.
    """
    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    space.rng = np.random.RandomState(seed)  # pylint: disable=no-member

    n_small = max(1, round(0.20 * n))
    n_large = max(1, round(0.20 * n))
    n_medium = n - n_small - n_large

    zk_friendly_activations = {"poly2", "poly3", "linear"}

    def _tier(d_model: int) -> str:
        if d_model <= 384:
            return "small"
        if d_model <= 768:
            return "medium"
        return "large"

    def _sample_tier(target_tier: str, count: int) -> list[TransformerConfig]:
        """Sample *count* configs from the given tier by rejection sampling."""
        sampled: list[TransformerConfig] = []
        max_attempts = count * 200
        attempts = 0
        while len(sampled) < count and attempts < max_attempts:
            cfg = space.random_config()
            if _tier(cfg.d_model) == target_tier:
                sampled.append(cfg)
            attempts += 1
        # Fill remaining with any valid config if rejection sampling stalled.
        while len(sampled) < count:
            sampled.append(space.random_config())
        return sampled

    small_configs = _sample_tier("small", n_small)
    medium_configs = _sample_tier("medium", n_medium)
    large_configs = _sample_tier("large", n_large)

    configs: list[TransformerConfig] = small_configs + medium_configs + large_configs

    # Ensure at least 10 ZK-friendly activation configs exist when n >= 10.
    if n >= 10:
        zk_count = sum(1 for c in configs if c.activation in zk_friendly_activations)
        zk_needed = max(0, 10 - zk_count)
        if zk_needed > 0:
            # Append extra ZK-friendly configs (do not replace; avoids index errors).
            zk_friendly_candidates = space.zk_friendly_candidates(n=zk_needed * 10)
            added = 0
            extra: list[TransformerConfig] = []
            for candidate in zk_friendly_candidates:
                if added >= zk_needed:
                    break
                if candidate.activation in zk_friendly_activations:
                    extra.append(candidate)
                    added += 1
            # Replace tail of configs with extra ZK-friendly ones so total stays n.
            if extra:
                replace_count = min(len(extra), len(configs))
                configs[-replace_count:] = extra[:replace_count]

    # Shuffle deterministically.
    indices = rng.permutation(len(configs)).tolist()
    configs = [configs[i] for i in indices]

    return configs[:n]


# ---------------------------------------------------------------------------
# Per-config proxy computation
# ---------------------------------------------------------------------------


@dataclass
class _ConfigResult:
    """All proxy scores and PPL for one config."""

    config: TransformerConfig
    decoder_params: float
    grad_norm: float
    snip: float
    near: float
    calibration_ppl: float


def _compute_proxy_scores(
    config: TransformerConfig,
    device: str,
    seq_len: int,
) -> tuple[float, float, float, float, "ZKTransformer"]:
    """Build model and compute all four proxy scores.

    The built model is returned so the caller can reuse it for calibration
    training without building it a second time.

    Returns
    -------
    tuple[float, float, float, float, ZKTransformer]
        ``(decoder_params, grad_norm, snip, near, model)``.
        All four scores are non-negative floats; higher = better accuracy.
    """
    dp = decoder_params_score(config)

    model = ZKTransformer(config)
    data = torch.randint(
        0,
        config.vocab_size,
        (_PROXY_BATCH_SIZE, seq_len),
    )

    gn = grad_norm_score(model, data, device=device, n_batches=_N_PROXY_BATCHES)
    sn = snip_score(model, data, device=device, n_batches=_N_PROXY_BATCHES)
    nr = near_score_lm(model, data, device=device)

    return dp, gn, sn, nr, model


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between *x* and *y*."""
    if len(x) < 3:
        return float("nan")
    result = spearmanr(x, y)
    return float(result.statistic)


def _kendall(x: list[float], y: list[float]) -> float:
    """Kendall rank correlation between *x* and *y*."""
    if len(x) < 3:
        return float("nan")
    result = kendalltau(x, y)
    return float(result.statistic)


def _top10_precision(
    proxy_scores: list[float],
    ppls: list[float],
    higher_is_better: bool = True,
) -> float:
    """Fraction of true top-10% (by PPL) that proxy correctly ranks in top-10%.

    Lower PPL = better accuracy, so "true top-10%" = configs with lowest PPL.
    For proxy scores, higher is better (unless ``higher_is_better=False``).

    Parameters
    ----------
    proxy_scores:
        Raw proxy scores for each config.
    ppls:
        Calibration PPL for each config (lower = better).
    higher_is_better:
        When True (default), higher proxy score = better predicted accuracy.

    Returns
    -------
    float
        Precision at 10%, in [0, 1].
    """
    n = len(proxy_scores)
    if n < 2:
        return float("nan")

    k = max(1, math.ceil(0.10 * n))

    # True top-k: lowest PPL configs.
    ppl_rank = np.argsort(ppls)  # ascending: index 0 = best (lowest PPL)
    true_top_k: set[int] = set(ppl_rank[:k].tolist())

    # Predicted top-k: highest proxy score (or lowest if not higher_is_better).
    scores_arr = np.array(proxy_scores)
    if higher_is_better:
        pred_rank = np.argsort(-scores_arr)
    else:
        pred_rank = np.argsort(scores_arr)
    predicted_top_k: set[int] = set(pred_rank[:k].tolist())

    overlap = len(true_top_k & predicted_top_k)
    return overlap / k


# ---------------------------------------------------------------------------
# Incremental save helper
# ---------------------------------------------------------------------------


def _save_partial_results(
    output_dir: str,
    partial_items: list[_ConfigResult],
    n_configs: int,
    calibration_steps: int,
) -> None:
    """Save partial results JSON after each config to survive crashes."""
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "n_configs_total": n_configs,
        "n_configs_done": len(partial_items),
        "calibration_steps": calibration_steps,
        "configs": [_config_to_dict(r.config) for r in partial_items],
        "scores": {
            "decoder_params": [r.decoder_params for r in partial_items],
            "grad_norm": [r.grad_norm for r in partial_items],
            "snip": [r.snip for r in partial_items],
            "near": [r.near for r in partial_items],
        },
        "ppls": [r.calibration_ppl for r in partial_items],
    }
    path = os.path.join(output_dir, "results_partial.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,too-many-positional-arguments
def run_proxy_validation(
    n_configs: int = 50,
    calibration_steps: int = 500,
    seq_len: int = 128,
    output_dir: str = "experiments/results/proxy_validation",
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Validate zero-cost proxies against short-training PPL on WikiText-2.

    Algorithm
    ---------
    1. Sample *n_configs* diverse configs from :class:`TransformerSearchSpace`
       (stratified: ~20% small, ~60% GPT-2-class, ~20% large).
    2. For each config:

       a. Compute ``decoder_params_score`` (free, no model).
       b. Build model, compute ``grad_norm_score`` (one backward pass on 4 batches).
       c. Compute ``snip_score`` (one backward pass).
       d. Compute ``near_score_lm`` (forward pass only).
       e. Train for *calibration_steps* steps, record calibration PPL.

    3. Compute Spearman and Kendall correlations: each proxy vs calibration PPL.
    4. Compute top-10 % precision for each proxy.
    5. Save results to *output_dir* as JSON + markdown table.
    6. Return results dict.

    Parameters
    ----------
    n_configs:
        Number of architectures to evaluate.
    calibration_steps:
        Gradient steps for the short training run.
    seq_len:
        Sequence length (must be in the search space grid).
    output_dir:
        Directory to write ``results.json``, ``table1.md``, and
        ``proxy_correlation.png``.
    seed:
        Random seed for reproducibility.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    dict
        Full results dict; see module docstring for schema.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    space = TransformerSearchSpace(vocab_size=50257, seed=seed)
    configs = stratified_sample(space, n=n_configs, seed=seed)

    training_cfg = LMTrainingConfig(
        max_steps=calibration_steps,
        eval_steps=calibration_steps,  # evaluate only at the end
        batch_size=4,
        seq_len=seq_len,
        lr=3e-4,
        warmup_steps=min(50, calibration_steps // 4),
        weight_decay=0.1,
        grad_clip=1.0,
        device=device,
        seed=seed,
    )
    trainer = LMTrainer(training_cfg)

    partial_results: list[_ConfigResult] = []

    for idx, config in enumerate(configs):
        # Override seq_len in config to match training seq_len, keeping validity.
        if config.seq_len != seq_len:
            try:
                config = TransformerConfig(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_layers=config.n_layers,
                    seq_len=seq_len,
                    ff_mult=config.ff_mult,
                    activation=config.activation,
                    use_rotary=config.use_rotary,
                    use_layernorm=config.use_layernorm,
                    vocab_size=config.vocab_size,
                    tie_embeddings=config.tie_embeddings,
                )
            except ValueError:
                logger.warning(
                    "Could not override seq_len for config %d; skipping.", idx
                )
                continue

        # --- Compute proxy scores (also builds model for calibration reuse) ---
        try:
            dp, gn, sn, nr, model = _compute_proxy_scores(
                config, device=device, seq_len=seq_len
            )
        except Exception:  # pylint: disable=broad-except
            logger.warning("Proxy computation failed for config %d; skipping.", idx, exc_info=True)
            continue

        # --- Short calibration run (reuse model built by _compute_proxy_scores) ---
        try:
            train_result = trainer.calibration_run(model, n_steps=calibration_steps)
            ppl = train_result.val_perplexity
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Calibration run failed for config %d; skipping.", idx, exc_info=True
            )
            continue

        result = _ConfigResult(
            config=config,
            decoder_params=dp,
            grad_norm=gn,
            snip=sn,
            near=nr,
            calibration_ppl=ppl,
        )
        partial_results.append(result)
        _save_partial_results(output_dir, partial_results, n_configs, calibration_steps)

        print(
            f"[{idx + 1}/{n_configs}] d_model={config.d_model} "
            f"n_layers={config.n_layers}: "
            f"grad_norm={gn:.1f} ppl={ppl:.1f}"
        )

    # --- Aggregate ---
    if not partial_results:
        raise RuntimeError("No configs evaluated successfully; check logs.")

    ppls: list[float] = [r.calibration_ppl for r in partial_results]
    proxy_scores: dict[str, list[float]] = {
        "decoder_params": [r.decoder_params for r in partial_results],
        "grad_norm": [r.grad_norm for r in partial_results],
        "snip": [r.snip for r in partial_results],
        "near": [r.near for r in partial_results],
    }

    proxy_stats: dict[str, dict[str, float]] = {}
    for proxy_name in _PROXY_NAMES:
        scores = proxy_scores[proxy_name]
        sp = _spearman(scores, ppls)
        kt = _kendall(scores, ppls)
        # Proxy scores higher = better accuracy; lower PPL = better accuracy
        # So we expect negative Spearman (higher score → lower PPL).
        # top10_precision uses "higher proxy = better" directly.
        p10 = _top10_precision(scores, ppls, higher_is_better=True)
        proxy_stats[proxy_name] = {
            "spearman": sp,
            "kendall": kt,
            "top10_prec": p10,
        }

    # Determine best proxy by absolute Spearman.
    best_proxy = max(
        _PROXY_NAMES,
        key=lambda p: abs(proxy_stats[p]["spearman"])
        if math.isfinite(proxy_stats[p]["spearman"])
        else -1.0,
    )

    results_dict: dict[str, Any] = {
        "n_configs": len(partial_results),
        "calibration_steps": calibration_steps,
        "proxies": proxy_stats,
        "best_proxy": best_proxy,
        "configs": [_config_to_dict(r.config) for r in partial_results],
        "scores": proxy_scores,
        "ppls": ppls,
    }

    # --- Persist ---
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results_dict, fh, indent=2)
    logger.info("Results saved to %s", results_path)

    table_path = os.path.join(output_dir, "table1.md")
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(markdown_table(results_dict))
    logger.info("Markdown table saved to %s", table_path)

    plot_path = os.path.join(output_dir, "proxy_correlation.png")
    plot_proxy_correlations(results_dict, plot_path)
    logger.info("Correlation plot saved to %s", plot_path)

    return results_dict


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_proxy_correlations(
    validation_results: dict[str, Any],
    output_path: str,
) -> None:
    """Scatter plots: each proxy score vs calibration PPL.

    4-panel figure (one per proxy), saved as PNG.

    Parameters
    ----------
    validation_results:
        Results dict as returned by :func:`run_proxy_validation`.
    output_path:
        Absolute path for the output PNG file.
    """
    ppls: list[float] = validation_results["ppls"]
    scores_dict: dict[str, list[float]] = validation_results["scores"]
    proxy_stats: dict[str, dict[str, float]] = validation_results["proxies"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for i, proxy_name in enumerate(_PROXY_NAMES):
        ax = axes_flat[i]
        scores = scores_dict[proxy_name]
        sp = proxy_stats[proxy_name]["spearman"]
        sp_label = f"Spearman r={sp:.3f}" if math.isfinite(sp) else "Spearman r=N/A"

        ax.scatter(scores, ppls, alpha=0.6, s=30, label=sp_label)
        ax.set_xlabel(_PROXY_DISPLAY[proxy_name], fontsize=10)
        ax.set_ylabel("Calibration PPL", fontsize=10)
        ax.set_title(_PROXY_DISPLAY[proxy_name], fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Zero-Cost Proxy vs. Calibration PPL (WikiText-2)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def markdown_table(validation_results: dict[str, Any]) -> str:
    """Format proxy validation results as a markdown table for the paper.

    The table is sorted by absolute Spearman rank (best first).

    Parameters
    ----------
    validation_results:
        Results dict as returned by :func:`run_proxy_validation`.

    Returns
    -------
    str
        Multi-line markdown table string.
    """
    proxy_stats: dict[str, dict[str, float]] = validation_results["proxies"]

    header = "| Proxy | Spearman ρ | Kendall τ | P@10% | Notes |"
    separator = "|-------|------------|-----------|-------|-------|"

    # Sort by absolute Spearman, best first.
    sorted_proxies = sorted(
        _PROXY_NAMES,
        key=lambda p: abs(proxy_stats[p]["spearman"])
        if math.isfinite(proxy_stats[p]["spearman"])
        else -1.0,
        reverse=True,
    )

    rows = [header, separator]
    for proxy_name in sorted_proxies:
        stats = proxy_stats[proxy_name]
        sp = stats["spearman"]
        kt = stats["kendall"]
        p10 = stats["top10_prec"]
        sp_str = f"{sp:.2f}" if math.isfinite(sp) else "N/A"
        kt_str = f"{kt:.2f}" if math.isfinite(kt) else "N/A"
        p10_str = f"{p10:.2f}" if math.isfinite(p10) else "N/A"
        note = _PROXY_NOTES[proxy_name]
        display = _PROXY_DISPLAY[proxy_name]
        rows.append(f"| {display} | {sp_str} | {kt_str} | {p10_str} | {note} |")

    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate zero-cost accuracy proxies against calibration PPL."
    )
    parser.add_argument("--n-configs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--output-dir", default="experiments/results/proxy_validation"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_proxy_validation(
        n_configs=args.n_configs,
        calibration_steps=args.steps,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    print(markdown_table(results))

"""GPT-2-class transformer search space and model builder for ZK-NAS.

This module defines:
- ``SEARCH_SPACE``: the discrete hyperparameter grid.
- ``Poly2Activation``, ``Poly3Activation``, ``LinearActivation``: ZK-friendly
  activation nn.Modules (polynomial, avoids lookup tables in ZK circuits).
- ``RMSNorm``: Root-mean-square layer normalisation.
- ``TransformerBlock``: One transformer block supporting all search space variants.
- ``ZKTransformer``: Full GPT-2-class model for ZK-NAS evaluation.
- ``TransformerSearchSpace``: Samples / grids configs and estimates ZK cost.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from proxy.features import TransformerConfig

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

#: Discrete hyperparameter grid for the GPT-2-class search space.
SEARCH_SPACE: dict[str, list[Any]] = {
    "d_model": [256, 384, 512, 640, 768, 1024],
    "n_heads": [4, 6, 8, 12, 16],
    "n_layers": [2, 4, 6, 8, 10, 12, 16, 20, 24],
    "seq_len": [64, 128, 256, 512],
    "ff_mult": [2.0, 2.67, 4.0],
    "activation": ["gelu", "relu", "poly2", "poly3", "linear"],
    "use_rotary": [True, False],
    "use_layernorm": [True, False],
    "tie_embeddings": [True, False],
}

#: ZK-friendly activation names (polynomial / linear — no lookup tables).
_ZK_FRIENDLY_ACTIVATIONS: frozenset[str] = frozenset({"poly2", "poly3", "linear"})

# ---------------------------------------------------------------------------
# Activation modules (nn.Module subclasses for ONNX compatibility)
# ---------------------------------------------------------------------------


class Poly2Activation(nn.Module):
    """ZK-friendly activation: f(x) = x^2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * x


class Poly3Activation(nn.Module):
    """ZK-friendly activation: f(x) = x^3 + x."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * x * x + x


class LinearActivation(nn.Module):
    """ZK-friendly activation: identity f(x) = x."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


def _build_activation(name: str) -> nn.Module:
    """Return an activation nn.Module for *name*."""
    activations: dict[str, nn.Module] = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "poly2": Poly2Activation(),
        "poly3": Poly3Activation(),
        "linear": LinearActivation(),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation {name!r}; must be one of {sorted(activations)}"
        )
    return activations[name]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no mean subtraction).

    Formula: ``x / sqrt(mean(x^2) + eps) * weight``
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # x: (..., d_model)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------


def _rotary_freqs(head_dim: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Compute RoPE frequency tensor.

    Returns shape ``(seq_len, head_dim)`` where each row is the rotation
    angles for that position.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    # (seq_len, half)
    freqs = torch.outer(positions, inv_freq)
    # (seq_len, head_dim) by repeating sin/cos
    return torch.cat([freqs, freqs], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE application."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_len: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to queries and keys.

    Parameters
    ----------
    q, k:
        Shape ``(batch, n_heads, seq_len, head_dim)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Rotated ``(q, k)`` with the same shape.
    """
    freqs = _rotary_freqs(head_dim, seq_len, device)  # (seq_len, head_dim)
    cos = freqs.cos()[None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin = freqs.sin()[None, None, :, :]  # (1, 1, seq_len, head_dim)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Two-layer FFN with configurable activation and hidden dimension."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden = int(config.d_model * config.ff_mult)
        self.fc1 = nn.Linear(config.d_model, hidden, bias=True)
        self.act = _build_activation(config.activation)
        self.fc2 = nn.Linear(hidden, config.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.fc2(self.act(self.fc1(x)))  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention
# ---------------------------------------------------------------------------


# Finite fill for masked attention scores (exp(-100) underflows to 0 in fp32, so
# it is numerically equivalent to -inf).  It must stay modest: EZKL sizes its
# softmax lookup range, and hence ``logrows``, from the most negative input, so
# a fill of -1e4 inflates logrows from 19 to 26 for the same circuit rows.
CAUSAL_MASK_FILL = -100.0


class MultiHeadSelfAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-head self-attention supporting rotary embeddings and linear attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.use_rotary = config.use_rotary
        self.use_linear_attn = config.activation == "linear"
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute self-attention.

        Parameters
        ----------
        x:
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary positional embeddings if requested
        if self.use_rotary:
            q, k = _apply_rotary_emb(q, k, seq_len, self.head_dim, x.device)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # (batch, n_heads, seq_len, seq_len)

        # Causal mask: position i may only attend to positions <= i.  Without
        # it the model sees future tokens and next-token perplexity is leaked.
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # Linear attention: skip softmax; standard attention: apply softmax
        if self.use_linear_attn:
            attn_weights = scores * causal
        else:
            # Finite fill (not -inf): exp(fill) is negligible in fp32 and the graph stays
            # quantisable by ZK backends that cannot represent infinities.
            attn_weights = F.softmax(
                scores.masked_fill(~causal, CAUSAL_MASK_FILL), dim=-1
            )

        # Weighted value aggregation
        out = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return self.out_proj(out)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """One transformer block supporting all search space variants.

    Supports:
    - Rotary vs. learned positional embeddings (rotary applied in attention).
    - LayerNorm vs. RMSNorm (``use_layernorm=False`` → RMSNorm).
    - FFN activations: gelu | relu | poly2 (x²) | poly3 (x³+x) | linear (identity).
    - Linear attention (``activation='linear'``): no softmax in attention scores.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        norm_cls: type[nn.Module]
        if config.use_layernorm:
            norm_cls = nn.LayerNorm
        else:
            norm_cls = RMSNorm

        self.norm1: nn.Module = norm_cls(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2: nn.Module = norm_cls(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm transformer block with residual connections."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ZKTransformer
# ---------------------------------------------------------------------------


class ZKTransformer(nn.Module):
    """Full GPT-2-class transformer for ZK-NAS.

    Supports all activation variants including ZK-friendly polynomial
    activations (``poly2``, ``poly3``, ``linear``).  Can be exported to ONNX
    for ZK cost estimation.

    Parameters
    ----------
    config:
        Architectural hyperparameters.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding (learned positional embeddings added when not rotary)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        if not config.use_rotary:
            self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        else:
            self.pos_emb = None  # type: ignore[assignment]

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final normalisation
        if config.use_layernorm:
            self.final_norm: nn.Module = nn.LayerNorm(config.d_model)
        else:
            self.final_norm = RMSNorm(config.d_model)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with lm_head
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with a small normal distribution."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute language model logits.

        Parameters
        ----------
        input_ids:
            Integer token tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        _batch, seq_len = input_ids.shape
        x = self.token_emb(input_ids)  # (batch, seq_len, d_model)

        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_emb(positions)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)  # type: ignore[no-any-return]  # (batch, seq_len, vocab_size)

    def get_sample_input(self, batch_size: int = 1) -> torch.Tensor:
        """Return a sample integer input tensor for ONNX export.

        Parameters
        ----------
        batch_size:
            Number of sequences in the batch.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, seq_len)``, dtype ``torch.long``.
        """
        return torch.zeros(batch_size, self.config.seq_len, dtype=torch.long)


# ---------------------------------------------------------------------------
# TransformerSearchSpace
# ---------------------------------------------------------------------------


class TransformerSearchSpace:
    """Defines and samples from the GPT-2-class NAS search space.

    Parameters
    ----------
    vocab_size:
        Vocabulary size for all sampled configs.
    seed:
        Random seed for reproducible sampling.
    """

    def __init__(self, vocab_size: int = 50257, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)  # pylint: disable=no-member

    # ------------------------------------------------------------------
    # Validity
    # ------------------------------------------------------------------

    def is_valid(self, config: TransformerConfig) -> bool:
        """Check all validity constraints.

        Verifies the hard architectural constraint that ``d_model`` must be
        divisible by ``n_heads`` (required for multi-head attention) and that
        all field values satisfy the basic positivity / type requirements
        enforced by ``TransformerConfig.__post_init__``.

        Note: configs whose values lie outside the discrete ``SEARCH_SPACE``
        grids (e.g. the GPT-2 baseline with ``seq_len=1024``) are still
        considered valid — they are representable transformer architectures
        even if they are not sampled during search.
        """
        positive = (
            config.d_model > 0
            and config.n_heads > 0
            and config.n_layers > 0
            and config.seq_len > 0
            and config.ff_mult > 0.0
            and config.vocab_size > 0
        )
        return positive and config.d_model % config.n_heads == 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def random_config(self) -> TransformerConfig:
        """Sample a random valid config from the search space.

        Samples uniformly from all valid (d_model, n_heads) pairs, then
        independently from all other dimensions.
        """
        valid_pairs = [
            (d, h)
            for d in SEARCH_SPACE["d_model"]
            for h in SEARCH_SPACE["n_heads"]
            if d % h == 0
        ]
        idx = self.rng.randint(0, len(valid_pairs))
        d_model, n_heads = valid_pairs[idx]

        def _choice(key: str) -> Any:
            opts = SEARCH_SPACE[key]
            return opts[self.rng.randint(0, len(opts))]

        return TransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(_choice("n_layers")),
            seq_len=int(_choice("seq_len")),
            ff_mult=float(_choice("ff_mult")),
            activation=str(_choice("activation")),
            use_rotary=bool(_choice("use_rotary")),
            use_layernorm=bool(_choice("use_layernorm")),
            vocab_size=self.vocab_size,
            tie_embeddings=bool(_choice("tie_embeddings")),
        )

    # ------------------------------------------------------------------
    # Grid configs
    # ------------------------------------------------------------------

    def grid_configs(  # pylint: disable=too-many-locals
        self, max_configs: int = 1000
    ) -> list[TransformerConfig]:
        """Return a structured grid of configs for systematic evaluation.

        Covers corners of the search space and densely samples the
        ZK-friendly region (poly activations, smaller d_model).

        Parameters
        ----------
        max_configs:
            Upper bound on the number of returned configs.

        Returns
        -------
        list[TransformerConfig]
            Deduplicated, validity-checked configs.
        """
        seen: set[tuple[Any, ...]] = set()
        configs: list[TransformerConfig] = []

        def _cfg_key(cfg: TransformerConfig) -> tuple[Any, ...]:
            return (
                cfg.d_model,
                cfg.n_heads,
                cfg.n_layers,
                cfg.seq_len,
                cfg.ff_mult,
                cfg.activation,
                cfg.use_rotary,
                cfg.use_layernorm,
                cfg.tie_embeddings,
            )

        def _add(cfg: TransformerConfig) -> None:
            key = _cfg_key(cfg)
            if key not in seen and self.is_valid(cfg):
                seen.add(key)
                configs.append(cfg)

        def _try_add(**kwargs: Any) -> None:
            try:
                _add(TransformerConfig(vocab_size=self.vocab_size, **kwargs))
            except ValueError:
                pass

        # 1. Extremal corners: (min/max d_model) × (min/max n_layers) × activations
        extremal_pairs = [(256, 4), (1024, 16)]
        extremal_layers = [SEARCH_SPACE["n_layers"][0], SEARCH_SPACE["n_layers"][-1]]
        extremal_seqs = [SEARCH_SPACE["seq_len"][0], SEARCH_SPACE["seq_len"][-1]]
        for (d_model, n_heads), n_layers, activation, seq_len in itertools.product(
            extremal_pairs, extremal_layers, SEARCH_SPACE["activation"], extremal_seqs
        ):
            _try_add(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                seq_len=seq_len,
                ff_mult=4.0,
                activation=activation,
                use_rotary=False,
                use_layernorm=True,
                tie_embeddings=True,
            )
        if len(configs) >= max_configs:
            return configs[:max_configs]

        # 2. Dense ZK-friendly region
        zk_pairs = [(256, 4), (384, 6), (512, 8), (512, 4)]
        zk_layers = [2, 4, 6]
        zk_acts = ["poly2", "poly3", "linear"]
        for (
            (d_model, n_heads),
            n_layers,
            activation,
            seq_len,
            ff_mult,
        ) in itertools.product(
            zk_pairs,
            zk_layers,
            zk_acts,
            SEARCH_SPACE["seq_len"],
            SEARCH_SPACE["ff_mult"],
        ):
            _try_add(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                seq_len=seq_len,
                ff_mult=ff_mult,
                activation=activation,
                use_rotary=True,
                use_layernorm=False,
                tie_embeddings=True,
            )
            if len(configs) >= max_configs:
                return configs[:max_configs]

        # 3. Remaining: systematic Cartesian product (filtered)
        valid_pairs = [
            (d, h)
            for d in SEARCH_SPACE["d_model"]
            for h in SEARCH_SPACE["n_heads"]
            if d % h == 0
        ]
        for (d_model, n_heads), n_layers, activation, seq_len in itertools.product(
            valid_pairs,
            SEARCH_SPACE["n_layers"],
            SEARCH_SPACE["activation"],
            SEARCH_SPACE["seq_len"],
        ):
            _try_add(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                seq_len=seq_len,
                ff_mult=4.0,
                activation=activation,
                use_rotary=False,
                use_layernorm=True,
                tie_embeddings=True,
            )
            if len(configs) >= max_configs:
                return configs[:max_configs]

        return configs[:max_configs]

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def gpt2_baseline(self) -> TransformerConfig:
        """Return GPT-2 small config as the baseline."""
        return TransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            seq_len=1024,
            ff_mult=4.0,
            activation="gelu",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=50257,
            tie_embeddings=True,
        )

    # ------------------------------------------------------------------
    # ZK-friendly candidates
    # ------------------------------------------------------------------

    def zk_friendly_candidates(self, n: int = 100) -> list[TransformerConfig]:
        """Sample configs biased toward ZK-friendly design choices.

        Bias factors:
        - Polynomial activations (poly2, poly3, linear).
        - Smaller d_model (lower total MACs).
        - Fewer layers (less depth).
        - Smaller seq_len (less attention cost).

        These candidates are most likely to achieve 10x ZK cost reduction.

        Parameters
        ----------
        n:
            Number of configs to return.

        Returns
        -------
        list[TransformerConfig]
            Valid ZK-friendly configs.
        """
        zk_activations: list[str] = ["poly2", "poly3", "linear"]
        zk_d_models: list[int] = [256, 384, 512]
        zk_n_layers: list[int] = [2, 4, 6, 8]
        zk_seq_lens: list[int] = [64, 128, 256]

        valid_pairs = [
            (d, h) for d in zk_d_models for h in SEARCH_SPACE["n_heads"] if d % h == 0
        ]

        configs: list[TransformerConfig] = []
        for _ in range(n * 20):
            if len(configs) >= n:
                break
            cfg = self._sample_zk_friendly(
                valid_pairs, zk_activations, zk_n_layers, zk_seq_lens
            )
            if cfg is not None:
                configs.append(cfg)

        return configs[:n]

    def _sample_zk_friendly(
        self,
        valid_pairs: list[tuple[int, int]],
        zk_activations: list[str],
        zk_n_layers: list[int],
        zk_seq_lens: list[int],
    ) -> TransformerConfig | None:
        """Sample one ZK-friendly config; return None if the sample is invalid."""

        def _pick(opts: list[Any]) -> Any:
            return opts[self.rng.randint(0, len(opts))]

        d_model, n_heads = valid_pairs[self.rng.randint(0, len(valid_pairs))]
        try:
            return TransformerConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=int(_pick(zk_n_layers)),
                seq_len=int(_pick(zk_seq_lens)),
                ff_mult=float(_pick(SEARCH_SPACE["ff_mult"])),
                activation=str(_pick(zk_activations)),
                use_rotary=bool(_pick(SEARCH_SPACE["use_rotary"])),
                use_layernorm=bool(_pick(SEARCH_SPACE["use_layernorm"])),
                vocab_size=self.vocab_size,
                tie_embeddings=bool(_pick(SEARCH_SPACE["tie_embeddings"])),
            )
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def build_model(self, config: TransformerConfig) -> ZKTransformer:
        """Instantiate a :class:`ZKTransformer` from *config*.

        Parameters
        ----------
        config:
            Architectural hyperparameters.

        Returns
        -------
        ZKTransformer
            Model in evaluation mode.
        """
        model = ZKTransformer(config)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # ZK cost estimation
    # ------------------------------------------------------------------

    def estimate_zk_cost(self, config: TransformerConfig) -> dict[str, int]:
        """Quick ONNX-based ZK cost estimate for *config*.

        Uses ``zkml_bench.metrics.zk_proxy.compute_proxy`` to run the proxy.

        Parameters
        ----------
        config:
            Architectural config to evaluate.

        Returns
        -------
        dict
            Keys: ``constraint_estimate``, ``nonlinear_ops``, ``linear_ops``,
            ``multiplicative_depth``, ``param_count``, ``flops``.
            All values are non-negative integers (or 0 on failure).

        Warns
        -----
        RuntimeWarning
            If the ONNX proxy fails; returns zeros rather than raising.
        """
        try:
            from zkml_bench.metrics.zk_proxy import (  # pylint: disable=import-outside-toplevel
                compute_proxy,
            )
        except ImportError as exc:
            warnings.warn(
                f"zkml_bench not available: {exc}; returning zero costs",
                RuntimeWarning,
                stacklevel=2,
            )
            return _zero_cost_dict()

        model = self.build_model(config)
        sample_input = model.get_sample_input(batch_size=1)

        try:
            result = compute_proxy(model, sample_input)
            return {
                "constraint_estimate": int(result.constraint_estimate),
                "nonlinear_ops": int(result.nonlinear_op_count),
                "linear_ops": int(result.linear_op_count),
                "multiplicative_depth": int(result.multiplicative_depth),
                "param_count": int(result.param_count),
                "flops": int(result.flops),
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            warnings.warn(
                f"ZK proxy computation failed for config {config}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _zero_cost_dict()

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def n_configs(self) -> int:
        """Total number of valid configs in the full Cartesian grid.

        Counts (d_model, n_heads) pairs where d_model % n_heads == 0, then
        multiplies by the product of all other dimension sizes.

        Returns
        -------
        int
            Total valid config count.
        """
        valid_pair_count = sum(
            1
            for d in SEARCH_SPACE["d_model"]
            for h in SEARCH_SPACE["n_heads"]
            if d % h == 0
        )
        other_dims = (
            len(SEARCH_SPACE["n_layers"])
            * len(SEARCH_SPACE["seq_len"])
            * len(SEARCH_SPACE["ff_mult"])
            * len(SEARCH_SPACE["activation"])
            * len(SEARCH_SPACE["use_rotary"])
            * len(SEARCH_SPACE["use_layernorm"])
            * len(SEARCH_SPACE["tie_embeddings"])
        )
        return valid_pair_count * other_dims


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zero_cost_dict() -> dict[str, int]:
    """Return a zeroed cost dict with all expected keys."""
    return {
        "constraint_estimate": 0,
        "nonlinear_ops": 0,
        "linear_ops": 0,
        "multiplicative_depth": 0,
        "param_count": 0,
        "flops": 0,
    }

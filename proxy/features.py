"""Feature extraction for transformer architecture configs.

Extracts a fixed-length numeric feature vector from a :class:`TransformerConfig`
for use by proxy learning models that predict ZK proving cost.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activation encoding
# ---------------------------------------------------------------------------

#: Maps activation name to its integer identifier.
ACTIVATION_IDS: dict[str, int] = {
    "gelu": 0,
    "relu": 1,
    "poly2": 2,
    "poly3": 3,
    "linear": 4,
}

#: Activations that require no lookup tables in ZK circuits.
ZK_FRIENDLY_ACTIVATIONS: frozenset[str] = frozenset({"poly2", "poly3", "linear"})

# Sentinel used for missing ONNX proxy values in the feature vector.
_MISSING_ONNX_SENTINEL: float = -1.0


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:  # pylint: disable=too-many-instance-attributes
    """Architectural hyperparameters for a GPT-2-style transformer."""

    d_model: int
    n_heads: int
    n_layers: int
    seq_len: int
    ff_mult: float = 4.0
    activation: str = "gelu"
    use_rotary: bool = False
    use_layernorm: bool = True
    vocab_size: int = 50257
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATION_IDS:
            raise ValueError(
                f"activation must be one of {sorted(ACTIVATION_IDS)}; "
                f"got {self.activation!r}"
            )
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive; got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive; got {self.n_layers}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive; got {self.seq_len}")
        if self.ff_mult <= 0.0:
            raise ValueError(f"ff_mult must be positive; got {self.ff_mult}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive; got {self.vocab_size}")


# ---------------------------------------------------------------------------
# Internal helpers (split from from_config to keep function complexity low)
# ---------------------------------------------------------------------------


def _compute_arch_features(config: TransformerConfig) -> dict:
    """Compute all architecture-derived features from *config*.

    Returns a dict with keys matching every :class:`ProxyFeatures` field
    except the three ``onnx_*`` fields.
    """
    head_dim: float = config.d_model / config.n_heads

    # Parameter count (analytical; no biases, no positional embedding).
    embedding_params = config.vocab_size * config.d_model
    per_layer_params = _params_per_layer(config)
    lm_head_params = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    total_params = (
        embedding_params + config.n_layers * per_layer_params + lm_head_params
    )

    # MAC estimates.
    attn_macs = _attn_macs(config)
    ffn_macs = _ffn_macs(config)
    total_macs = config.n_layers * (attn_macs + ffn_macs)

    # Activation encoding.
    activation_id = ACTIVATION_IDS[config.activation]
    is_zk_friendly = config.activation in ZK_FRIENDLY_ACTIVATIONS

    # ZK structural counts.
    layernorm_count = config.n_layers * 2 + 1
    softmax_count = config.n_layers if config.activation != "linear" else 0
    gelu_count = config.n_layers if config.activation == "gelu" else 0

    embed_params = config.vocab_size * config.d_model
    pos_embed_params = 0 if config.use_rotary else config.seq_len * config.d_model
    lm_head_params = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    decoder_params = total_params - embed_params - pos_embed_params - lm_head_params

    return {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "seq_len": config.seq_len,
        "ff_mult": config.ff_mult,
        "head_dim": head_dim,
        "total_params": total_params,
        "decoder_params": decoder_params,
        "attn_macs": attn_macs,
        "ffn_macs": ffn_macs,
        "total_macs": total_macs,
        "activation_id": activation_id,
        "is_zk_friendly": is_zk_friendly,
        "layernorm_count": layernorm_count,
        "softmax_count": softmax_count,
        "gelu_count": gelu_count,
    }


def _params_per_layer(config: TransformerConfig) -> int:
    """Return the parameter count for a single transformer block (no biases)."""
    attn = 4 * config.d_model * config.d_model  # Q, K, V, O projections
    ffn = int(2 * config.ff_mult * config.d_model * config.d_model)  # up + down
    norms = 4 * config.d_model  # weight + bias for 2 norm layers
    return attn + ffn + norms


def _attn_macs(config: TransformerConfig) -> int:
    """Return MACs for one attention layer.

    Decomposes as:
    - QKV projection: 2 * seq_len * d_model^2
    - Attention scores + weighted-value accumulation: 2 * seq_len^2 * d_model
    - Output projection: 2 * seq_len * d_model^2
    """
    d, s = config.d_model, config.seq_len
    return 2 * s * d * d + 2 * s * s * d + 2 * s * d * d


def _ffn_macs(config: TransformerConfig) -> int:
    """Return MACs for one FFN layer (two linear layers)."""
    return int(
        2 * config.seq_len * config.d_model * config.ff_mult * config.d_model * 2
    )


def _compute_onnx_fields(
    config: TransformerConfig,
    model: Optional[nn.Module],
) -> dict:
    """Run the ZK proxy on *model* and return a dict with the three ``onnx_*`` keys.

    When *model* is ``None``, all values are ``None``.  Proxy failures are
    surfaced as warnings rather than exceptions so that NAS loops remain robust.
    """
    _empty: dict = {
        "onnx_constraint_estimate": None,
        "onnx_nonlinear_ops": None,
        "onnx_linear_ops": None,
    }

    if model is None:
        return _empty

    try:
        from zkml_bench.metrics.zk_proxy import (  # pylint: disable=import-outside-toplevel
            compute_proxy,
        )

        model.eval()
        sample_input = torch.zeros(1, config.seq_len, dtype=torch.long)
        result = compute_proxy(model, sample_input)
        return {
            "onnx_constraint_estimate": result.constraint_estimate,
            "onnx_nonlinear_ops": result.nonlinear_op_count,
            "onnx_linear_ops": result.linear_op_count,
        }
    except (RuntimeError, ImportError, ValueError) as exc:  # pragma: no cover
        warnings.warn(
            f"ONNX proxy computation failed: {exc}; onnx_* fields set to None",
            stacklevel=3,
        )
        return _empty


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProxyFeatures:  # pylint: disable=too-many-instance-attributes
    """Numeric feature vector derived from a :class:`TransformerConfig`.

    Raw architectural features, derived cost estimates, activation encoding,
    ZK-specific operation counts, and optional ONNX proxy measurements are
    combined into a fixed-length vector via :meth:`to_vector`.
    """

    # --- Raw architectural features ---
    d_model: int
    n_heads: int
    n_layers: int
    seq_len: int
    ff_mult: float
    head_dim: float  # d_model / n_heads

    #: Analytically computed parameter count (no bias terms; see :meth:`from_config`).
    total_params: int

    #: Non-embedding parameter count (total_params minus embedding layers).
    #: Primary accuracy proxy for GPT-2-class NAS (Spearman 0.967 on WikiText-103).
    decoder_params: int

    # --- Derived cost features ---
    #: MACs for one attention layer (QKV proj + scores/values + output proj).
    attn_macs: int
    #: MACs for one FFN layer (two linear layers).
    ffn_macs: int
    #: Total MACs across all layers.
    total_macs: int

    # --- Activation encoding ---
    #: Integer identifier: gelu=0, relu=1, poly2=2, poly3=3, linear=4.
    activation_id: int
    #: True when activation requires no lookup tables (poly2, poly3, linear).
    is_zk_friendly: bool

    # --- ZK-specific structural counts ---
    #: Total LayerNorm / RMSNorm layers (pre-attn + pre-ffn per block + final).
    layernorm_count: int
    #: Attention softmax operations (one per block; 0 for linear attention).
    softmax_count: int
    #: GELU activation count across all blocks.
    gelu_count: int

    # --- ONNX proxy breakdown (populated when a model is provided) ---
    #: Raw constraint estimate from the ZK proxy; None when no model was supplied.
    onnx_constraint_estimate: Optional[int]
    #: Total non-linear ONNX op count; None when no model was supplied.
    onnx_nonlinear_ops: Optional[int]
    #: Total linear ONNX op count; None when no model was supplied.
    onnx_linear_ops: Optional[int]

    def to_vector(self) -> np.ndarray:
        """Return a fixed-length 1-D float64 numpy array for ML models.

        The vector length is always identical regardless of whether the ONNX
        proxy fields are populated.  Missing ONNX values are filled with
        ``-1.0`` as a sentinel.

        Returns
        -------
        np.ndarray
            Shape ``(len(feature_names()),)``, dtype float64.
        """
        onnx_constraint = (
            float(self.onnx_constraint_estimate)
            if self.onnx_constraint_estimate is not None
            else _MISSING_ONNX_SENTINEL
        )
        onnx_nonlinear = (
            float(self.onnx_nonlinear_ops)
            if self.onnx_nonlinear_ops is not None
            else _MISSING_ONNX_SENTINEL
        )
        onnx_linear = (
            float(self.onnx_linear_ops)
            if self.onnx_linear_ops is not None
            else _MISSING_ONNX_SENTINEL
        )

        return np.array(
            [
                float(self.d_model),
                float(self.n_heads),
                float(self.n_layers),
                float(self.seq_len),
                float(self.ff_mult),
                float(self.head_dim),
                float(self.total_params),
                float(self.decoder_params),
                float(self.attn_macs),
                float(self.ffn_macs),
                float(self.total_macs),
                float(self.activation_id),
                float(self.is_zk_friendly),
                float(self.layernorm_count),
                float(self.softmax_count),
                float(self.gelu_count),
                onnx_constraint,
                onnx_nonlinear,
                onnx_linear,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_config(
        cls,
        config: TransformerConfig,
        model: Optional[nn.Module] = None,
    ) -> "ProxyFeatures":
        """Extract features from a :class:`TransformerConfig`.

        Parameters
        ----------
        config:
            The transformer architecture to featurise.
        model:
            Optional PyTorch module.  When provided, the module is exported to
            ONNX and the ZK proxy is computed to populate ``onnx_*`` fields.
            When ``None``, those fields are set to ``None``.

        Returns
        -------
        ProxyFeatures
            Populated feature object.

        Raises
        ------
        ValueError
            If ``config.activation`` is not a recognised value.
        """
        arch = _compute_arch_features(config)
        onnx_fields = _compute_onnx_fields(config, model)
        return cls(**arch, **onnx_fields)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

#: Ordered feature names matching :meth:`ProxyFeatures.to_vector`.
_FEATURE_NAMES: list[str] = [
    "d_model",
    "n_heads",
    "n_layers",
    "seq_len",
    "ff_mult",
    "head_dim",
    "total_params",
    "decoder_params",
    "attn_macs",
    "ffn_macs",
    "total_macs",
    "activation_id",
    "is_zk_friendly",
    "layernorm_count",
    "softmax_count",
    "gelu_count",
    "onnx_constraint_estimate",
    "onnx_nonlinear_ops",
    "onnx_linear_ops",
]


def feature_names() -> list[str]:
    """Return the ordered names of the feature vector produced by :meth:`ProxyFeatures.to_vector`.

    The returned list has the same length as the array returned by
    ``ProxyFeatures.to_vector()``, and each entry names the corresponding
    column, enabling interpretable feature importance analysis.

    Returns
    -------
    list[str]
        A copy of the canonical feature-name list.
    """
    return list(_FEATURE_NAMES)

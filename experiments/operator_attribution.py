"""Operator-level ZK circuit cost attribution for GPT-2-class transformers.

This module generates the key paper figure: a breakdown of where ZK circuit
cost goes (matmuls vs nonlinear ops vs normalization vs embedding) for
GPT-2-class transformer architectures.

Analytic formulas are used instead of ONNX export for clean per-category
attribution.  Costs are in units of Halo2+plookup constraints, scaled by
``OP_COSTS`` (vendored in ``proxy/op_costs.py``).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

# Matplotlib backend must be set before pyplot is imported.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

sys.path.insert(0, str(Path(__file__).parents[1]))

# zkml_bench is a third-party package; proxy is first-party — these imports
# must come after sys.path is mutated, so pylint wrong-import-position is
# suppressed intentionally.
from proxy.op_costs import OP_COSTS  # noqa: E402  pylint: disable=C0413  (vendored table)
from proxy.features import TransformerConfig  # noqa: E402  pylint: disable=C0413

# ---------------------------------------------------------------------------
# OpBreakdown dataclass
# ---------------------------------------------------------------------------


@dataclass
class OpBreakdown:  # pylint: disable=too-many-instance-attributes
    """Cost breakdown by operation category for one model.

    All ``*_constraints`` fields are in the same units as
    ``ZKProxyResult.constraint_estimate`` (Halo2+plookup constraint count).
    """

    config: TransformerConfig
    total_constraints: int

    # Linear ops (matmuls, elementwise mul) — dominate under Halo2+plookup
    attn_qkv_proj_constraints: int  # Q, K, V projection matmuls
    attn_score_constraints: int  # Q@K^T matmuls
    attn_value_constraints: int  # attn@V matmuls
    attn_out_proj_constraints: int  # output projection matmuls
    ffn_up_constraints: int  # FFN up-projection
    ffn_down_constraints: int  # FFN down-projection
    embedding_constraints: int  # token + positional embedding lookups

    # Nonlinear ops — typically small fraction under Halo2+plookup
    softmax_constraints: int  # attention softmax (or 0 if linear attn)
    activation_constraints: int  # GELU/ReLU/poly in FFN
    layernorm_constraints: int  # LayerNorm/RMSNorm
    rope_constraints: int  # Rotary embeddings (sin/cos)

    # Metadata
    linear_fraction: float  # linear_ops / total
    nonlinear_fraction: float  # nonlinear_ops / total

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dict.

        The nested ``config`` field is expanded to its own sub-dict so that
        the output can be round-tripped through :func:`json.dumps` without
        error.
        """
        # asdict recurses into nested dataclasses automatically.
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _OpCosts:  # pylint: disable=too-many-instance-attributes
    """Constraint costs per operation, read from ``OP_COSTS``."""

    matmul: int
    softmax: int
    layernorm: int
    sin: int
    relu: int
    erf: int
    mul: int
    gather: int

    @classmethod
    def from_table(cls) -> "_OpCosts":
        """Populate from the global ``OP_COSTS`` dict."""
        return cls(
            matmul=OP_COSTS.get("MatMul", 1),
            softmax=OP_COSTS.get("Softmax", 32),
            layernorm=OP_COSTS.get("LayerNormalization", 24),
            sin=OP_COSTS.get("Sin", 16),
            relu=OP_COSTS.get("Relu", 2),
            erf=OP_COSTS.get("Erf", 24),
            mul=OP_COSTS.get("Mul", 1),
            gather=OP_COSTS.get("Gather", 2),
        )


def _attention_constraints(
    config: TransformerConfig,
    costs: _OpCosts,
) -> tuple[int, int, int, int, int]:
    """Return per-layer-summed attention constraint counts.

    Returns
    -------
    tuple
        ``(attn_qkv, attn_score, attn_value, attn_out, softmax)`` —
        all already multiplied by ``config.n_layers``.
    """
    d = config.d_model
    s = config.seq_len
    h = config.n_heads
    head_dim = d // h
    nl = config.n_layers

    attn_qkv = nl * 3 * (2 * s * d * d) * costs.matmul
    attn_score = nl * h * (2 * s * s * head_dim) * costs.matmul
    attn_value = nl * h * (2 * s * s * head_dim) * costs.matmul
    attn_out = nl * (2 * s * d * d) * costs.matmul
    softmax = 0 if config.activation == "linear" else nl * h * s * s * costs.softmax

    return attn_qkv, attn_score, attn_value, attn_out, softmax


def _activation_cost_per_elem(activation: str, costs: _OpCosts) -> int:
    """Return constraint cost per FFN activation element for *activation*."""
    if activation == "gelu":
        return costs.erf
    if activation == "relu":
        return costs.relu
    if activation in ("poly2", "poly3"):
        return costs.mul * 2
    # "linear" or unknown: identity, zero cost
    return 0


def _ffn_constraints(
    config: TransformerConfig,
    costs: _OpCosts,
) -> tuple[int, int, int]:
    """Return per-layer-summed FFN constraint counts.

    Returns
    -------
    tuple
        ``(ffn_up, ffn_down, activation)`` — all multiplied by ``n_layers``.
    """
    d = config.d_model
    s = config.seq_len
    ff_hidden = int(config.ff_mult * d)
    nl = config.n_layers

    ffn_up = nl * (2 * s * d * ff_hidden) * costs.matmul
    ffn_down = nl * (2 * s * ff_hidden * d) * costs.matmul
    act_cost = _activation_cost_per_elem(config.activation, costs)
    activation = nl * s * ff_hidden * act_cost
    return ffn_up, ffn_down, activation


def _norm_rope_constraints(
    config: TransformerConfig,
    costs: _OpCosts,
) -> tuple[int, int]:
    """Return layernorm and RoPE constraint counts.

    Returns
    -------
    tuple
        ``(layernorm, rope)``.
    """
    d = config.d_model
    s = config.seq_len
    nl = config.n_layers
    # 2 norm layers per block + 1 final
    layernorm = (2 * nl + 1) * s * d * costs.layernorm
    # Q and K each get sin+cos rotation; approximate as 2 * sin_cost per element.
    rope = (nl * 2 * s * d * costs.sin) if config.use_rotary else 0
    return layernorm, rope


def _embedding_constraints(config: TransformerConfig, costs: _OpCosts) -> int:
    """Return embedding constraint count (token + optional positional)."""
    d = config.d_model
    token_emb = config.vocab_size * d * costs.gather
    pos_emb = 0 if config.use_rotary else config.seq_len * d * costs.gather
    return token_emb + pos_emb


# ---------------------------------------------------------------------------
# Analytic breakdown computation
# ---------------------------------------------------------------------------


def _build_breakdown(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    config: TransformerConfig,
    attn_qkv: int,
    attn_score: int,
    attn_value: int,
    attn_out: int,
    ffn_up: int,
    ffn_down: int,
    embedding: int,
    softmax: int,
    activation: int,
    layernorm: int,
    rope: int,
) -> OpBreakdown:
    """Assemble an :class:`OpBreakdown` from pre-computed component counts."""
    matmul = (
        attn_qkv + attn_score + attn_value + attn_out + ffn_up + ffn_down + embedding
    )
    nonlinear = softmax + activation + layernorm + rope
    total = matmul + nonlinear
    lin_frac = matmul / total if total > 0 else 0.0
    return OpBreakdown(
        config=config,
        total_constraints=total,
        attn_qkv_proj_constraints=attn_qkv,
        attn_score_constraints=attn_score,
        attn_value_constraints=attn_value,
        attn_out_proj_constraints=attn_out,
        ffn_up_constraints=ffn_up,
        ffn_down_constraints=ffn_down,
        embedding_constraints=embedding,
        softmax_constraints=softmax,
        activation_constraints=activation,
        layernorm_constraints=layernorm,
        rope_constraints=rope,
        linear_fraction=lin_frac,
        nonlinear_fraction=nonlinear / total if total > 0 else 0.0,
    )


def compute_op_breakdown(config: TransformerConfig) -> OpBreakdown:
    """Compute operator-level cost breakdown analytically.

    Uses the ``OP_COSTS`` table from ``zkml_bench.metrics.zk_proxy`` to
    compute per-operation costs analytically (no ONNX export needed — cleaner
    attribution).

    Parameters
    ----------
    config:
        Transformer architecture to analyse.

    Returns
    -------
    OpBreakdown
        Per-category constraint estimates and aggregate fractions.
    """
    costs = _OpCosts.from_table()
    attn_qkv, attn_score, attn_value, attn_out, softmax = _attention_constraints(
        config, costs
    )
    ffn_up, ffn_down, activation = _ffn_constraints(config, costs)
    layernorm, rope = _norm_rope_constraints(config, costs)
    return _build_breakdown(
        config,
        attn_qkv,
        attn_score,
        attn_value,
        attn_out,
        ffn_up,
        ffn_down,
        _embedding_constraints(config, costs),
        softmax,
        activation,
        layernorm,
        rope,
    )


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def _matmul_total_from_bd(bd: OpBreakdown) -> int:
    """Sum all linear matmul-derived constraint fields from *bd*."""
    return (
        bd.attn_qkv_proj_constraints
        + bd.attn_score_constraints
        + bd.attn_value_constraints
        + bd.attn_out_proj_constraints
        + bd.ffn_up_constraints
        + bd.ffn_down_constraints
        + bd.embedding_constraints
    )


def breakdown_table(arch_breakdowns: Dict[str, OpBreakdown]) -> str:
    """Return a Markdown table of op breakdowns across architectures.

    Columns: Architecture, Total (billion constraints), MatMul%, Softmax%,
    Activation%, LayerNorm%, RoPE%.

    Parameters
    ----------
    arch_breakdowns:
        Mapping from architecture name to its :class:`OpBreakdown`.

    Returns
    -------
    str
        Multiline Markdown string including header, separator, and data rows.
    """
    header = "| Architecture | Total | MatMul% | Softmax% | Activation% | LayerNorm% | RoPE% |"
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]

    for name, bd in arch_breakdowns.items():
        total = bd.total_constraints
        if total == 0:
            rows.append(f"| {name} | 0 | — | — | — | — | — |")
            continue

        matmul_pct = _matmul_total_from_bd(bd) / total * 100
        softmax_pct = bd.softmax_constraints / total * 100
        act_pct = bd.activation_constraints / total * 100
        ln_pct = bd.layernorm_constraints / total * 100
        rope_pct = bd.rope_constraints / total * 100
        total_b = total / 1e9

        rows.append(
            f"| {name} | {total_b:.1f}B | {matmul_pct:.1f}% | "
            f"{softmax_pct:.1f}% | {act_pct:.1f}% | {ln_pct:.1f}% | {rope_pct:.1f}% |"
        )

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _bd_fractions(bd: OpBreakdown) -> tuple[float, float, float, float, float]:
    """Return (matmul, softmax, activation, layernorm, rope) fractions for *bd*."""
    total = bd.total_constraints
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        _matmul_total_from_bd(bd) / total,
        bd.softmax_constraints / total,
        bd.activation_constraints / total,
        bd.layernorm_constraints / total,
        bd.rope_constraints / total,
    )


def _collect_fractions(
    arch_breakdowns: Dict[str, OpBreakdown],
) -> tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Extract per-category fraction lists from *arch_breakdowns*.

    Returns
    -------
    tuple
        Five equal-length lists: matmul, softmax, activation, layernorm, rope
        fractions (values in [0, 1]).
    """
    rows = [_bd_fractions(bd) for bd in arch_breakdowns.values()]
    col: list[List[float]] = (
        [list(c) for c in zip(*rows)] if rows else [[], [], [], [], []]
    )
    return col[0], col[1], col[2], col[3], col[4]


def plot_cost_breakdown(
    arch_breakdowns: Dict[str, OpBreakdown], output_path: str
) -> None:
    """Stacked bar chart: fractional cost by op category for each architecture.

    X-axis: architecture names.
    Y-axis: fraction of total cost (0–1).
    Stacked bars: matmul (blue), softmax (red), activation (orange),
                  layernorm (green), RoPE (purple).

    The chart illustrates that linear ops (matmuls) dominate activation cost
    under the Halo2+plookup constraint model.

    Parameters
    ----------
    arch_breakdowns:
        Mapping from architecture name to :class:`OpBreakdown`.
    output_path:
        Absolute or relative path for the output PNG file.
    """
    arch_names = list(arch_breakdowns.keys())
    fracs_tuple = _collect_fractions(arch_breakdowns)
    categories = list(
        zip(
            ("MatMul", "Softmax", "Activation", "LayerNorm", "RoPE"),
            fracs_tuple,
            ("#4C72B0", "#DD4444", "#FF8C00", "#2CA02C", "#9467BD"),
        )
    )
    n = len(arch_names)
    x_pos = list(range(n))
    fig, ax = plt.subplots(figsize=(max(8, n * 1.6), 6))
    bottoms = [0.0] * n

    for label, fracs, color in categories:
        ax.bar(x_pos, fracs, bottom=bottoms, label=label, color=color)
        bottoms = [b + f for b, f in zip(bottoms, fracs)]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(arch_names, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Fraction of total ZK constraint cost")
    ax.set_title("ZK Circuit Cost Breakdown by Operation Category\n(Halo2+plookup)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from search.transformer_space import TransformerSearchSpace  # noqa: F401

    space = TransformerSearchSpace()

    _configs = {
        "GPT-2 baseline": TransformerConfig(
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
        ),
        "GPT-2 + linear attn": TransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            seq_len=1024,
            ff_mult=4.0,
            activation="linear",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=50257,
            tie_embeddings=True,
        ),
        "GPT-2 + ReLU FFN": TransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            seq_len=1024,
            ff_mult=4.0,
            activation="relu",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=50257,
            tie_embeddings=True,
        ),
        "compact ZK (512/8L/256)": TransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=8,
            seq_len=256,
            ff_mult=4.0,
            activation="poly2",
            use_rotary=True,
            use_layernorm=False,
            vocab_size=50257,
            tie_embeddings=True,
        ),
        "tiny ZK (384/6L/256)": TransformerConfig(
            d_model=384,
            n_heads=6,
            n_layers=6,
            seq_len=256,
            ff_mult=4.0,
            activation="poly2",
            use_rotary=True,
            use_layernorm=False,
            vocab_size=50257,
            tie_embeddings=True,
        ),
    }

    OUTPUT_DIR = Path("experiments/results/op_attribution")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _breakdowns = {name: compute_op_breakdown(cfg) for name, cfg in _configs.items()}
    print(breakdown_table(_breakdowns))
    plot_cost_breakdown(_breakdowns, str(OUTPUT_DIR / "op_breakdown.png"))

    (OUTPUT_DIR / "op_breakdown.json").write_text(
        json.dumps({k: v.to_dict() for k, v in _breakdowns.items()}, indent=2)
    )
    print(f"Saved to {OUTPUT_DIR}/")

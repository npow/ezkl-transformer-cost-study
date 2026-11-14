#!/usr/bin/env python3
"""Random-search and hand-designed baselines for ZK-NAS comparison.

Generates the critical NAS comparison figure: ZK-NAS (active learning) vs
Random Search vs hand-designed baselines, showing hypervolume vs number of
evaluated configs.  This answers the key reviewer question: "does ZK-NAS beat
just sampling randomly?"

Usage::

    python3 experiments/run_random_search_baseline.py

Outputs:
    experiments/results/random_search_baseline/
        random_search_results.json
        hypervolume_comparison.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — must come before first-party imports so that the project
# root is on sys.path regardless of the working directory.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parents[1]))

# Matplotlib Agg backend must be selected before pyplot is imported.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from experiments.operator_attribution import (  # noqa: E402
    compute_op_breakdown,
)
from proxy.accuracy_proxies import decoder_params_score  # noqa: E402
from proxy.features import TransformerConfig  # noqa: E402
from search.transformer_space import TransformerSearchSpace  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directory containing Phase 1 checkpoint files.
_PHASE1_DIR = Path(__file__).parent / "results" / "nas_phase1"

#: Number of configs evaluated per Phase 1 iteration (batch_size=8).
_PHASE1_BATCH_SIZE = 8

#: Default vocab size used across all configs.
_VOCAB_SIZE = 50257

#: Fixed seq_len for iso-context comparison.
_ISO_SEQ_LEN = 512


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _make_model_points(
    dp_scores: list[float],
    zk_costs: list[int],
    prefix: str = "rs",
) -> list:  # type: ignore[type-arg]
    """Build a list of ``ModelPoint`` objects from parallel score/cost lists.

    Parameters
    ----------
    dp_scores:
        Decoder-params accuracy proxy scores (higher = better).
    zk_costs:
        Analytical ZK constraint counts (lower = cheaper).
    prefix:
        Name prefix for each ``ModelPoint``.

    Returns
    -------
    list
        One ``ModelPoint`` per entry in *dp_scores* / *zk_costs*.
    """
    from zkml_bench.metrics.pareto import (  # pylint: disable=import-outside-toplevel
        ModelPoint,
    )

    return [
        ModelPoint(
            name=f"{prefix}_{j}",
            accuracy_mean=dp_scores[j],
            accuracy_std=None,
            cost=float(zk_costs[j]),
        )
        for j in range(len(dp_scores))
    ]


def run_random_search(  # pylint: disable=too-many-locals
    n_configs: int = 160,
    seed: int = 99,
    seq_len_fixed: int = _ISO_SEQ_LEN,
) -> dict:
    """Random-search baseline: sample configs uniformly, compute analytical ZK cost.

    Configs are drawn uniformly from the full search space (same as ZK-NAS
    Phase 1) but with ``seq_len`` forced to *seq_len_fixed* for a fair
    iso-context comparison.  After each config is evaluated the Pareto
    hypervolume of the current frontier is updated, producing the
    ``hypervolume_curve``.

    Parameters
    ----------
    n_configs:
        Total number of configs to evaluate (budget).  Defaults to 160,
        the same budget used by ZK-NAS Phase 1.
    seed:
        Random seed for reproducible sampling.  Use a seed different from
        the ZK-NAS seed (42) to ensure an independent comparison.
    seq_len_fixed:
        ``seq_len`` forced onto every sampled config for iso-context fairness.

    Returns
    -------
    dict
        Keys:

        * ``configs`` — list of :class:`~proxy.features.TransformerConfig`
          objects in evaluation order.
        * ``zk_costs`` — list of integer analytical ZK constraint counts.
        * ``dp_scores`` — list of ``decoder_params_score`` floats.
        * ``hypervolume_curve`` — list of ``(n_evaluated, hypervolume)``
          tuples, one entry per evaluated config.
        * ``final_hypervolume`` — scalar hypervolume after *n_configs* evals.
        * ``pareto_size`` — number of configs on the final Pareto frontier.
    """
    from zkml_bench.metrics.pareto import (  # pylint: disable=import-outside-toplevel
        compute_pareto,
    )

    space = TransformerSearchSpace(seed=seed, vocab_size=_VOCAB_SIZE)

    configs: list[TransformerConfig] = []
    zk_costs: list[int] = []
    dp_scores: list[float] = []
    hypervolume_curve: list[tuple[int, float]] = []

    for i in range(n_configs):
        cfg = _sample_with_fixed_seq_len(space, seq_len_fixed)
        zk_costs.append(int(compute_op_breakdown(cfg).total_constraints))
        dp_scores.append(decoder_params_score(cfg))
        configs.append(cfg)

        # Compute Pareto hypervolume over all configs evaluated so far.
        pts = _make_model_points(dp_scores, zk_costs)
        hv = float(compute_pareto(pts, cost_metric="constraint_estimate").hypervolume)
        hypervolume_curve.append((i + 1, hv))

    final_hypervolume = hypervolume_curve[-1][1] if hypervolume_curve else 0.0
    final_pts = _make_model_points(dp_scores, zk_costs)
    pareto_size = len(
        compute_pareto(final_pts, cost_metric="constraint_estimate").frontier
    )

    return {
        "configs": configs,
        "zk_costs": zk_costs,
        "dp_scores": dp_scores,
        "hypervolume_curve": hypervolume_curve,
        "final_hypervolume": final_hypervolume,
        "pareto_size": pareto_size,
    }


def _sample_with_fixed_seq_len(
    space: TransformerSearchSpace,
    seq_len: int,
) -> TransformerConfig:
    """Sample one config from *space*, overriding its ``seq_len``.

    Parameters
    ----------
    space:
        Initialised :class:`~search.transformer_space.TransformerSearchSpace`.
    seq_len:
        ``seq_len`` value to force onto the sampled config.

    Returns
    -------
    TransformerConfig
        Valid config with ``seq_len`` pinned to *seq_len*.
    """
    raw = space.random_config()
    return TransformerConfig(
        d_model=raw.d_model,
        n_heads=raw.n_heads,
        n_layers=raw.n_layers,
        seq_len=seq_len,
        ff_mult=raw.ff_mult,
        activation=raw.activation,
        use_rotary=raw.use_rotary,
        use_layernorm=raw.use_layernorm,
        vocab_size=raw.vocab_size,
        tie_embeddings=raw.tie_embeddings,
    )


def run_hand_designed_baselines() -> dict:
    """Evaluate four hand-designed matmul-reduction strategies at seq_len=512.

    Each strategy progressively shrinks model capacity to reduce ZK circuit
    cost while maintaining reasonable accuracy (as predicted by
    ``decoder_params_score``).  The four baselines represent common intuitive
    approaches a practitioner might try before running NAS:

    1. **width-only** — fix depth at 12 layers, shrink ``d_model``.
    2. **depth-only** — fix ``d_model=768``, reduce ``n_layers``.
    3. **proportional** — reduce both ``d_model`` and ``n_layers``
       proportionally to keep the depth-to-width ratio constant.
    4. **arch-optimal** — manually chosen best architecture (``384d/8L/relu``),
       informed by the ZK cost analysis in ``experiments/operator_attribution.py``.

    Returns
    -------
    dict
        Maps strategy name → sub-dict with keys:

        * ``config`` — :class:`~proxy.features.TransformerConfig`.
        * ``zk_cost`` — integer analytical constraint count.
        * ``dp_score`` — ``decoder_params_score`` float.
    """
    baseline_configs: dict[str, TransformerConfig] = {
        "width-only (12L/256d)": TransformerConfig(
            d_model=256,
            n_heads=4,
            n_layers=12,
            seq_len=_ISO_SEQ_LEN,
            ff_mult=4.0,
            activation="gelu",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=_VOCAB_SIZE,
            tie_embeddings=True,
        ),
        "depth-only (768d/4L)": TransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=4,
            seq_len=_ISO_SEQ_LEN,
            ff_mult=4.0,
            activation="gelu",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=_VOCAB_SIZE,
            tie_embeddings=True,
        ),
        "proportional (512d/6L)": TransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
            seq_len=_ISO_SEQ_LEN,
            ff_mult=4.0,
            activation="gelu",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=_VOCAB_SIZE,
            tie_embeddings=True,
        ),
        "arch-optimal (384d/8L/relu)": TransformerConfig(
            d_model=384,
            n_heads=6,
            n_layers=8,
            seq_len=_ISO_SEQ_LEN,
            ff_mult=4.0,
            activation="relu",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=_VOCAB_SIZE,
            tie_embeddings=True,
        ),
    }

    results: dict[str, dict] = {}
    for name, cfg in baseline_configs.items():
        zk_cost = int(compute_op_breakdown(cfg).total_constraints)
        dp = decoder_params_score(cfg)
        results[name] = {
            "config": cfg,
            "zk_cost": zk_cost,
            "dp_score": dp,
        }

    return results


def load_zknas_curve(
    phase1_dir: Optional[Path] = None,
    batch_size: int = _PHASE1_BATCH_SIZE,
) -> list[tuple[int, float]]:
    """Load the ZK-NAS hypervolume curve from Phase 1 checkpoint files.

    Scans *phase1_dir* for files matching ``phase1_iter{N:04d}.json``,
    reads the pre-computed ``hypervolume`` field, and pairs it with the
    cumulative number of evaluated configs ``(iteration + 1) * batch_size``.

    Parameters
    ----------
    phase1_dir:
        Directory containing ``phase1_iter*.json`` files.  Defaults to
        ``experiments/results/nas_phase1/``.
    batch_size:
        Number of configs evaluated per iteration (default 8).

    Returns
    -------
    list[tuple[int, float]]
        Sorted list of ``(n_evaluated, hypervolume)`` pairs.  Empty list if
        *phase1_dir* does not exist or contains no matching checkpoints.
    """
    if phase1_dir is None:
        phase1_dir = _PHASE1_DIR

    checkpoint_files = sorted(phase1_dir.glob("phase1_iter*.json"))
    if not checkpoint_files:
        logger.warning(
            "No Phase 1 checkpoint files found in %s; returning empty curve.",
            phase1_dir,
        )
        return []

    curve: list[tuple[int, float]] = []
    for ckpt_path in checkpoint_files:
        try:
            payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read checkpoint %s: %s", ckpt_path, exc)
            continue

        iteration: int = int(payload.get("iteration", 0))
        hypervolume: float = float(payload.get("hypervolume", 0.0))
        n_evaluated = (iteration + 1) * batch_size
        curve.append((n_evaluated, hypervolume))

    return sorted(curve)


def compute_search_efficiency(
    zknas_final_hv: float,
    random_curve: list[tuple[int, float]],
) -> Optional[int]:
    """Find how many random-search configs achieve ZK-NAS's final hypervolume.

    Scans *random_curve* in order and returns the first ``n_evaluated``
    value where the random-search hypervolume exceeds or equals
    *zknas_final_hv*.  Returns ``None`` when random search never reaches
    that threshold.

    Parameters
    ----------
    zknas_final_hv:
        Final hypervolume achieved by ZK-NAS.
    random_curve:
        ``(n_evaluated, hypervolume)`` pairs from random search.

    Returns
    -------
    int or None
        Minimum configs needed for random search to match ZK-NAS hypervolume.
    """
    for n_evaluated, hv in random_curve:
        if hv >= zknas_final_hv:
            return n_evaluated
    return None


def plot_hypervolume_comparison(
    zknas_curve: list[tuple[int, float]],
    random_curve: list[tuple[int, float]],
    output_path: str,
) -> None:
    """Plot hypervolume vs n_configs_evaluated for ZK-NAS vs Random Search.

    Both curves are normalised by the final random-search hypervolume so the
    Y-axis is unit-free and directly comparable across budget levels.  This
    is the key paper figure showing that active learning finds the Pareto
    frontier faster than uniform random sampling.

    Parameters
    ----------
    zknas_curve:
        ``(n_evaluated, hypervolume)`` pairs for ZK-NAS (active learning).
    random_curve:
        ``(n_evaluated, hypervolume)`` pairs for random search.
    output_path:
        Absolute or relative path for the output PNG file.
    """
    if not random_curve:
        logger.warning("random_curve is empty; skipping plot.")
        return

    # Normalise by random search's final hypervolume.
    final_random_hv = random_curve[-1][1]
    norm = final_random_hv if final_random_hv > 0 else 1.0

    def _norm_curve(
        curve: list[tuple[int, float]],
    ) -> tuple[list[int], list[float]]:
        xs = [n for n, _ in curve]
        ys = [hv / norm for _, hv in curve]
        return xs, ys

    zk_xs, zk_ys = _norm_curve(zknas_curve)
    rs_xs, rs_ys = _norm_curve(random_curve)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(zk_xs, zk_ys, color="#2166AC", linewidth=2, linestyle="-", label="ZK-NAS (active learning)")
    ax.plot(rs_xs, rs_ys, color="#D6604D", linewidth=2, linestyle="--", label="Random Search")

    ax.set_xlabel("Number of configs evaluated", fontsize=12)
    ax.set_ylabel("Pareto hypervolume (normalised)", fontsize=12)
    ax.set_title(
        "ZK-NAS vs Random Search: Hypervolume vs Evaluation Budget\n"
        "(higher is better; normalised by random search final hypervolume)",
        fontsize=11,
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hypervolume comparison plot to %s", output_path)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: TransformerConfig) -> dict:
    """Serialise a :class:`TransformerConfig` to a plain dict."""
    return {
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "seq_len": cfg.seq_len,
        "ff_mult": cfg.ff_mult,
        "activation": cfg.activation,
        "use_rotary": cfg.use_rotary,
        "use_layernorm": cfg.use_layernorm,
        "vocab_size": cfg.vocab_size,
        "tie_embeddings": cfg.tie_embeddings,
    }


def _results_to_json_safe(results: dict) -> dict:
    """Return a JSON-serialisable copy of random-search *results*.

    Replaces :class:`TransformerConfig` objects with plain dicts and converts
    ``hypervolume_curve`` tuples to lists.
    """
    return {
        "configs": [_config_to_dict(c) for c in results["configs"]],
        "zk_costs": results["zk_costs"],
        "dp_scores": results["dp_scores"],
        "hypervolume_curve": [list(pt) for pt in results["hypervolume_curve"]],
        "final_hypervolume": results["final_hypervolume"],
        "pareto_size": results["pareto_size"],
    }


def _baselines_to_json_safe(baselines: dict) -> dict:
    """Return a JSON-serialisable copy of *baselines*."""
    out: dict = {}
    for name, entry in baselines.items():
        out[name] = {
            "config": _config_to_dict(entry["config"]),
            "zk_cost": entry["zk_cost"],
            "dp_score": entry["dp_score"],
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    """Entry point for standalone execution."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(__file__).parent / "results" / "random_search_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Random Search ---
    print("Running random search (n=160, seed=99, seq_len=512)...")
    rs_results = run_random_search(n_configs=160, seed=99, seq_len_fixed=512)
    print(f"  Final hypervolume  : {rs_results['final_hypervolume']:.4e}")
    print(f"  Pareto size        : {rs_results['pareto_size']}")

    rs_json_path = output_dir / "random_search_results.json"
    rs_json_path.write_text(json.dumps(_results_to_json_safe(rs_results), indent=2))
    print(f"  Saved -> {rs_json_path}")

    # --- Hand-designed Baselines ---
    print("\nEvaluating hand-designed baselines...")
    hand_baselines = run_hand_designed_baselines()
    for baseline_name, baseline_entry in hand_baselines.items():
        print(
            f"  {baseline_name:35s}  "
            f"zk={baseline_entry['zk_cost']:.3e}  "
            f"dp={baseline_entry['dp_score']:.4f}"
        )

    baselines_json_path = output_dir / "hand_designed_baselines.json"
    baselines_json_path.write_text(
        json.dumps(_baselines_to_json_safe(hand_baselines), indent=2)
    )
    print(f"  Saved -> {baselines_json_path}")

    # --- Load ZK-NAS curve ---
    print("\nLoading ZK-NAS Phase 1 hypervolume curve...")
    zk_curve = load_zknas_curve()
    if zk_curve:
        print(
            f"  Loaded {len(zk_curve)} checkpoints; "
            f"final HV={zk_curve[-1][1]:.4e} at n={zk_curve[-1][0]}"
        )
    else:
        print("  WARNING: No Phase 1 checkpoints found.")

    # --- Efficiency ratio ---
    if zk_curve:
        zk_final_hv = zk_curve[-1][1]
        rs_budget = compute_search_efficiency(
            zk_final_hv, rs_results["hypervolume_curve"]
        )
        if rs_budget is not None:
            ratio = rs_budget / zk_curve[-1][0]
            print(
                f"\nSearch efficiency: random search needs {rs_budget} configs to "
                f"match ZK-NAS ({zk_curve[-1][0]} configs) -> "
                f"{ratio:.2f}x more evaluations."
            )
        else:
            print(
                "\nSearch efficiency: random search never reached ZK-NAS final "
                f"hypervolume ({zk_final_hv:.4e}) within "
                f"{len(rs_results['hypervolume_curve'])} evals."
            )

    # --- Plot ---
    print("\nGenerating hypervolume comparison plot...")
    comparison_plot_path = str(output_dir / "hypervolume_comparison.png")
    plot_hypervolume_comparison(
        zknas_curve=zk_curve,
        random_curve=rs_results["hypervolume_curve"],
        output_path=comparison_plot_path,
    )
    print(f"  Saved -> {comparison_plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    _main()

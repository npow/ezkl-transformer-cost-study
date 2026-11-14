#!/usr/bin/env python3
"""Multi-seed baseline comparison for ZK-NAS.

Runs four search strategies × 10 seeds and compares Pareto hypervolume:

  1. ZK-NAS (EHVI)    — greedy Expected Hypervolume Improvement selection
  2. Random Search    — uniform random sampling (same budget)
  3. Accuracy-only    — top-N by decoder_params_score, ignores ZK cost
  4. Cheapest-only    — top-N by lowest ZK constraint count, ignores quality

Results are stored in::

    experiments/results/multi_seed_baselines/
        results.json
        hv_violin.png

Usage::

    python3 experiments/run_multi_seed_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parents[1]))

# Matplotlib Agg backend must be selected before pyplot is imported.
import matplotlib  # noqa: E402  pylint: disable=wrong-import-position

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from experiments.operator_attribution import compute_op_breakdown  # noqa: E402
from proxy.accuracy_proxies import decoder_params_score  # noqa: E402
from proxy.active_learning import (  # noqa: E402
    _compute_2d_hypervolume,
    _update_pareto_front,
)
from proxy.features import TransformerConfig  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LEN: int = 512
VOCAB: int = 50257
N_CONFIGS: int = 160
N_SEEDS: int = 10
WARMUP_SIZE: int = 10
EHVI_CANDIDATES: int = 500  # candidates sampled per EHVI step (speed/quality trade-off)

_OUTPUT_DIR = Path(__file__).parent / "results" / "multi_seed_baselines"


# ---------------------------------------------------------------------------
# Search-space construction
# ---------------------------------------------------------------------------


def build_iso_configs() -> list[TransformerConfig]:
    """Build the full iso-context configuration grid (3168 configs).

    Enumerates all valid combinations of d_model, n_heads (must divide
    d_model), n_layers, activation, and use_rotary at fixed seq_len/vocab.

    Returns
    -------
    list[TransformerConfig]
        Enumerated search space.
    """
    configs: list[TransformerConfig] = []
    for d_model in [256, 384, 512, 640, 768, 1024]:
        for n_heads in [4, 6, 8, 12, 16]:
            if d_model % n_heads != 0:
                continue
            for n_layers in [2, 4, 6, 8, 10, 12, 16, 20, 24]:
                for activation in ["gelu", "relu", "poly2", "linear"]:
                    for use_rotary in [True, False]:
                        cfg = TransformerConfig(
                            d_model=d_model,
                            n_heads=n_heads,
                            n_layers=n_layers,
                            seq_len=SEQ_LEN,
                            ff_mult=4.0,
                            activation=activation,
                            use_rotary=use_rotary,
                            use_layernorm=True,
                            vocab_size=VOCAB,
                            tie_embeddings=True,
                        )
                        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def precompute_scores(
    configs: list[TransformerConfig],
) -> list[tuple[float, int]]:
    """Pre-compute (accuracy_proxy, zk_constraints) for every config.

    Computes ``decoder_params_score`` and analytical ZK constraint count once
    so all search methods share the same oracle values.

    Parameters
    ----------
    configs:
        Full search-space config list.

    Returns
    -------
    list[tuple[float, int]]
        Parallel list of ``(acc_proxy, zk_cost)`` for each config.
    """
    scores: list[tuple[float, int]] = []
    for cfg in configs:
        acc = decoder_params_score(cfg)
        zk = int(compute_op_breakdown(cfg).total_constraints)
        scores.append((acc, zk))
    return scores


# ---------------------------------------------------------------------------
# Hypervolume helper
# ---------------------------------------------------------------------------


def pareto_hv(
    selected_indices: list[int],
    scores: list[tuple[float, int]],
) -> float:
    """Compute the 2-D hypervolume of a set of selected configs' Pareto front.

    Objectives are ``(acc_proxy, -log(zk_cost))`` — both maximised.

    Parameters
    ----------
    selected_indices:
        Indices into *scores* of the selected configs.
    scores:
        Pre-computed ``(acc, zk_cost)`` for the full search space.

    Returns
    -------
    float
        Hypervolume indicator (>= 0).
    """
    if not selected_indices:
        return 0.0
    pts = [
        (scores[i][0], -np.log(max(scores[i][1], 1))) for i in selected_indices
    ]
    front = _update_pareto_front(pts)
    if not front:
        return 0.0
    ref = (
        min(p[0] for p in front) - 0.1,
        min(p[1] for p in front) - 0.1,
    )
    return _compute_2d_hypervolume(front, ref)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def run_random(
    seed: int,
    scores: list[tuple[float, int]],
) -> float:
    """Random search baseline: sample N_CONFIGS configs uniformly at random.

    Parameters
    ----------
    seed:
        NumPy random seed for reproducibility.
    scores:
        Pre-computed ``(acc, zk_cost)`` for all configs.

    Returns
    -------
    float
        Hypervolume of the randomly selected set's Pareto front.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(scores), size=N_CONFIGS, replace=False).tolist()
    return pareto_hv(idx, scores)


def run_greedy_pareto(
    seed: int,
    scores: list[tuple[float, int]],
) -> float:
    """ZK-NAS EHVI: greedy Expected Hypervolume Improvement selection.

    Starts with WARMUP_SIZE random configs, then iteratively adds the config
    that maximises incremental hypervolume gain, sampling EHVI_CANDIDATES
    candidates per step for speed.

    Parameters
    ----------
    seed:
        NumPy random seed for warmup and candidate sampling.
    scores:
        Pre-computed ``(acc, zk_cost)`` for all configs.

    Returns
    -------
    float
        Hypervolume of the selected set's Pareto front.
    """
    rng = np.random.RandomState(seed)
    n_total = len(scores)

    warmup_idx: list[int] = rng.choice(n_total, size=WARMUP_SIZE, replace=False).tolist()
    selected: list[int] = list(warmup_idx)
    selected_set: set[int] = set(selected)

    while len(selected) < N_CONFIGS:
        current_pts = [
            (scores[i][0], -np.log(max(scores[i][1], 1))) for i in selected
        ]
        front = _update_pareto_front(current_pts)
        if front:
            ref = (
                min(p[0] for p in front) - 0.1,
                min(p[1] for p in front) - 0.1,
            )
        else:
            ref = (-0.1, -0.1)
        current_hv = _compute_2d_hypervolume(front, ref)

        # Sample a pool of candidate indices not yet selected.
        available = [i for i in range(n_total) if i not in selected_set]
        pool_size = min(EHVI_CANDIDATES, len(available))
        if pool_size == 0:
            break
        candidates = rng.choice(available, size=pool_size, replace=False)

        best_gain = -1.0
        best_idx = int(candidates[0])
        for idx in candidates:
            pt = (scores[idx][0], -np.log(max(scores[idx][1], 1)))
            new_pts = current_pts + [pt]
            new_front = _update_pareto_front(new_pts)
            new_hv = _compute_2d_hypervolume(new_front, ref)
            gain = new_hv - current_hv
            if gain > best_gain:
                best_gain = gain
                best_idx = int(idx)

        selected.append(best_idx)
        selected_set.add(best_idx)

    return pareto_hv(selected, scores)


def run_params_only(
    seed: int,
    scores: list[tuple[float, int]],
) -> float:
    """Accuracy-only baseline: top-N by decoder_params_score with random tiebreaking.

    Selects the N_CONFIGS configs with the highest accuracy proxy, ignoring
    ZK cost entirely.

    Parameters
    ----------
    seed:
        NumPy random seed for random tiebreaking.
    scores:
        Pre-computed ``(acc, zk_cost)`` for all configs.

    Returns
    -------
    float
        Hypervolume of the selected set's Pareto front.
    """
    rng = np.random.RandomState(seed)
    noise = rng.uniform(0.0, 1e-6, size=len(scores))
    sorted_idx = sorted(
        range(len(scores)), key=lambda i: scores[i][0] + noise[i]
    )
    selected = sorted_idx[-N_CONFIGS:]
    return pareto_hv(selected, scores)


def run_cheapest_only(
    seed: int,
    scores: list[tuple[float, int]],
) -> float:
    """Cheapest-only baseline: top-N by lowest ZK constraint count.

    Selects the N_CONFIGS configs with the smallest ZK circuit cost, ignoring
    accuracy proxy entirely.

    Parameters
    ----------
    seed:
        NumPy random seed for random tiebreaking.
    scores:
        Pre-computed ``(acc, zk_cost)`` for all configs.

    Returns
    -------
    float
        Hypervolume of the selected set's Pareto front.
    """
    rng = np.random.RandomState(seed)
    noise = rng.uniform(0.0, 1e8, size=len(scores))
    sorted_idx = sorted(
        range(len(scores)), key=lambda i: scores[i][1] + noise[i]
    )
    selected = sorted_idx[:N_CONFIGS]
    return pareto_hv(selected, scores)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def mann_whitney_vs_zknas(
    zknas_hvs: list[float],
    baseline_hvs: list[float],
) -> tuple[float, float]:
    """One-sided Mann-Whitney U test: ZK-NAS > baseline.

    Parameters
    ----------
    zknas_hvs:
        Hypervolume samples for ZK-NAS.
    baseline_hvs:
        Hypervolume samples for the baseline method.

    Returns
    -------
    tuple[float, float]
        ``(U_statistic, p_value)`` for the one-sided alternative
        that ZK-NAS produces larger HV than the baseline.
    """
    stat, p_val = mannwhitneyu(
        zknas_hvs, baseline_hvs, alternative="greater"
    )
    return float(stat), float(p_val)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_violin(
    method_names: list[str],
    method_hvs: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Violin plot of HV distributions for all methods.

    Parameters
    ----------
    method_names:
        Ordered list of method display names.
    method_hvs:
        Maps method name -> list of HV values (one per seed).
    output_path:
        Path to save the PNG.
    """
    data = [method_hvs[name] for name in method_names]
    colors = ["#2166AC", "#D6604D", "#F4A582", "#B2182B"]

    fig, ax = plt.subplots(figsize=(9, 5))

    parts = ax.violinplot(
        data,
        positions=range(len(method_names)),
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    # Colour each violin body.
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    for part_name in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.2)

    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylabel("Pareto Hypervolume", fontsize=11)
    ax.set_title(
        f"ZK-NAS vs Baselines: Pareto HV Distribution\n"
        f"(N={N_CONFIGS} configs, {N_SEEDS} seeds, "
        f"search space ≈3168 configs)",
        fontsize=11,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved violin plot to %s", output_path)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_results_table(
    method_names: list[str],
    method_hvs: dict[str, list[float]],
    zknas_key: str,
) -> None:
    """Print a results table with mean ± std HV and Mann-Whitney p-values.

    Parameters
    ----------
    method_names:
        Ordered list of method display names.
    method_hvs:
        Maps method name -> list of HV values.
    zknas_key:
        Key in *method_hvs* for the ZK-NAS method (used as reference).
    """
    zknas_hvs = method_hvs[zknas_key]
    col_w = 20

    header = (
        f"{'Method':<{col_w}}  {'Mean HV':>14}  {'Std HV':>12}  "
        f"{'Min HV':>12}  {'Max HV':>12}  {'p-value (vs ZK-NAS)':>22}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for name in method_names:
        hvs = np.array(method_hvs[name])
        mean_hv = float(np.mean(hvs))
        std_hv = float(np.std(hvs, ddof=1))
        min_hv = float(np.min(hvs))
        max_hv = float(np.max(hvs))

        if name == zknas_key:
            p_str = "—  (reference)"
        else:
            _, p_val = mann_whitney_vs_zknas(list(zknas_hvs), list(hvs))
            sig = " *" if p_val < 0.05 else "  "
            p_str = f"{p_val:.4f}{sig}"

        print(
            f"{name:<{col_w}}  {mean_hv:>14.4e}  {std_hv:>12.4e}  "
            f"{min_hv:>12.4e}  {max_hv:>12.4e}  {p_str:>22}"
        )

    print(sep)
    print("* p < 0.05 (one-sided Mann-Whitney U: ZK-NAS > baseline)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build search space                                                   #
    # ------------------------------------------------------------------ #
    print("Building iso-context search space...", flush=True)
    all_configs = build_iso_configs()
    print(f"Search space: {len(all_configs)} configs", flush=True)

    # ------------------------------------------------------------------ #
    # Pre-compute (acc, zk_cost) for all configs once                     #
    # ------------------------------------------------------------------ #
    print("Pre-computing scores for all configs...", flush=True)
    t0 = time.perf_counter()
    scores = precompute_scores(all_configs)
    print(
        f"Pre-computation done in {time.perf_counter() - t0:.1f}s", flush=True
    )

    # ------------------------------------------------------------------ #
    # Define strategies                                                    #
    # ------------------------------------------------------------------ #
    methods: dict[str, Callable[[int, list[tuple[float, int]]], float]] = {
        "ZK-NAS (EHVI)": run_greedy_pareto,
        "Random Search": run_random,
        "Accuracy-only": run_params_only,
        "Cheapest-only": run_cheapest_only,
    }
    method_names = list(methods.keys())
    zknas_key = "ZK-NAS (EHVI)"

    # ------------------------------------------------------------------ #
    # Run all methods × all seeds                                          #
    # ------------------------------------------------------------------ #
    method_hvs: dict[str, list[float]] = {name: [] for name in method_names}

    for name, fn in methods.items():
        t_method = time.perf_counter()
        print(f"\nRunning {name} ({N_SEEDS} seeds)...", flush=True)
        for seed in range(N_SEEDS):
            t_seed = time.perf_counter()
            hv = fn(seed, scores)
            method_hvs[name].append(hv)
            elapsed = time.perf_counter() - t_seed
            print(
                f"  seed={seed:2d}  HV={hv:.4e}  ({elapsed:.1f}s)",
                flush=True,
            )
        total_t = time.perf_counter() - t_method
        print(
            f"  => mean={np.mean(method_hvs[name]):.4e}  "
            f"std={np.std(method_hvs[name], ddof=1):.4e}  "
            f"(total {total_t:.1f}s)",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Print summary table                                                  #
    # ------------------------------------------------------------------ #
    print("\n\nRESULTS TABLE")
    print_results_table(method_names, method_hvs, zknas_key)

    # ------------------------------------------------------------------ #
    # Violin plot                                                          #
    # ------------------------------------------------------------------ #
    plot_path = _OUTPUT_DIR / "hv_violin.png"
    plot_violin(method_names, method_hvs, plot_path)
    print(f"Violin plot saved: {plot_path}", flush=True)

    # ------------------------------------------------------------------ #
    # Save JSON                                                            #
    # ------------------------------------------------------------------ #
    summary: dict[str, object] = {}
    for name in method_names:
        hvs = np.array(method_hvs[name])
        baseline_hvs = method_hvs[name]
        if name != zknas_key:
            u_stat, p_val = mann_whitney_vs_zknas(
                method_hvs[zknas_key], baseline_hvs
            )
        else:
            u_stat, p_val = None, None  # type: ignore[assignment]

        summary[name] = {
            "hvs": [float(v) for v in hvs],
            "mean": float(np.mean(hvs)),
            "std": float(np.std(hvs, ddof=1)),
            "min": float(np.min(hvs)),
            "max": float(np.max(hvs)),
            "mann_whitney_u": u_stat,
            "p_value_vs_zknas": p_val,
        }

    json_path = _OUTPUT_DIR / "results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results JSON saved: {json_path}", flush=True)


if __name__ == "__main__":
    _main()

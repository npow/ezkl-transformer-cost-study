#!/usr/bin/env python3
"""Matched-Cost Ablation: Is ZK-NAS finding genuinely better architectures?

Gap #2 for main-track submission.  At the *same* EZKL row budget, compare
ZK-NAS discovered activations against naive practitioner choices.

Experiment design
-----------------
Two budget tiers are evaluated:

  ~11x cheaper than GPT-2 (~34B EZKL rows):
    zk_11x_linear   -- ZK-NAS pick: 384d/4L, linear attention, rotary
    naive_11x_gelu  -- 384d/4L, GELU, no rotary  (naive practitioner choice)
    naive_11x_relu  -- 384d/4L, ReLU, rotary      (common ZK-ML advice)
    zk_11x_poly2    -- 384d/4L, poly2, rotary      (another ZK-friendly pick)

  ~3x cheaper than GPT-2 (~118B EZKL rows):
    zk_3x_relu      -- ZK-NAS pick: 512d/8L, relu, rotary
    naive_3x_gelu   -- 512d/8L, GELU, no rotary   (naive practitioner choice)

Key finding:
  - If naive_11x_gelu beats zk_11x_linear: ZK-friendly activations HURT quality
    with no ZK benefit (activation choice matters more than expected).
  - If they are similar: shape dominates, activation choice is largely irrelevant.
  - Either result is publishable as the paper's core ablation.

Usage::

    python experiments/run_matched_cost_ablation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — must come first so local packages resolve correctly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[1]))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from experiments.operator_attribution import compute_op_breakdown  # noqa: E402
from proxy.features import TransformerConfig  # noqa: E402
from search.lm_trainer import LMTrainer, LMTrainingConfig  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

SEQ_LEN: int = 256
VOCAB: int = 50257

#: EZKL row calibration factor (constraints × factor → estimated EZKL rows).
EZKL_ROW_FACTOR: float = 3.83

MAX_STEPS: int = 3000
BATCH_SIZE: int = 8
EVAL_STEPS: int = 1000
MAX_EVAL_BATCHES: int = 20

OUTPUT_DIR = Path(__file__).parent / "results" / "matched_cost_ablation"

# ---------------------------------------------------------------------------
# Architecture configs
# ---------------------------------------------------------------------------

#: All configs to benchmark.  Positional args match TransformerConfig order:
#: d_model, n_heads, n_layers, seq_len, ff_mult, activation,
#: use_rotary, use_layernorm, vocab_size, tie_embeddings.
CONFIGS: dict[str, TransformerConfig] = {
    # ~11x cheaper than GPT-2 — same shape (384d/4L/6h), different activations
    "zk_11x_linear": TransformerConfig(384, 6, 4, SEQ_LEN, 4.0, "linear", True,  True, VOCAB, True),
    "naive_11x_gelu": TransformerConfig(384, 6, 4, SEQ_LEN, 4.0, "gelu",  False, True, VOCAB, True),
    "naive_11x_relu": TransformerConfig(384, 6, 4, SEQ_LEN, 4.0, "relu",  True,  True, VOCAB, True),
    "zk_11x_poly2":  TransformerConfig(384, 6, 4, SEQ_LEN, 4.0, "poly2", True,  True, VOCAB, True),
    # ~3x cheaper than GPT-2 — same shape (512d/8L/8h), different activations
    "zk_3x_relu":    TransformerConfig(512, 8, 8, SEQ_LEN, 4.0, "relu",  True,  True, VOCAB, True),
    "naive_3x_gelu": TransformerConfig(512, 8, 8, SEQ_LEN, 4.0, "gelu",  False, True, VOCAB, True),
}

#: Human-readable shape label for the table.
_SHAPE_LABELS: dict[str, str] = {
    "zk_11x_linear":  "384d/4L",
    "naive_11x_gelu": "384d/4L",
    "naive_11x_relu": "384d/4L",
    "zk_11x_poly2":   "384d/4L",
    "zk_3x_relu":     "512d/8L",
    "naive_3x_gelu":  "512d/8L",
}

#: Which config is the "naive baseline" for relative PPL comparison, per tier.
_NAIVE_BASELINE: dict[str, str] = {
    "zk_11x_linear":  "naive_11x_gelu",
    "naive_11x_gelu": "naive_11x_gelu",
    "naive_11x_relu": "naive_11x_gelu",
    "zk_11x_poly2":   "naive_11x_gelu",
    "zk_3x_relu":     "naive_3x_gelu",
    "naive_3x_gelu":  "naive_3x_gelu",
}


# ---------------------------------------------------------------------------
# ZK cost helpers
# ---------------------------------------------------------------------------


def compute_zk_metrics(cfg: TransformerConfig) -> tuple[int, float]:
    """Return ``(total_constraints, ezkl_rows_est)`` for *cfg*.

    Parameters
    ----------
    cfg:
        Transformer architecture to evaluate.

    Returns
    -------
    tuple[int, float]
        Raw constraint count and calibrated EZKL row estimate.
    """
    breakdown = compute_op_breakdown(cfg)
    zk_cost = breakdown.total_constraints
    ezkl_rows = zk_cost * EZKL_ROW_FACTOR
    return zk_cost, ezkl_rows


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_one(
    name: str,
    cfg: TransformerConfig,
    trainer: LMTrainer,
) -> dict[str, Any]:
    """Train *cfg* with *trainer* and return a metrics dict.

    Parameters
    ----------
    name:
        Config identifier (used for logging only).
    cfg:
        Transformer architecture to train.
    trainer:
        Pre-configured :class:`~search.lm_trainer.LMTrainer`.

    Returns
    -------
    dict[str, Any]
        Keys: ``val_ppl``, ``val_loss``, ``zk_cost``, ``ezkl_rows_est``,
        ``train_time_s``, ``steps_run``, ``perplexity_curve``.
    """
    logger.info("Building model for %s", name)
    model = trainer.build_and_initialize(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    zk_cost, ezkl_rows = compute_zk_metrics(cfg)
    logger.info("  ZK constraints: %s  EZKL rows: %.1fB", f"{zk_cost:,}", ezkl_rows / 1e9)

    t0 = time.time()
    result = trainer.train(model)
    elapsed = time.time() - t0

    return {
        "val_ppl": result.val_perplexity,
        "val_loss": result.val_loss,
        "best_val_ppl": result.best_val_perplexity,
        "final_train_loss": result.final_train_loss,
        "zk_cost": zk_cost,
        "ezkl_rows_est": ezkl_rows,
        "train_time_s": elapsed,
        "steps_run": result.steps_run,
        "n_params": n_params,
        "perplexity_curve": result.perplexity_curve,
    }


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def _pct_vs_naive(val_ppl: float, naive_ppl: float) -> str:
    """Return a signed percentage string of PPL change relative to *naive_ppl*.

    Negative means *val_ppl* is better (lower PPL).
    """
    if naive_ppl == 0 or naive_ppl is None:
        return "N/A"
    delta_pct = (val_ppl - naive_ppl) / naive_ppl * 100.0
    return f"{delta_pct:+.1f}%"


def print_results_table(
    results: dict[str, dict[str, Any]],
    configs: dict[str, TransformerConfig],
) -> None:
    """Print a formatted ablation results table to stdout.

    Parameters
    ----------
    results:
        Mapping from config name to metrics dict (output of :func:`run_one`).
    configs:
        Architecture configs, used for activation labels.
    """
    print()
    print("=" * 78)
    print("Matched-Cost Ablation Results (3000 steps, seq=256, random init):")
    print("=" * 78)
    header = (
        f"{'Config':<20} | {'Shape':<9} | {'Activation':<10} | "
        f"{'PPL':>7} | {'ZK rows':>9} | {'vs naive':>9} |"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name, row in results.items():
        cfg = configs[name]
        shape = _SHAPE_LABELS.get(name, "?")
        act = cfg.activation
        ppl = row["val_ppl"]
        ezkl_b = row["ezkl_rows_est"] / 1e9

        naive_key = _NAIVE_BASELINE.get(name)
        if naive_key == name:
            vs_str = "baseline"
        elif naive_key and naive_key in results:
            naive_ppl = results[naive_key]["val_ppl"]
            vs_str = _pct_vs_naive(ppl, naive_ppl)
        else:
            vs_str = "N/A"

        print(
            f"{name:<20} | {shape:<9} | {act:<10} | "
            f"{ppl:>7.1f} | {ezkl_b:>8.1f}B | {vs_str:>9} |"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the matched-cost ablation and save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_cfg = LMTrainingConfig(
        max_steps=MAX_STEPS,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        eval_steps=EVAL_STEPS,
        max_eval_batches=MAX_EVAL_BATCHES,
        device="cpu",
        use_pretrained=False,
        freeze_embeddings=False,
    )

    trainer = LMTrainer(training_cfg)

    print()
    print("=" * 60)
    print("ZK-NAS Matched-Cost Ablation")
    print(f"  max_steps={MAX_STEPS}  seq_len={SEQ_LEN}  batch_size={BATCH_SIZE}")
    print(f"  eval_steps={EVAL_STEPS}  max_eval_batches={MAX_EVAL_BATCHES}")
    print(f"  use_pretrained=False  (random init)")
    print("=" * 60)
    print()

    results: dict[str, dict[str, Any]] = {}

    for name, cfg in CONFIGS.items():
        print(f"\n{'—' * 60}")
        print(f"Config: {name}  ({cfg.activation}, rotary={cfg.use_rotary})")
        print(f"{'—' * 60}")
        try:
            results[name] = run_one(name, cfg, trainer)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to train config %s", name)
            raise

        # Save intermediate results after each config completes.
        _save_results(results, OUTPUT_DIR)

    _save_results(results, OUTPUT_DIR)
    print_results_table(results, CONFIGS)
    _print_findings(results)


def _save_results(results: dict[str, dict[str, Any]], out_dir: Path) -> None:
    """Persist *results* to JSON, making perplexity curves serialisable."""
    serialisable = {}
    for name, row in results.items():
        serialisable[name] = dict(row)
        # perplexity_curve is list[tuple[int, float]]; convert to list[list]
        serialisable[name]["perplexity_curve"] = [
            list(pt) for pt in row.get("perplexity_curve", [])
        ]
    out_path = out_dir / "matched_cost_ablation.json"
    out_path.write_text(json.dumps(serialisable, indent=2))
    logger.info("Results saved to %s", out_path)


def _print_findings(results: dict[str, dict[str, Any]]) -> None:
    """Print a plain-English interpretation of the key finding."""
    if "naive_11x_gelu" not in results or "zk_11x_linear" not in results:
        return

    zk_ppl = results["zk_11x_linear"]["val_ppl"]
    naive_ppl = results["naive_11x_gelu"]["val_ppl"]
    delta = naive_ppl - zk_ppl  # positive = ZK is better

    print("KEY FINDING (11x tier):")
    print(f"  zk_11x_linear  PPL = {zk_ppl:.1f}")
    print(f"  naive_11x_gelu PPL = {naive_ppl:.1f}")
    print(f"  Delta           = {delta:+.1f} (positive = ZK-NAS is better)")

    if delta > 2.0:
        print("  => ZK-NAS activation choices yield measurably better PPL at same ZK cost.")
        print("     Claim: ZK-friendly activations are free lunch — lower proof cost AND better quality.")
    elif delta < -2.0:
        print("  => ZK-friendly activations HURT quality vs GELU at similar ZK cost.")
        print("     Claim: Practitioners should not blindly swap activations for ZK; use NAS to find right arch.")
    else:
        print("  => Activation choice has minimal impact at fixed shape; shape dominates.")
        print("     Claim: ZK-NAS value lies in shape selection, not activation substitution.")

    print()


if __name__ == "__main__":
    main()

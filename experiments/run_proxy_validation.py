#!/usr/bin/env python3
"""Run proxy validation experiment: compare accuracy proxies on 40 configs.

Trains each config for 200 steps (fast calibration) and measures:
- Spearman correlation between each proxy and calibration PPL
- This is Table 1 of the paper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.proxy_validation import run_proxy_validation, markdown_table

OUTPUT_DIR = Path(__file__).parent / "results" / "proxy_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Proxy Validation Experiment (Table 1) ===")
print("Training 40 configs for 200 steps each on WikiText-2...")
print("Comparing: decoder_params, grad_norm, SNIP, NEAR")
print()

results = run_proxy_validation(
    n_configs=40,
    calibration_steps=200,
    seq_len=128,
    output_dir=str(OUTPUT_DIR),
    seed=42,
    device="cpu",
)

print("\n=== Table 1: Proxy Correlation Results ===")
print(markdown_table(results))
print(f"\nBest proxy: {results['best_proxy']}")
print(f"Results saved to {OUTPUT_DIR}/")

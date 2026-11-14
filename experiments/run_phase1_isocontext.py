#!/usr/bin/env python3
"""ZK-NAS Phase 1 with FIXED seq_len=512 (iso-context).

The original Phase 1 searched over all seq_lens (64-512), giving the NAS an
unfair advantage by finding trivially-cheap seq_len=64 configs. This experiment
fixes seq_len=512 for a fair comparison against random search.

Both ZK-NAS and random search now operate on the same space:
  d_model × n_layers × n_heads × ff_mult × activation × use_rotary × use_layernorm
  at fixed seq_len=512.

This generates the key comparison Figure for the paper.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from proxy.features import TransformerConfig
from proxy.accuracy_proxies import decoder_params_score
from experiments.operator_attribution import compute_op_breakdown
from experiments.run_random_search_baseline import (
    run_random_search,
    load_zknas_curve,
    compute_search_efficiency,
    plot_hypervolume_comparison,
)
from search.nas_loop import ZKNASLoop
from search.transformer_space import TransformerSearchSpace

OUTPUT_DIR = Path(__file__).parent / "results" / "nas_phase1_isocontext"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 512
VOCAB = 50257

print("=== ZK-NAS Phase 1: Iso-Context (seq_len=512) ===")
print(f"Output: {OUTPUT_DIR}")

# Override TransformerSearchSpace to only sample seq_len=512 configs
class IsoContextSpace(TransformerSearchSpace):
    """Search space with fixed seq_len=512."""
    def random_config(self) -> TransformerConfig:
        cfg = super().random_config()
        return TransformerConfig(
            cfg.d_model, cfg.n_heads, cfg.n_layers, SEQ_LEN,
            cfg.ff_mult, cfg.activation, cfg.use_rotary, cfg.use_layernorm,
            VOCAB, cfg.tie_embeddings,
        )

    def zk_friendly_candidates(self, n: int = 100) -> list[TransformerConfig]:
        cfgs = super().zk_friendly_candidates(n)
        return [TransformerConfig(c.d_model, c.n_heads, c.n_layers, SEQ_LEN,
                c.ff_mult, c.activation, c.use_rotary, c.use_layernorm, VOCAB, True)
                for c in cfgs]

space = IsoContextSpace(seed=42)

loop = ZKNASLoop(
    output_dir=str(OUTPUT_DIR),
    seed=42,
    phase1_iterations=20,
    phase1_batch_size=8,
    phase1_pool_size=100,
    prove=False,
    verbosity=1,
)
# Inject the iso-context search space
loop._search_space = space

print("\nRunning Phase 1 NAS (iso-context, seq_len=512)...")
state = loop.run(phases=[1])

print(f"\nPhase 1 complete: {len(state.results)} configs, HV={state.hypervolume:.2e}")

# Build ZK-NAS hypervolume curve
zknas_curve = [(i * 8, state.hypervolume) for i in range(1, 21)]  # simplified
# Load actual per-iteration checkpoints
zknas_curve_actual = []
for i in range(20):
    ckpt = OUTPUT_DIR / f"phase1_iter{i:04d}.json"
    if ckpt.exists():
        d = json.loads(ckpt.read_text())
        zknas_curve_actual.append(((i + 1) * 8, d.get("hypervolume", 0)))
if zknas_curve_actual:
    zknas_curve = zknas_curve_actual

# Random search baseline (same seq_len=512, same budget)
print("\nRunning random search baseline (seq_len=512, 160 configs)...")
random_results = run_random_search(n_configs=160, seed=99, seq_len_fixed=SEQ_LEN)
random_curve = random_results["hypervolume_curve"]

# Compare
zknas_final_hv = zknas_curve[-1][1] if zknas_curve else 0
random_final_hv = random_results["final_hypervolume"]
efficiency = compute_search_efficiency(zknas_final_hv, random_curve)

print(f"\n=== Comparison ===")
print(f"ZK-NAS final HV:     {zknas_final_hv:.3e}")
print(f"Random Search HV:    {random_final_hv:.3e}")
print(f"ZK-NAS / Random:     {zknas_final_hv/random_final_hv:.2f}×")
print(f"Random needs {efficiency} configs to match ZK-NAS final HV")

# Plot
plot_hypervolume_comparison(
    zknas_curve, random_curve,
    str(OUTPUT_DIR / "hypervolume_comparison.png"),
)
print(f"\nSaved hypervolume_comparison.png to {OUTPUT_DIR}")

# Save results
summary = {
    "zknas_final_hv": zknas_final_hv,
    "random_final_hv": random_final_hv,
    "ratio": zknas_final_hv / random_final_hv if random_final_hv > 0 else 0,
    "random_configs_to_match_zknas": efficiency,
    "zknas_curve": zknas_curve,
    "random_curve": random_curve,
}
(OUTPUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

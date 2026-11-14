#!/usr/bin/env python3
"""Run ZK-NAS Phase 1: proxy-only search over transformer space.

No training, no proving — just ONNX ZK cost estimates + decoder_params.
Should complete in minutes on CPU.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from search.nas_loop import ZKNASLoop
from search.transformer_space import TransformerSearchSpace

OUTPUT_DIR = Path(__file__).parent / "results" / "nas_phase1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== ZK-NAS Phase 1: Proxy Search ===")
print(f"Output: {OUTPUT_DIR}")

space = TransformerSearchSpace(seed=42)
print(f"Search space: {space.n_configs():,} valid configs")
print(f"GPT-2 baseline: {space.gpt2_baseline()}")

loop = ZKNASLoop(
    output_dir=str(OUTPUT_DIR),
    seed=42,
    phase1_iterations=20,       # 20 iterations × 8 configs = 160 evaluations
    phase1_batch_size=8,
    phase1_pool_size=100,       # smaller pool, faster per iteration
    prove=False,
    verbosity=1,
)

print("\nRunning Phase 1...")
state = loop.run(phases=[1])

print("\n=== Phase 1 Complete ===")
print(f"Total configs evaluated: {len(state.results)}")
print(f"Pareto frontier size: {sum(1 for r in state.results if r.on_pareto_frontier)}")
print(f"Hypervolume: {state.hypervolume:.6f}")

# Save summary
gpt2 = space.gpt2_baseline()
from proxy.accuracy_proxies import decoder_params_score
from proxy.features import ProxyFeatures
gpt2_zk = space.estimate_zk_cost(gpt2).get("constraint_estimate", 0)
gpt2_score = decoder_params_score(gpt2)

print(f"\nGPT-2 baseline: decoder_params_score={gpt2_score:.3f}  zk_cost={gpt2_zk:,}")
print(f"\n{loop.pareto_table(state)}")

# Find configs with 10x lower ZK cost than GPT-2 baseline
if gpt2_zk > 0:
    cheap = [r for r in state.results if r.zk_cost > 0 and r.zk_cost <= gpt2_zk / 10]
    print(f"\nConfigs with ≥10× lower ZK cost than GPT-2: {len(cheap)}")
    for r in sorted(cheap, key=lambda x: -x.decoder_params_score)[:5]:
        print(f"  {r.config.d_model}d/{r.config.n_layers}L/{r.config.n_heads}h "
              f"act={r.config.activation} "
              f"decoder_score={r.decoder_params_score:.3f} "
              f"zk_cost={r.zk_cost:,} "
              f"ratio={gpt2_zk/r.zk_cost:.1f}×")

summary = {
    "n_evaluated": len(state.results),
    "pareto_size": sum(1 for r in state.results if r.on_pareto_frontier),
    "hypervolume": state.hypervolume,
    "gpt2_zk_cost": gpt2_zk,
    "gpt2_decoder_score": gpt2_score,
    "configs_10x_cheaper": len(cheap) if gpt2_zk > 0 else 0,
}
(OUTPUT_DIR / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSummary saved to {OUTPUT_DIR}/phase1_summary.json")

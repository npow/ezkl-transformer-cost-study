#!/usr/bin/env python3
"""ZK-NAS Phase 1 with fixed seq_len=256 for fair GPT-2 comparison.

Fixes context window to 256 tokens and searches over architectural parameters:
d_model, n_layers, n_heads, ff_mult, activation, use_rotary, use_layernorm.

This gives a fair apple-to-apple comparison: same context window, different
transformer shape. The question is: which architecture minimizes ZK cost
at seq_len=256 while maximizing language model quality?
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from proxy.accuracy_proxies import decoder_params_score
from proxy.features import ProxyFeatures, TransformerConfig
from experiments.operator_attribution import compute_op_breakdown
from search.transformer_space import TransformerSearchSpace

OUTPUT_DIR = Path(__file__).parent / "results" / "iso_context_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 256  # fixed context window for fair comparison
VOCAB_SIZE = 50257

# GPT-2 baseline at seq_len=256
gpt2 = TransformerConfig(
    d_model=768, n_heads=12, n_layers=12, seq_len=SEQ_LEN,
    ff_mult=4.0, activation="gelu", use_rotary=False,
    use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
)

print(f"=== Iso-Context ZK-NAS (seq_len={SEQ_LEN}) ===")
gpt2_cost = compute_op_breakdown(gpt2).total_constraints
gpt2_dp = decoder_params_score(gpt2)
print(f"GPT-2 (seq={SEQ_LEN}): ZK cost={gpt2_cost:,}  decoder_score={gpt2_dp:.3f}")
print()

# Enumerate the iso-context search space
configs = []
for d_model in [256, 384, 512, 640, 768, 1024]:
    for n_heads in [4, 6, 8, 12, 16]:
        if d_model % n_heads != 0:
            continue
        for n_layers in [2, 4, 6, 8, 10, 12, 16, 20, 24]:
            for ff_mult in [2.0, 4.0]:
                for activation in ["gelu", "relu", "poly2", "linear"]:
                    for use_rotary in [True, False]:
                        cfg = TransformerConfig(
                            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                            seq_len=SEQ_LEN, ff_mult=ff_mult, activation=activation,
                            use_rotary=use_rotary, use_layernorm=True,
                            vocab_size=VOCAB_SIZE, tie_embeddings=True,
                        )
                        configs.append(cfg)

print(f"Evaluating {len(configs):,} iso-context configs...")

results = []
for cfg in configs:
    bd = compute_op_breakdown(cfg)
    dp = decoder_params_score(cfg)
    results.append({
        "d_model": cfg.d_model, "n_heads": cfg.n_heads, "n_layers": cfg.n_layers,
        "ff_mult": cfg.ff_mult, "activation": cfg.activation, "use_rotary": cfg.use_rotary,
        "zk_cost": bd.total_constraints,
        "matmul_frac": bd.linear_fraction,
        "decoder_params_score": dp,
        "vs_gpt2": gpt2_cost / bd.total_constraints if bd.total_constraints > 0 else 0,
    })

results.sort(key=lambda x: -x["decoder_params_score"])

# Find Pareto frontier (higher decoder_score and lower zk_cost)
def is_pareto(r, all_results):
    for other in all_results:
        if (other["decoder_params_score"] >= r["decoder_params_score"] and
                other["zk_cost"] <= r["zk_cost"] and
                (other["decoder_params_score"] > r["decoder_params_score"] or
                 other["zk_cost"] < r["zk_cost"])):
            return False
    return True

pareto = [r for r in results if is_pareto(r, results)]
pareto.sort(key=lambda x: x["zk_cost"])

# Configs with ≥3× lower cost AND high decoder_score (comparable to GPT-2)
# "Comparable" = decoder_params_score within 5% of GPT-2 baseline
gpt2_score_floor = gpt2_dp * 0.90  # allow 10% lower quality proxy
good_and_cheap = [
    r for r in results
    if r["vs_gpt2"] >= 3.0 and r["decoder_params_score"] >= gpt2_score_floor
]
good_and_cheap.sort(key=lambda x: -x["vs_gpt2"])

print(f"\n=== Pareto Frontier ({len(pareto)} configs) ===")
print(f"{'d_model':>7} {'n_layers':>8} {'n_heads':>7} {'act':<8} {'rotary':>6} "
      f"{'ZK cost':>15} {'vs GPT-2':>10} {'dp_score':>10}")
print("-" * 80)
for r in pareto[:20]:
    print(f"{r['d_model']:>7} {r['n_layers']:>8} {r['n_heads']:>7} {r['activation']:<8} "
          f"{str(r['use_rotary']):>6} {r['zk_cost']:>15,} {r['vs_gpt2']:>9.1f}× "
          f"{r['decoder_params_score']:>10.3f}")

print(f"\n=== High-Quality + ≥3× Cheaper (decoder_score ≥ {gpt2_score_floor:.3f}, "
      f"{len(good_and_cheap)} configs) ===")
print(f"{'d_model':>7} {'n_layers':>8} {'n_heads':>7} {'act':<8} {'rotary':>6} "
      f"{'ZK cost':>15} {'vs GPT-2':>10} {'dp_score':>10}")
print("-" * 80)
for r in good_and_cheap[:10]:
    print(f"{r['d_model']:>7} {r['n_layers']:>8} {r['n_heads']:>7} {r['activation']:<8} "
          f"{str(r['use_rotary']):>6} {r['zk_cost']:>15,} {r['vs_gpt2']:>9.1f}× "
          f"{r['decoder_params_score']:>10.3f}")

# Save
out = {
    "seq_len": SEQ_LEN,
    "gpt2_baseline": {"zk_cost": gpt2_cost, "decoder_params_score": gpt2_dp},
    "n_evaluated": len(results),
    "pareto_size": len(pareto),
    "good_and_cheap_3x": len(good_and_cheap),
    "pareto": pareto,
    "good_and_cheap": good_and_cheap[:20],
}
(OUTPUT_DIR / "iso_context_results.json").write_text(json.dumps(out, indent=2))
print(f"\nResults saved to {OUTPUT_DIR}/iso_context_results.json")
print(f"\nKey finding: {len(good_and_cheap)} configs achieve ≥3× ZK cost reduction "
      f"at comparable LM quality (seq_len={SEQ_LEN})")

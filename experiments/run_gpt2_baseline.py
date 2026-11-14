#!/usr/bin/env python3
"""Measure ZK cost breakdown for GPT-2 small baseline and ZK-friendly variants.

Computes ONNX proxy cost for:
1. GPT-2 small (standard softmax + GELU) - the baseline
2. GPT-2-class with poly2 attention (squared, no softmax lookup)
3. GPT-2-class with ReLU FFN (no GELU lookup)
4. GPT-2-class with both (poly2 + ReLU)
5. Compact variant: smaller d_model with ZK-friendly ops

This gives the core cost reduction numbers for the paper.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from proxy.features import TransformerConfig, ProxyFeatures
from search.transformer_space import TransformerSearchSpace

OUTPUT_DIR = Path(__file__).parent / "results" / "baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

space = TransformerSearchSpace(seed=42)

configs = {
    "gpt2_baseline": TransformerConfig(
        d_model=768, n_heads=12, n_layers=12, seq_len=1024,
        ff_mult=4.0, activation="gelu", use_rotary=False,
        use_layernorm=True, vocab_size=50257, tie_embeddings=True,
    ),
    "gpt2_poly2_attn": TransformerConfig(
        d_model=768, n_heads=12, n_layers=12, seq_len=1024,
        ff_mult=4.0, activation="linear", use_rotary=True,
        use_layernorm=False, vocab_size=50257, tie_embeddings=True,
    ),
    "gpt2_relu_ffn": TransformerConfig(
        d_model=768, n_heads=12, n_layers=12, seq_len=512,
        ff_mult=4.0, activation="relu", use_rotary=True,
        use_layernorm=False, vocab_size=50257, tie_embeddings=True,
    ),
    "gpt2_zk_friendly": TransformerConfig(
        d_model=768, n_heads=12, n_layers=12, seq_len=512,
        ff_mult=4.0, activation="poly2", use_rotary=True,
        use_layernorm=False, vocab_size=50257, tie_embeddings=True,
    ),
    "compact_zk_friendly": TransformerConfig(
        d_model=512, n_heads=8, n_layers=8, seq_len=256,
        ff_mult=4.0, activation="poly2", use_rotary=True,
        use_layernorm=False, vocab_size=50257, tie_embeddings=True,
    ),
    "tiny_zk_friendly": TransformerConfig(
        d_model=384, n_heads=6, n_layers=6, seq_len=256,
        ff_mult=4.0, activation="poly2", use_rotary=True,
        use_layernorm=False, vocab_size=50257, tie_embeddings=True,
    ),
}

print("=== ZK Cost Breakdown by Architecture ===\n")
print(f"{'Name':<25} {'d_model':>7} {'layers':>6} {'seq_len':>7} {'activation':<10} {'ZK cost':>12} {'vs GPT-2':>10}")
print("-" * 85)

results = {}
gpt2_cost = None

for name, cfg in configs.items():
    cost_info = space.estimate_zk_cost(cfg)
    zk_cost = cost_info.get("constraint_estimate", 0)
    feats = ProxyFeatures.from_config(cfg)

    if name == "gpt2_baseline":
        gpt2_cost = zk_cost

    ratio = f"{gpt2_cost/zk_cost:.1f}×" if gpt2_cost and zk_cost > 0 else "—"

    print(f"{name:<25} {cfg.d_model:>7} {cfg.n_layers:>6} {cfg.seq_len:>7} "
          f"{cfg.activation:<10} {zk_cost:>12,} {ratio:>10}")

    results[name] = {
        "config": {k: getattr(cfg, k) for k in
                   ["d_model", "n_heads", "n_layers", "seq_len", "ff_mult",
                    "activation", "use_rotary", "use_layernorm"]},
        "zk_cost": zk_cost,
        "decoder_params": feats.decoder_params,
        "total_params": feats.total_params,
        "nonlinear_ops": feats.gelu_count + feats.softmax_count,
        "vs_gpt2": round(gpt2_cost / zk_cost, 2) if gpt2_cost and zk_cost > 0 else None,
    }

(OUTPUT_DIR / "baseline_costs.json").write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {OUTPUT_DIR}/baseline_costs.json")

if gpt2_cost:
    print(f"\nGPT-2 baseline ZK cost: {gpt2_cost:,} constraints")
    for name, r in results.items():
        if r["vs_gpt2"] and r["vs_gpt2"] >= 5:
            print(f"  {name}: {r['vs_gpt2']}× cheaper ← promising")

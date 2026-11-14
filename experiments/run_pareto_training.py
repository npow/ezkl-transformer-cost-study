#!/usr/bin/env python3
"""Phase 2+3: Train Pareto frontier candidates to get actual PPL numbers.

Trains 6 configs spanning the ZK-cost/quality Pareto frontier at seq_len=512.
Each config is trained from scratch (or fine-tuned from GPT-2 init) and
evaluated on WikiText-2 validation perplexity.

This produces the main paper result: Table 3 (Pareto frontier with PPL + ZK cost).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from proxy.features import TransformerConfig
from experiments.operator_attribution import compute_op_breakdown
from proxy.accuracy_proxies import decoder_params_score
from search.transformer_space import TransformerSearchSpace
from search.lm_trainer import LMTrainer, LMTrainingConfig

OUTPUT_DIR = Path(__file__).parent / "results" / "pareto_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 512
VOCAB_SIZE = 50257

# GPT-2 baseline ZK cost at seq_len=512
gpt2_cost = 98_611_619_328  # from operator attribution

# Pareto frontier candidates at seq_len=512
# Selected to span the cost/quality frontier:
CANDIDATES = {
    "gpt2_baseline": TransformerConfig(
        d_model=768, n_heads=12, n_layers=12, seq_len=SEQ_LEN,
        ff_mult=4.0, activation="gelu", use_rotary=False,
        use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
    ),
    "zk_2x_cheaper": TransformerConfig(  # 640d/10L — ~2x cheaper
        d_model=640, n_heads=8, n_layers=10, seq_len=SEQ_LEN,
        ff_mult=4.0, activation="gelu", use_rotary=True,
        use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
    ),
    "zk_3x_cheaper": TransformerConfig(  # 512d/8L — ~3x cheaper
        d_model=512, n_heads=8, n_layers=8, seq_len=SEQ_LEN,
        ff_mult=4.0, activation="relu", use_rotary=True,
        use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
    ),
    "zk_5x_cheaper": TransformerConfig(  # 384d/10L — ~5x cheaper
        d_model=384, n_heads=6, n_layers=10, seq_len=SEQ_LEN,
        ff_mult=4.0, activation="poly2", use_rotary=True,
        use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
    ),
    "zk_8x_cheaper": TransformerConfig(  # 384d/6L — ~8x cheaper
        d_model=384, n_heads=6, n_layers=6, seq_len=SEQ_LEN,
        ff_mult=4.0, activation="linear", use_rotary=True,
        use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
    ),
    "zk_11x_cheaper": TransformerConfig(  # 384d/4L — ~11x cheaper
        d_model=384, n_heads=6, n_layers=4, seq_len=SEQ_LEN,
        ff_mult=4.0, activation="linear", use_rotary=True,
        use_layernorm=True, vocab_size=VOCAB_SIZE, tie_embeddings=True,
    ),
}

# Training config: calibration mode (500 steps) for proxy validation
# Full training mode (5000 steps) for paper results
CALIBRATION_STEPS = 500   # Phase 2: validate proxy correlation
FULL_STEPS = 5000         # Phase 3: get actual paper PPL numbers
BATCH_SIZE = 4

print("=== Pareto Training Experiment ===")
print(f"seq_len={SEQ_LEN}, {len(CANDIDATES)} candidates")
print()

# Step 1: Compute ZK costs for all candidates
print("ZK Cost Breakdown:")
print(f"{'Name':<20} {'ZK cost':>18} {'vs GPT-2':>10} {'dp_score':>10}")
print("-" * 62)
zk_costs = {}
for name, cfg in CANDIDATES.items():
    cost = compute_op_breakdown(cfg).total_constraints
    dp = decoder_params_score(cfg)
    ratio = gpt2_cost / cost
    zk_costs[name] = cost
    print(f"{name:<20} {cost:>18,} {ratio:>9.1f}× {dp:>10.3f}")

print()

# Step 2: Phase 2 calibration — 500 steps each (validates proxy correlation)
print(f"=== Phase 2: Calibration Training ({CALIBRATION_STEPS} steps each) ===")
trainer = LMTrainer(LMTrainingConfig(
    max_steps=CALIBRATION_STEPS,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    eval_interval=CALIBRATION_STEPS,  # eval once at end
    max_eval_batches=20,              # fast eval
    device="cpu",
))

calibration_results = {}
for name, cfg in CANDIDATES.items():
    print(f"\nTraining {name} ({CALIBRATION_STEPS} steps)...")
    model = trainer.build_and_initialize(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    result = trainer.calibration_run(model, n_steps=CALIBRATION_STEPS)
    calibration_results[name] = {
        "val_ppl": result.val_perplexity,
        "train_loss": result.final_train_loss,
        "steps": result.steps_run,
        "time_s": result.train_time_s,
        "zk_cost": zk_costs[name],
        "vs_gpt2": gpt2_cost / zk_costs[name],
    }
    print(f"  val_ppl={result.val_perplexity:.1f}  train_loss={result.final_train_loss:.4f}  "
          f"time={result.train_time_s:.0f}s")

# Save calibration results
(OUTPUT_DIR / "calibration_results.json").write_text(
    json.dumps(calibration_results, indent=2)
)

print("\n=== Calibration Results (Table 3 - calibration PPL) ===")
print(f"{'Name':<20} {'Cal PPL':>10} {'ZK cost':>18} {'vs GPT-2':>10}")
print("-" * 62)
for name, r in calibration_results.items():
    print(f"{name:<20} {r['val_ppl']:>10.1f} {r['zk_cost']:>18,} {r['vs_gpt2']:>9.1f}×")

print(f"\nResults saved to {OUTPUT_DIR}/")

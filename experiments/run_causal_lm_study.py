#!/usr/bin/env python3
"""Causal-LM quality study with a leak-free protocol.

Replaces every perplexity number produced before the causal-mask fix
(see ``tests/test_transformer_space.py::test_model_is_causal``).

Protocol (identical for every model)
------------------------------------
* WikiText-2 raw, GPT-2 BPE (tiktoken), non-overlapping blocks of SEQ_LEN.
* Random initialisation (``use_pretrained=False``, embeddings trainable).
* AdamW, cosine schedule, STEPS steps, batch BATCH, 3 seeds.
* Checkpoint = best validation perplexity (evaluated every 250 steps); the
  test split is reported at that checkpoint.  Final-checkpoint numbers are
  stored too.
* Reference rows: GPT-2-small pretrained on WebText (zero-shot, same blocks)
  and the GPT-2-small architecture trained from scratch with this protocol.

Usage::

    python experiments/run_causal_lm_study.py --group shape --seeds 42 43 44
    python experiments/run_causal_lm_study.py --group ablation
    python experiments/run_causal_lm_study.py --group reference
    python experiments/run_causal_lm_study.py --group distill
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxy.features import TransformerConfig  # noqa: E402
from search.lm_trainer import LMTrainer, LMTrainingConfig, WikiText2Dataset  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "causal_study"
OUT.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 256
BATCH = 16
STEPS = 5000
VOCAB = 50257
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cfg(d, heads, layers, act, rotary=True):
    return TransformerConfig(
        d_model=d, n_heads=heads, n_layers=layers, seq_len=SEQ_LEN, ff_mult=4.0,
        activation=act, use_rotary=rotary, use_layernorm=True,
        vocab_size=VOCAB, tie_embeddings=True,
    )


GROUPS = {
    # Shape/activation configurations used in the paper's Pareto table.
    "shape": {
        "gpt2_arch_gelu_learnedpos": cfg(768, 12, 12, "gelu", rotary=False),
        "zk_2x_640d10L_gelu": cfg(640, 8, 10, "gelu"),
        "zk_3x_512d8L_relu": cfg(512, 8, 8, "relu"),
        "zk_5x_384d10L_poly2": cfg(384, 6, 10, "poly2"),
        "zk_11x_384d4L_linear": cfg(384, 6, 4, "linear"),
    },
    # Controlled ablation: identical shape, rotary, everything but activation.
    # Same size as GPT-2-small with rotary instead of learned absolute positions.
    "gpt2_rotary": {"gpt2_arch_gelu_rotary": cfg(768, 12, 12, "gelu", rotary=True)},
    # 2x2 decoupling of attention (softmax vs unnormalised linear) from the FFN
    # activation; (softmax, gelu) and (linear, identity) are in "ablation".
    "decomp": {
        "384d6L_attn-linear_ffn-gelu": (cfg(384, 6, 6, "gelu"), True),
        "384d6L_attn-linear_ffn-relu": (cfg(384, 6, 6, "relu"), True),
        "384d6L_attn-softmax_ffn-identity": (cfg(384, 6, 6, "linear"), False),
    },
    "ablation": {f"384d6L_{a}": cfg(384, 6, 6, a) for a in ("gelu", "relu", "poly2", "linear")},
}


def eval_split(forward, split: str, device: str) -> float:
    """Token-weighted perplexity of ``forward(tokens)->logits`` on a split."""
    ds = WikiText2Dataset(split=split, seq_len=SEQ_LEN)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=BATCH, shuffle=False, drop_last=False):
            tokens = batch.to(device)
            logits = forward(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                tokens[:, 1:].reshape(-1), reduction="sum")
            total += loss.item()
            n += tokens[:, 1:].numel()
    return math.exp(total / n)


def run_scratch(name: str, config: TransformerConfig, seed: int, distill: float = 0.0,
                linear_attn=None) -> dict:
    path = OUT / f"{name}_seed{seed}{'_kd' if distill else ''}.json"
    if path.exists():
        return json.loads(path.read_text())
    tcfg = LMTrainingConfig(
        max_steps=STEPS, eval_steps=250, batch_size=BATCH, seq_len=SEQ_LEN,
        warmup_steps=200, device=DEVICE, seed=seed, max_eval_batches=0,
        use_pretrained=False, freeze_embeddings=False, distillation_alpha=distill)
    trainer = LMTrainer(tcfg)
    torch.manual_seed(seed)
    model = trainer.build_and_initialize(config)
    if linear_attn is not None:                      # decouple attention from FFN
        for blk in model.blocks:
            blk.attn.use_linear_attn = linear_attn
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    res = trainer.train(model)
    model.eval()
    final_val = eval_split(model, "validation", DEVICE)
    final_test = eval_split(model, "test", DEVICE)
    # Checkpoint selection on VALIDATION only; test is reported at that checkpoint.
    model.load_state_dict({k: v.to(DEVICE) for k, v in trainer.best_state.items()})
    model.eval()
    out = {
        "name": name, "seed": seed, "distill_alpha": distill, "n_params": n_params,
        "steps": STEPS, "seq_len": SEQ_LEN, "batch": BATCH,
        "final_train_loss": res.final_train_loss,
        "best_step": trainer.best_step,
        "val_ppl": eval_split(model, "validation", DEVICE),   # at selected checkpoint
        "test_ppl": eval_split(model, "test", DEVICE),        # at selected checkpoint
        "final_val_ppl": final_val, "final_test_ppl": final_test,
        "val_ppl_curve": res.perplexity_curve, "train_time_s": time.time() - t0,
    }
    path.write_text(json.dumps(out, indent=1))
    print(f"{name} seed={seed} kd={distill}: val={out['val_ppl']:.2f} test={out['test_ppl']:.2f} "
          f"(best_step={out['best_step']} final_val={out['final_val_ppl']:.1f})", flush=True)
    return out


def run_gpt2_zero_shot() -> dict:
    from transformers import AutoModelForCausalLM
    path = OUT / "gpt2_pretrained_zero_shot.json"
    if path.exists():
        return json.loads(path.read_text())
    m = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    fwd = lambda t: m(t).logits  # noqa: E731
    out = {"name": "gpt2_pretrained_zero_shot", "seq_len": SEQ_LEN,
           "val_ppl": eval_split(fwd, "validation", DEVICE),
           "test_ppl": eval_split(fwd, "test", DEVICE)}
    path.write_text(json.dumps(out, indent=1))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["shape", "ablation", "reference", "distill", "gpt2_rotary", "decomp"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--only", default=None, help="run only the config with this name")
    a = ap.parse_args()
    if a.group == "reference":
        run_gpt2_zero_shot()
    elif a.group == "distill":
        for s in a.seeds:
            run_scratch("zk_11x_384d4L_linear", GROUPS["shape"]["zk_11x_384d4L_linear"], s, distill=0.7)
    elif a.group == "decomp":
        for name, (c, lin) in GROUPS["decomp"].items():
            if a.only and name != a.only:
                continue
            for s in a.seeds:
                run_scratch(name, c, s, linear_attn=lin)
    else:
        for name, c in GROUPS[a.group].items():
            for s in a.seeds:
                run_scratch(name, c, s)

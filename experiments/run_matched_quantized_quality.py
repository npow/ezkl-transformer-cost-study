#!/usr/bin/env python3
"""Train, compile, and compare full precision/EZKL quality on the same weights.

The 512-symbol vocabulary is a deliberately small proxy: the 511 most frequent
GPT-2 BPE tokens in WikiText-2 train are kept and the rest map to UNK. The
script saves the vocabulary map, state dict, ONNX, EZKL settings, and per-split
metrics so the single matched feasibility point can be reproduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import ezkl
import torch
import torch.nn.functional as F

from experiments.onnx_export import export_onnx
from proxy.features import TransformerConfig
from search.lm_trainer import WikiText2Dataset
from search.transformer_space import ZKTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "revision_matched"
OUT.mkdir(parents=True, exist_ok=True)
VOCAB, SEQ, SEED = 512, 16, 51051
N_EVAL = 64  # 960 next-token targets per split after the context position


def corpus(split: str) -> torch.Tensor:
    # Use the exact GPT-2 BPE token stream used by the existing study.
    return WikiText2Dataset(split=split, seq_len=SEQ)._tokens


def make_map(tokens: torch.Tensor) -> dict[str, int]:
    counts = Counter(tokens.tolist())
    ranked = sorted(counts, key=lambda tok: (-counts[tok], tok))[: VOCAB - 1]
    return {str(tok): i for i, tok in enumerate(ranked)}


def convert(tokens: torch.Tensor, token_map: dict[str, int]) -> torch.Tensor:
    # A dense LUT avoids a Python dictionary lookup for every token.
    lut = torch.full((max(50257, int(tokens.max()) + 1),), VOCAB - 1, dtype=torch.long)
    for raw, mapped in token_map.items():
        lut[int(raw)] = mapped
    return lut[tokens]


def batches(tokens: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = tokens[: (len(tokens) // SEQ) * SEQ].reshape(-1, SEQ)
    # Deterministic, evenly spaced sample across the held-out split.
    ix = torch.linspace(0, len(blocks) - 1, min(n, len(blocks))).long()
    return blocks[ix], ix


def audit_sample(data: torch.Tensor, indices: torch.Tensor) -> dict:
    contexts = data[:, :-1].contiguous().numpy().astype("<i8", copy=False).tobytes()
    targets = data[:, 1:].contiguous().numpy().astype("<i8", copy=False).tobytes()
    return {
        "sequence_count": int(len(data)),
        "target_token_count": int(len(data) * (SEQ - 1)),
        "sequence_length": SEQ,
        "block_indices_sha256": hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest(),
        "context_ids_sha256": hashlib.sha256(contexts).hexdigest(),
        "target_ids_sha256": hashlib.sha256(targets).hexdigest(),
    }


def ppl(model: torch.nn.Module, data: torch.Tensor) -> tuple[float, float, int]:
    total, denom = 0.0, 0
    model.eval()
    with torch.no_grad():
        for x in data:
            logits = model(x[None])
            total += F.cross_entropy(logits[:, :-1].float().reshape(-1, VOCAB),
                                     x[None, 1:].reshape(-1), reduction="sum").item()
            denom += SEQ - 1
    return math.exp(total / denom), total, denom


def decode_ezkl_outputs(witness: dict, scale: int) -> torch.Tensor:
    """Decode EZKL's little-endian BN254 field outputs at the fixed-point scale."""
    field = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    flat = witness["outputs"][0]
    values = []
    for encoded in flat:
        value = int.from_bytes(bytes.fromhex(encoded), "little")
        if value > field // 2:
            value -= field
        values.append(value / (2**scale))
    logits = torch.tensor(values, dtype=torch.float32).reshape(1, SEQ, VOCAB)
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)

    train_raw = corpus("train")
    token_map = make_map(train_raw)
    (OUT / "token_map.json").write_text(json.dumps(token_map, sort_keys=True))
    train = convert(train_raw, token_map)
    cfg = TransformerConfig(128, 4, 2, SEQ, 4.0, "relu", False, True, VOCAB, True)
    model = ZKTransformer(cfg).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    blocks = train[: (len(train) // SEQ) * SEQ].reshape(-1, SEQ)
    started = time.time()
    model.train()
    for step in range(args.steps):
        ix = torch.randint(len(blocks), (16,))
        x = blocks[ix]
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    torch.save(model.state_dict(), OUT / "model_state.pt")

    raw_val, raw_test = corpus("validation"), corpus("test")
    val, val_ix = batches(convert(raw_val, token_map), N_EVAL)
    test, test_ix = batches(convert(raw_test, token_map), N_EVAL)
    audit = {"validation": audit_sample(val, val_ix), "test": audit_sample(test, test_ix)}
    (OUT / "evaluation_sample_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    full_val, _, full_val_n = ppl(model, val)
    full_test, _, full_test_n = ppl(model, test)

    # Compile this exact trained instance with EZKL. Calibration samples are
    # drawn deterministically from the mapped training stream.
    with tempfile.TemporaryDirectory(prefix="matched_ezkl_") as td:
        td = Path(td)
        onnx, settings, cal, circuit = (td / n for n in ("model.onnx", "settings.json", "cal.json", "model.ezkl"))
        export_onnx(model, torch.zeros(1, SEQ, dtype=torch.long), str(onnx))
        json_data = {"input_data": train[: 4 * SEQ].reshape(4, SEQ).tolist()}
        cal.write_text(json.dumps(json_data))
        ezkl.gen_settings(model=str(onnx), output=str(settings))
        ezkl.calibrate_settings(data=str(cal), model=str(onnx), settings=str(settings),
                                target="resources", scales=[11])
        setting_obj = json.loads(settings.read_text())
        ezkl.compile_circuit(model=str(onnx), compiled_circuit=str(circuit), settings_path=str(settings))
        (OUT / "model.onnx").write_bytes(onnx.read_bytes())
        (OUT / "settings.json").write_text(settings.read_text())
        (OUT / "model.ezkl").write_bytes(circuit.read_bytes())

        # Evaluate quantized logits from the same compiled circuit on the exact
        # held-out blocks used for the full-precision measurement.
        quantized = {}
        witness_decode_check = None
        for split, data in (("validation", val), ("test", test)):
            total, denom = 0.0, 0
            for i, x in enumerate(data):
                inp, wit = td / "input.json", td / "witness.json"
                inp.write_text(json.dumps({"input_data": [x.tolist()]}))
                witness = ezkl.gen_witness(data=str(inp), model=str(circuit), output=str(wit))
                if not witness:
                    raise RuntimeError(f"EZKL witness generation failed on {split} sample {i}")
                logits = decode_ezkl_outputs(witness, setting_obj["run_args"]["param_scale"])
                if split == "validation" and i == 0:
                    with torch.no_grad():
                        reference_logits = model(x[None]).float()
                    delta = (logits - reference_logits).abs()
                    witness_decode_check = {
                        "sample": "validation block 0",
                        "max_abs_logit_error": float(delta.max()),
                        "mean_abs_logit_error": float(delta.mean()),
                    }
                targets = x[None, 1:]
                total += F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                                         targets.reshape(-1), reduction="sum").item()
                denom += SEQ - 1
            expected_denom = N_EVAL * (SEQ - 1)
            if denom != expected_denom:
                raise RuntimeError(f"{split}: expected {expected_denom} targets, got {denom}")
            quantized[split] = {"ppl": math.exp(total / denom), "nll": total,
                                "target_tokens": denom}

    manifest = {
        "seed": SEED, "steps": args.steps, "elapsed_total_run_s": time.time() - started,
        "architecture": {"d_model": 128, "heads": 4, "layers": 2, "seq_len": SEQ,
                         "activation": "relu", "vocab_size": VOCAB, "ff_mult": 4,
                         "positions": "learned", "layernorm": True, "tied_embeddings": True},
        "token_map_sha256": hashlib.sha256((OUT / "token_map.json").read_bytes()).hexdigest(),
        "state_dict_sha256": hashlib.sha256((OUT / "model_state.pt").read_bytes()).hexdigest(),
        "evaluation_sample_audit": audit,
        "witness_decode_check": witness_decode_check,
        "full_precision": {"validation_ppl": full_val, "validation_targets": full_val_n,
                           "test_ppl": full_test, "test_targets": full_test_n},
        "ezkl": {"version": ezkl.__version__ if hasattr(ezkl, "__version__") else "23.0.5",
                 "num_rows": setting_obj.get("num_rows"),
                 "num_inner_cols": setting_obj.get("run_args", {}).get("num_inner_cols"),
                 "total_dynamic_col_size": setting_obj.get("total_dynamic_col_size"),
                 "logrows": setting_obj.get("run_args", {}).get("logrows"),
                 "param_scale": setting_obj.get("run_args", {}).get("param_scale"),
                 "settings_version": setting_obj.get("version")},
        "compiled_artifact_sha256": {
            name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
            for name in ("model.onnx", "model.ezkl", "settings.json")
        },
        "quantized": quantized,
    }
    for split in ("validation", "test"):
        full_ppl = manifest["full_precision"][f"{split}_ppl"]
        manifest["quantized"][split]["relative_ppl_change_percent"] = (
            100 * (manifest["quantized"][split]["ppl"] / full_ppl - 1)
        )
    (OUT / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

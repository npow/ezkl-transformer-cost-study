#!/usr/bin/env python3
"""Matched EZKL quality at fixed-point scales 11--13 for ReLU/GELU/poly2.

The ReLU/scale-11 result reuses the already released matched model. GELU and
poly2 are trained with the same seed and protocol. Each activation's exact same
weights are evaluated in fp32 and through one compiled circuit per scale on the
same 64 held-out blocks per split (960 next-token targets).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import ezkl
import torch
import torch.nn.functional as F

from experiments.onnx_export import export_onnx
from experiments.run_matched_quantized_quality import (
    N_EVAL, OUT as MATCHED, SEQ, SEED, VOCAB, batches, corpus, convert,
    decode_ezkl_outputs, ppl,
)
from proxy.features import TransformerConfig
from search.transformer_space import ZKTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "revision_quantization"
OUT.mkdir(parents=True, exist_ok=True)
ACTIVATIONS = ("relu", "gelu", "poly2")
SCALES = (11, 12, 13)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(activation: str, train: torch.Tensor) -> torch.nn.Module:
    torch.manual_seed(SEED)
    cfg = TransformerConfig(128, 4, 2, SEQ, 4.0, activation, False, True, VOCAB, True)
    model = ZKTransformer(cfg).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    blocks = train[: (len(train) // SEQ) * SEQ].reshape(-1, SEQ)
    model.train()
    for step in range(500):
        ix = torch.randint(len(blocks), (16,))
        x = blocks[ix]
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


def run_scale(activation: str, scale: int, model_path: Path, model: torch.nn.Module,
              onnx: Path, val: torch.Tensor, test: torch.Tensor) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"q_{activation}_{scale}_") as td:
        td = Path(td)
        if activation == "relu" and scale == 11:
            circuit = MATCHED / "model.ezkl"
            settings = MATCHED / "settings.json"
            setting_obj = json.loads(settings.read_text())
        else:
            settings = OUT / f"{activation}_scale{scale}_settings.json"
            cal = td / "cal.json"
            circuit = OUT / f"{activation}_scale{scale}.ezkl"
            if not settings.exists():
                cal.write_text(json.dumps({"input_data": val[:4].tolist()}))
                ezkl.gen_settings(model=str(onnx), output=str(settings))
                ezkl.calibrate_settings(data=str(cal), model=str(onnx), settings=str(settings),
                                        target="resources", scales=[scale])
            setting_obj = json.loads(settings.read_text())
            actual = setting_obj.get("run_args", {}).get("param_scale")
            if actual != scale:
                raise RuntimeError(f"requested scale {scale}, got {actual}")
            ezkl.compile_circuit(model=str(onnx), compiled_circuit=str(circuit),
                                 settings_path=str(settings))

        compile_hash = sha(circuit)
        quantized = {}
        for split, data in (("validation", val), ("test", test)):
            total, denom = 0.0, 0
            for start in range(0, len(data), 8):
                stop = min(start + 8, len(data))
                request_path = OUT / f"{activation}_scale{scale}_{split}_{start:03d}.input.json"
                result_path = OUT / f"{activation}_scale{scale}_{split}_{start:03d}.nll.json"
                request_path.write_text(json.dumps(data[start:stop].tolist()))
                request_hash = sha(request_path)
                result = None
                if result_path.exists():
                    cached = json.loads(result_path.read_text())
                    if (cached.get("input_sha256") == request_hash
                            and cached.get("count") == stop - start
                            and cached.get("model_sha256") == compile_hash
                            and cached.get("scale") == scale):
                        result = cached
                if result is None:
                    worker = ROOT / "experiments" / "ezkl_quality_witness_chunk.py"
                    subprocess.run([sys.executable, str(worker), "--model", str(circuit),
                                    "--scale", str(scale), "--input", str(request_path),
                                    "--output", str(result_path)], check=True)
                    result = json.loads(result_path.read_text())
                total += sum(result["nlls"])
                denom += result["count"] * (SEQ - 1)
                print(f"{activation} scale={scale} {split} {stop}/{len(data)}", flush=True)
            if denom != N_EVAL * (SEQ - 1):
                raise RuntimeError(f"unexpected denominator for {activation}/{scale}/{split}: {denom}")
            quantized[split] = {"ppl": math.exp(total / denom), "nll": total,
                                "target_tokens": denom}

    for split in ("validation", "test"):
        f = quantized[split]["ppl"]
        reference = metrics[activation]["full_precision"][f"{split}_ppl"]
        quantized[split]["relative_ppl_change_percent"] = 100 * (f / reference - 1)
    return {
        "scale": scale,
        "ezkl": {"version": setting_obj.get("version"), "num_rows": setting_obj.get("num_rows"),
                 "num_inner_cols": setting_obj.get("run_args", {}).get("num_inner_cols"),
                 "total_dynamic_col_size": setting_obj.get("total_dynamic_col_size"),
                 "logrows": setting_obj.get("run_args", {}).get("logrows"),
                 "param_scale": setting_obj.get("run_args", {}).get("param_scale")},
        "settings_sha256": sha(MATCHED / "settings.json") if activation == "relu" and scale == 11
                           else sha(OUT / f"{activation}_scale{scale}_settings.json"),
        "compiled_circuit_sha256": compile_hash,
        "quantized": quantized,
    }


def main() -> None:
    torch.set_num_threads(4)
    raw_train = corpus("train")
    token_map = json.loads((MATCHED / "token_map.json").read_text())
    train = convert(raw_train, token_map)
    raw_val, raw_test = corpus("validation"), corpus("test")
    val, _ = batches(convert(raw_val, token_map), N_EVAL)
    test, _ = batches(convert(raw_test, token_map), N_EVAL)
    audits = json.loads((MATCHED / "evaluation_sample_audit.json").read_text())
    relu_base = json.loads((MATCHED / "metrics.json").read_text())

    global metrics
    metrics = {}
    for activation in ACTIVATIONS:
        started = time.time()
        cfg = TransformerConfig(128, 4, 2, SEQ, 4.0, activation, False, True, VOCAB, True)
        model = ZKTransformer(cfg).cpu()
        state_path = MATCHED / "model_state.pt" if activation == "relu" else OUT / f"{activation}_state.pt"
        if activation == "relu" or state_path.exists():
            model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        else:
            model = train_model(activation, train)
            torch.save(model.state_dict(), state_path)
        model.eval()
        onnx = MATCHED / "model.onnx" if activation == "relu" else OUT / f"{activation}.onnx"
        if activation != "relu":
            if not onnx.exists():
                export_onnx(model, torch.zeros(1, SEQ, dtype=torch.long), str(onnx))
        metrics_path = OUT / f"{activation}_metrics.json"
        if metrics_path.exists():
            metrics[activation] = json.loads(metrics_path.read_text())
            if metrics[activation]["state_sha256"] != sha(state_path):
                raise RuntimeError(f"saved state hash changed for {activation}")
        else:
            full_val, _, n_val = ppl(model, val)
            full_test, _, n_test = ppl(model, test)
            metrics[activation] = {
                "seed": SEED, "training_steps": 500, "activation": activation,
                "architecture": {"d_model": 128, "heads": 4, "layers": 2, "seq_len": SEQ,
                                 "vocab_size": VOCAB, "ff_mult": 4, "positions": "learned",
                                 "layernorm": True, "tied_embeddings": True},
                "state_sha256": sha(state_path), "onnx_sha256": sha(onnx),
                "token_map_sha256": sha(MATCHED / "token_map.json"),
                "evaluation_sample_audit": audits,
                "full_precision": {"validation_ppl": full_val, "validation_targets": n_val,
                                   "test_ppl": full_test, "test_targets": n_test},
                "scales": {},
            }
        for scale in SCALES:
            if str(scale) in metrics[activation]["scales"]:
                saved = metrics[activation]["scales"][str(scale)]
                circuit = (MATCHED / "model.ezkl" if activation == "relu" and scale == 11
                           else OUT / f"{activation}_scale{scale}.ezkl")
                if not circuit.exists():
                    settings = (MATCHED / "settings.json" if activation == "relu" and scale == 11
                                else OUT / f"{activation}_scale{scale}_settings.json")
                    ezkl.compile_circuit(model=str(onnx), compiled_circuit=str(circuit),
                                         settings_path=str(settings))
                artifact_hash = sha(circuit)
                saved.setdefault("evaluation_circuit_sha256", saved["compiled_circuit_sha256"])
                saved["compiled_circuit_sha256"] = artifact_hash
                metrics_path.write_text(json.dumps(metrics[activation], indent=2) + "\n")
                continue
            if activation == "relu" and scale == 11:
                metrics[activation]["scales"]["11"] = {
                    "scale": 11, "ezkl": relu_base["ezkl"],
                    "settings_sha256": relu_base["compiled_artifact_sha256"]["settings.json"],
                    "compiled_circuit_sha256": relu_base["compiled_artifact_sha256"]["model.ezkl"],
                    "quantized": relu_base["quantized"],
                }
                continue
            print(f"Starting {activation} scale={scale}", flush=True)
            metrics[activation]["scales"][str(scale)] = run_scale(
                activation, scale, state_path, model, onnx, val, test)
            metrics_path.write_text(json.dumps(metrics[activation], indent=2) + "\n")
        metrics[activation]["elapsed_total_s"] = time.time() - started
        metrics_path.write_text(json.dumps(metrics[activation], indent=2) + "\n")
        print(json.dumps({"activation": activation,
                          "full_precision": metrics[activation]["full_precision"],
                          "quantized": {s: v["quantized"] for s,v in metrics[activation]["scales"].items()}},
                         indent=2), flush=True)


if __name__ == "__main__":
    main()

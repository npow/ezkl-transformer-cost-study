#!/usr/bin/env python3
"""Compare an author-defined operation-count stand-in with EZKL circuits.

For a grid of small causal transformers, export ONNX, run ``ezkl.gen_settings``
+ ``ezkl.calibrate_settings`` and record the *measured* circuit statistics
(num_rows, total_assignments, logrows).  The same configs are scored with the explicit operation-count stand-in
(``compute_op_breakdown``); it formalizes a heuristic for this study and is not
a published EZKL or field cost model.  The paper reports the rank and
log-log correlation between the two.

Run inside an environment that has ``ezkl`` installed.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ezkl  # noqa: E402
import torch  # noqa: E402

from experiments.operator_attribution import compute_op_breakdown  # noqa: E402
from experiments.onnx_export import export_onnx  # noqa: E402
from proxy.features import TransformerConfig  # noqa: E402
from search.transformer_space import ZKTransformer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "ezkl_validation"
OUT.mkdir(parents=True, exist_ok=True)
VOCAB = 512
GEN_LOGROWS = None  # starting domain for gen_settings (needed for large models)
SCALE = None  # pin the EZKL fixed-point scale (default: let calibration choose, 11-13)

GRID = [
    (d, L, T, act)
    for d, L, T, act in itertools.product(
        (64, 96, 128), (1, 2, 4), (8, 16, 32), ("gelu", "relu", "poly2", "linear")
    )
]


def _gen_settings(onnx_path: str, settings: str) -> None:
    if GEN_LOGROWS:
        a = ezkl.PyRunArgs()
        a.logrows = GEN_LOGROWS
        ezkl.gen_settings(model=onnx_path, output=settings, py_run_args=a)
    else:
        ezkl.gen_settings(model=onnx_path, output=settings)


def _cal_tokens(T: int) -> list:
    """Four fixed random token sequences (degenerate all-zero inputs would give
    EZKL degenerate activation ranges)."""
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, VOCAB, (4, T), generator=g).tolist()


def measure(d: int, L: int, T: int, act: str) -> dict:
    name = f"d{d}_L{L}_T{T}_{act}"
    path = OUT / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    cfg = TransformerConfig(d, 4, L, T, 4.0, act, False, True, VOCAB, True)
    rec = {"name": name, "d_model": d, "n_layers": L, "seq_len": T, "activation": act,
           "analytical_constraints": int(compute_op_breakdown(cfg).total_constraints)}
    tmp = tempfile.mkdtemp()
    onnx_path, settings = os.path.join(tmp, "m.onnx"), os.path.join(tmp, "s.json")
    cal = os.path.join(tmp, "c.json")
    t0 = time.time()
    try:
        export_onnx(ZKTransformer(cfg).eval(), torch.zeros(1, T, dtype=torch.long), onnx_path)
        json.dump({"input_data": _cal_tokens(T)}, open(cal, "w"))
        _gen_settings(onnx_path, settings)
        ezkl.calibrate_settings(data=cal, model=onnx_path, settings=settings, target="resources",
                                scales=[SCALE] if SCALE else None)
        s = json.load(open(settings))
        rec.update(status="ok", logrows=s["run_args"]["logrows"],
                   num_rows=s.get("num_rows"), total_assignments=s.get("total_assignments"),
                   total_const_size=s.get("total_const_size"),
                   num_inner_cols=s["run_args"].get("num_inner_cols"),
                   total_dynamic_col_size=s.get("total_dynamic_col_size"),
                   num_lookups=len(s.get("required_lookups", [])),
                   lookup_range=s["run_args"].get("lookup_range"),
                   input_scale=s["run_args"].get("input_scale"),
                   param_scale=s["run_args"].get("param_scale"))
    except Exception as exc:  # EZKL quantisation / size failures are data, record them
        rec.update(status="failed", error=str(exc)[:300])
    rec["elapsed_s"] = time.time() - t0
    path.write_text(json.dumps(rec, indent=1))
    print(rec["name"], rec["status"], rec.get("logrows"), rec.get("num_rows"),
          rec["analytical_constraints"], f"{rec['elapsed_s']:.0f}s", flush=True)
    return rec


def measure_decomp(d: int, attn: str, ffn: str, L: int = 2, T: int = 16) -> dict:
    """Softmax-vs-linear attention crossed with FFN activation (identity = 'linear')."""
    name = f"ab_d{d}_L{L}_T{T}_attn-{attn}_ffn-{ffn}"
    path = OUT / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    cfg = TransformerConfig(d, 4, L, T, 4.0, ffn, False, True, VOCAB, True)
    rec = {"name": name, "d_model": d, "n_layers": L, "seq_len": T, "attention": attn, "ffn": ffn}
    tmp = tempfile.mkdtemp()
    onnx_path, settings, cal = (os.path.join(tmp, f) for f in ("m.onnx", "s.json", "c.json"))
    t0 = time.time()
    try:
        model = ZKTransformer(cfg).eval()
        for blk in model.blocks:                      # decouple attention from FFN
            blk.attn.use_linear_attn = attn == "linear"
        export_onnx(model, torch.zeros(1, T, dtype=torch.long), onnx_path)
        json.dump({"input_data": _cal_tokens(T)}, open(cal, "w"))
        ezkl.gen_settings(model=onnx_path, output=settings)
        ezkl.calibrate_settings(data=cal, model=onnx_path, settings=settings, target="resources",
                                scales=[SCALE] if SCALE else None)
        s = json.load(open(settings))
        rec.update(status="ok", logrows=s["run_args"]["logrows"], num_rows=s.get("num_rows"),
                   total_assignments=s.get("total_assignments"),
                   num_lookups=len(s.get("required_lookups", [])),
                   param_scale=s["run_args"].get("param_scale"))
    except Exception as exc:
        rec.update(status="failed", error=str(exc)[:300])
    rec["elapsed_s"] = time.time() - t0
    path.write_text(json.dumps(rec, indent=1))
    print(rec["name"], rec["status"], rec.get("logrows"), rec.get("num_rows"), f"{rec['elapsed_s']:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=None, help="pin EZKL scale (default: calibrated)")
    ap.add_argument("--out", default="ezkl_validation", help="subdirectory of artifacts/")
    ap.add_argument("--what", choices=["all", "decomp", "decomp_T", "grid", "large"], default="all")
    a = ap.parse_args()
    SCALE = a.scale
    OUT = Path(__file__).resolve().parent.parent / "artifacts" / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    variants = list(itertools.product(("softmax", "linear"), ("linear", "relu", "gelu", "poly2")))
    if a.what in ("all", "decomp"):
        for d, (attn, ffn) in itertools.product((64, 96, 128), variants):
            measure_decomp(d, attn, ffn)
        print("DECOMPDONE", flush=True)
    if a.what == "decomp_T":          # softmax share vs sequence length
        for T, d, (attn, ffn) in itertools.product((8, 16, 32, 64), (64, 128), variants):
            measure_decomp(d, attn, ffn, T=T)
        print("DECOMPTDONE", flush=True)
    if a.what == "large":             # out-of-range shapes for extrapolation tests
        GEN_LOGROWS = 24
        for (d, L, T), act in itertools.product(
                ((256, 2, 16), (256, 2, 64), (512, 1, 16), (512, 1, 64)), ("linear", "relu", "poly2", "gelu")):
            measure(d, L, T, act)
        print("LARGEDONE", flush=True)
    if a.what in ("all", "grid"):
        for g in GRID:
            measure(*g)
        print("EZKLDONE")

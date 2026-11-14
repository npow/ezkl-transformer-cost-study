#!/usr/bin/env python3
"""Real EZKL proofs for a stratified subset of the validation grid.

For each config: export ONNX -> gen_settings -> calibrate_settings (scale
pinned) -> compile_circuit -> gen_srs (LOCAL, insecure; fine for timing) ->
setup -> gen_witness -> prove -> verify.  Records setup/prove/verify wall time,
peak RSS of the process tree, proof size and key sizes next to ``num_rows`` and
``logrows``.  Run inside an environment with ``ezkl`` installed.

usage: python experiments/run_ezkl_proving.py [--max-logrows 22]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ezkl  # noqa: E402
import torch  # noqa: E402

from experiments.onnx_export import export_onnx  # noqa: E402
from experiments.run_ezkl_validation import VOCAB, _cal_tokens  # noqa: E402
from proxy.features import TransformerConfig  # noqa: E402
from search.transformer_space import ZKTransformer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "ezkl_proving"  # overridden by --out
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 11  # pinned: calibration otherwise picks 11-13 depending on activation


def _maybe_await(x):
    return asyncio.run(x) if asyncio.iscoroutine(x) else x


def prove_one(d: int, L: int, T: int, act: str, max_logrows: int) -> dict:
    name = f"d{d}_L{L}_T{T}_{act}"
    path = OUT / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    rec = {"name": name, "d_model": d, "n_layers": L, "seq_len": T, "activation": act, "scale": SCALE}
    cfg = TransformerConfig(d, 4, L, T, 4.0, act, False, True, VOCAB, True)
    tmp = Path(tempfile.mkdtemp())
    f = {k: str(tmp / v) for k, v in dict(onnx="m.onnx", settings="s.json", cal="c.json", circuit="m.ezkl",
                                         srs="k.srs", vk="vk.key", pk="pk.key", inp="in.json",
                                         wit="w.json", proof="p.json").items()}
    try:
        torch.manual_seed(0)
        export_onnx(ZKTransformer(cfg).eval(), torch.zeros(1, T, dtype=torch.long), f["onnx"])
        cal = _cal_tokens(T)
        json.dump({"input_data": cal}, open(f["cal"], "w"))
        json.dump({"input_data": [cal[0]]}, open(f["inp"], "w"))
        ezkl.gen_settings(model=f["onnx"], output=f["settings"])
        ezkl.calibrate_settings(data=f["cal"], model=f["onnx"], settings=f["settings"], target="resources",
                                scales=[SCALE], max_logrows=max_logrows)
        s = json.load(open(f["settings"]))
        rec.update(logrows=s["run_args"]["logrows"], num_rows=s.get("num_rows"),
                   param_scale=s["run_args"].get("param_scale"))
        if rec["logrows"] > max_logrows:
            rec.update(status="skipped_too_large")
        else:
            t = time.time(); ezkl.compile_circuit(model=f["onnx"], compiled_circuit=f["circuit"], settings_path=f["settings"])
            rec["compile_s"] = time.time() - t
            t = time.time(); _maybe_await(ezkl.gen_srs(f["srs"], rec["logrows"]))
            rec["gen_srs_s"] = time.time() - t
            t = time.time(); ezkl.setup(model=f["circuit"], vk_path=f["vk"], pk_path=f["pk"], srs_path=f["srs"])
            rec["setup_s"] = time.time() - t
            t = time.time(); _maybe_await(ezkl.gen_witness(data=f["inp"], model=f["circuit"], output=f["wit"]))
            rec["witness_s"] = time.time() - t
            t = time.time(); ezkl.prove(witness=f["wit"], model=f["circuit"], pk_path=f["pk"], proof_path=f["proof"], srs_path=f["srs"])
            rec["prove_s"] = time.time() - t
            t = time.time(); ok = ezkl.verify(proof_path=f["proof"], settings_path=f["settings"], vk_path=f["vk"], srs_path=f["srs"])
            rec["verify_s"] = time.time() - t
            rec.update(verified=bool(ok), status="ok", proof_bytes=os.path.getsize(f["proof"]),
                       pk_bytes=os.path.getsize(f["pk"]), vk_bytes=os.path.getsize(f["vk"]),
                       peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception as exc:
        rec.update(status="failed", error=str(exc)[:300])
    shutil.rmtree(tmp, ignore_errors=True)          # proving keys are ~10 GB each
    path.write_text(json.dumps(rec, indent=1))
    print(name, rec.get("status"), rec.get("logrows"), rec.get("num_rows"), rec.get("prove_s"), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-logrows", type=int, default=21)
    ap.add_argument("--out", default="ezkl_proving", help="subdirectory of artifacts/ (use one per repeat)")
    ap.add_argument("--small-only", action="store_true", help="only the shapes with domain <= 2^20 (for repeats)")
    ap.add_argument("--one", nargs=4, metavar=("D", "L", "T", "ACT"), help="run a single config (child mode)")
    a = ap.parse_args()
    OUT = Path(__file__).resolve().parent.parent / "artifacts" / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    if a.one:
        prove_one(int(a.one[0]), int(a.one[1]), int(a.one[2]), a.one[3], a.max_logrows)
    else:
        shapes = ((64, 1, 8), (64, 1, 16), (64, 2, 8), (96, 1, 8), (64, 2, 16), (128, 1, 8))
        if a.small_only:
            shapes = tuple(x for x in shapes if x != (64, 2, 16))
        # one fresh process per config so ru_maxrss is a per-config peak
        for d, L, T in shapes:
            for act in ("linear", "relu", "poly2", "gelu"):
                subprocess.run([sys.executable, __file__, "--one", str(d), str(L), str(T), act,
                                "--max-logrows", str(a.max_logrows), "--out", a.out], check=False)
        print("PROVEDONE", flush=True)

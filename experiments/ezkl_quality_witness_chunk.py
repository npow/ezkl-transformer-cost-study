#!/usr/bin/env python3
"""Generate quantized next-token NLLs for a short batch of EZKL witnesses.

Short-lived worker processes bound EZKL witness-memory growth during the
activation/scale sensitivity sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ezkl
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_matched_quantized_quality import VOCAB, SEQ, decode_ezkl_outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scale", required=True, type=int)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    sequences = json.loads(args.input.read_text())
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    nlls = []
    for i, sequence in enumerate(sequences):
        data, witness_path = args.input.parent / "witness_input.json", args.input.parent / "witness.json"
        data.write_text(json.dumps({"input_data": [sequence]}))
        witness = ezkl.gen_witness(data=str(data), model=args.model, output=str(witness_path))
        if not witness:
            raise RuntimeError(f"EZKL witness generation failed at chunk sample {i}")
        logits = decode_ezkl_outputs(witness, args.scale)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), targets,
                               reduction="sum").item()
        nlls.append(loss)
        del witness, logits
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.write_text(json.dumps({"nlls": nlls, "count": len(nlls),
                                       "input_sha256": input_hash,
                                       "model_sha256": model_hash,
                                       "scale": args.scale}, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""logrows vs. causal-mask fill value (EZKL sizes its softmax lookup range from it).

Two shapes x four activations x five fills.  Rows must be constant across fills;
logrows is not.  Run inside an environment with ``ezkl`` installed.
Writes artifacts/ezkl_validation/maskfill_sensitivity.json.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ezkl  # noqa: E402
import torch  # noqa: E402

import search.transformer_space as tso  # noqa: E402
from experiments.onnx_export import export_onnx  # noqa: E402
from experiments.run_ezkl_validation import VOCAB, _cal_tokens  # noqa: E402
from proxy.features import TransformerConfig  # noqa: E402
from search.transformer_space import ZKTransformer  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "ezkl_validation" / "maskfill_sensitivity.json"
FILLS = (-1e4, -1e3, -100.0, -30.0, -10.0)
SHAPES = ((64, 1, 8), (128, 2, 16))


def main() -> None:
    rows = []
    for d, L, T in SHAPES:
        for act in ("gelu", "relu", "poly2", "linear"):
            for fill in FILLS:
                if act == "linear" and fill != FILLS[0]:
                    continue  # no softmax, hence no mask lookup: one measurement
                tso.CAUSAL_MASK_FILL = fill
                cfg = TransformerConfig(d, 4, L, T, 4.0, act, False, True, VOCAB, True)
                tmp = tempfile.mkdtemp()
                p, s, c = (os.path.join(tmp, f) for f in ("m.onnx", "s.json", "c.json"))
                try:
                    export_onnx(ZKTransformer(cfg).eval(), torch.zeros(1, T, dtype=torch.long), p)
                    json.dump({"input_data": _cal_tokens(T)}, open(c, "w"))
                    ezkl.gen_settings(model=p, output=s)
                    ezkl.calibrate_settings(data=c, model=p, settings=s, target="resources")
                    j = json.load(open(s))
                    r = dict(d=d, L=L, T=T, act=act, fill=fill, logrows=j["run_args"]["logrows"],
                             num_rows=j.get("num_rows"), lookup_range=j["run_args"].get("lookup_range"),
                             param_scale=j["run_args"].get("param_scale"))
                except Exception as e:  # noqa: BLE001
                    r = dict(d=d, L=L, T=T, act=act, fill=fill, error=str(e)[:100])
                rows.append(r)
                print(json.dumps(r), flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    print("MASKDONE")


if __name__ == "__main__":
    main()

"""ONNX export helper shared by the EZKL experiment scripts (no zkml_bench dependency)."""
from __future__ import annotations

import io
import warnings

import onnx
import torch


def export_onnx(model: torch.nn.Module, sample_input: torch.Tensor, path: str) -> None:
    """Export model to ONNX at *path*, suppressing noisy torch.onnx logs."""
    buf = io.BytesIO()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            model,
            sample_input,
            buf,
            opset_version=18,
            input_names=["input_ids"],
            output_names=["logits"],
        )
    buf.seek(0)
    model_proto = onnx.load_from_string(buf.read())
    onnx.save(model_proto, path)

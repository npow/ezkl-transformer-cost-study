"""Isolated transformer kernel benchmarking for ZK proof cost estimation.

Runs ONNX proxy estimation (and optionally actual EZKL proofs) on minimal
PyTorch modules containing individual transformer kernels across a shape grid.
The resulting cost table feeds the KernelDecompositionProxy lookup.

Rationale: proving individual kernels gives 10-50x more labels per proving-hour
than proving full models. The kernel costs compose approximately additively.
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful EzklProver import
# ---------------------------------------------------------------------------

_EZKLProver: Any = None  # pylint: disable=invalid-name
_EZKL_AVAILABLE = False  # pylint: disable=invalid-name
try:
    import zkml_bench.provers.ezkl as _ezkl_mod

    _EZKLProver = _ezkl_mod.EZKLProver
    _EZKL_AVAILABLE = True  # pylint: disable=invalid-name
except ImportError:
    warnings.warn(
        "EZKLProver could not be imported; falling back to proxy-only mode. "
        "Set prove=False or install zkml-bench with EZKL support.",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

_VALID_KERNEL_TYPES: frozenset[str] = frozenset(
    {"layernorm", "softmax", "gelu", "attn_qk", "attn_av", "ffn", "rmsnorm"}
)


@dataclass
class KernelShape:
    """Describes the tensor dimensions for a single transformer kernel benchmark."""

    kernel_type: str  # 'layernorm'|'softmax'|'gelu'|'attn_qk'|'attn_av'|'ffn'|'rmsnorm'
    batch_size: int
    seq_len: int
    d_model: int
    n_heads: int  # only relevant for attn_* kernels
    ff_mult: float  # only relevant for ffn kernel

    def __post_init__(self) -> None:
        if self.kernel_type not in _VALID_KERNEL_TYPES:
            raise ValueError(
                f"kernel_type must be one of {sorted(_VALID_KERNEL_TYPES)}; "
                f"got {self.kernel_type!r}"
            )
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive; got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.ff_mult <= 0.0:
            raise ValueError(f"ff_mult must be positive; got {self.ff_mult}")


@dataclass
class KernelBenchmark:
    """Result of benchmarking one (kernel_type, shape) combination."""

    shape: KernelShape
    onnx_estimate: int  # from compute_proxy; -1 if unavailable
    ezkl_num_rows: Optional[int]  # None if not proved
    ezkl_prove_time: Optional[float]  # seconds; None if not proved
    error: Optional[str]  # non-None if something failed


# ---------------------------------------------------------------------------
# Kernel nn.Module implementations
# ---------------------------------------------------------------------------


class _LayerNormKernel(nn.Module):
    """LayerNorm applied to (batch, seq_len, d_model)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalisation to *x*."""
        out: torch.Tensor = self.norm(x)
        return out


class _RMSNormKernel(nn.Module):
    """RMSNorm applied to (batch, seq_len, d_model).

    Computes: x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation to *x*."""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out: torch.Tensor = (x / rms) * self.weight
        return out


class _SoftmaxKernel(nn.Module):
    """Softmax applied to (batch, n_heads, seq_len, seq_len)."""

    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax over the last dimension of *x*."""
        out: torch.Tensor = self.softmax(x)
        return out


class _GELUKernel(nn.Module):
    """GELU applied to (batch, seq_len, d_model)."""

    def __init__(self) -> None:
        super().__init__()
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation to *x* element-wise."""
        out: torch.Tensor = self.gelu(x)
        return out


class _AttnQKKernel(nn.Module):
    """Scaled dot-product QK^T.

    Input shape: (batch, n_heads, seq_len, 2 * head_dim).
    Q = input[..., :head_dim], K = input[..., head_dim:].
    Output: einsum('bhid,bhjd->bhij', q, k) / sqrt(head_dim),
    shape (batch, n_heads, seq_len, seq_len).
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.scale: float = math.sqrt(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scaled QK^T attention scores from packed Q|K input."""
        # x: (batch, n_heads, seq_len, 2 * head_dim)
        half = x.shape[-1] // 2
        q = x[..., :half]  # (batch, n_heads, seq_len, head_dim)
        k = x[..., half:]  # (batch, n_heads, seq_len, head_dim)
        out: torch.Tensor = torch.einsum("bhid,bhjd->bhij", q, k) / self.scale
        return out


class _AttnAVKernel(nn.Module):
    """Weighted sum of values: softmax(scores) @ V.

    Input shape: (batch, n_heads, seq_len, seq_len + head_dim).
    First seq_len columns = raw scores; remaining head_dim columns = V.
    Output: (batch, n_heads, seq_len, head_dim).

    This single-tensor packing avoids multi-input ONNX export constraints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted value sum from packed scores|V input."""
        # x: (batch, n_heads, seq_len, seq_len + head_dim)
        seq_len = x.shape[2]
        scores_raw = x[..., :seq_len]  # (batch, n_heads, seq_len, seq_len)
        v = x[..., seq_len:]  # (batch, n_heads, seq_len, head_dim)
        attn_weights = self.softmax(scores_raw)
        b_times_h = x.shape[0] * x.shape[1]
        head_dim = v.shape[-1]
        out: torch.Tensor = torch.bmm(
            attn_weights.reshape(b_times_h, seq_len, seq_len),
            v.reshape(b_times_h, seq_len, head_dim),
        ).reshape(x.shape[0], x.shape[1], seq_len, head_dim)
        return out


class _FFNKernel(nn.Module):
    """Two-layer MLP: Linear(d, ff*d) -> GELU -> Linear(ff*d, d)."""

    def __init__(self, d_model: int, ff_mult: float) -> None:
        super().__init__()
        ff_dim = int(d_model * ff_mult)
        self.up = nn.Linear(d_model, ff_dim)
        self.act = nn.GELU()
        self.down = nn.Linear(ff_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer FFN with GELU activation to *x*."""
        out: torch.Tensor = self.down(self.act(self.up(x)))
        return out


# ---------------------------------------------------------------------------
# KernelHarness
# ---------------------------------------------------------------------------


class KernelHarness:
    """Runs proofs on isolated transformer kernels across a shape grid.

    Rationale: proving individual kernels gives 10-50x more labels per
    proving-hour than proving full models. The kernel costs compose
    approximately additively, giving the KernelDecompositionProxy its
    lookup table.
    """

    def make_kernel_model(self, shape: KernelShape) -> nn.Module:
        """Build a minimal PyTorch module containing only one kernel.

        Each kernel is wrapped in a thin nn.Module so it can be:
        1. ONNX-exported for proxy estimation
        2. EZKL-proved for ground truth

        Parameters
        ----------
        shape:
            Fully specified kernel shape.

        Returns
        -------
        nn.Module
            Module in eval mode with a single-tensor ``forward(x)`` signature.

        Raises
        ------
        ValueError
            If ``shape.kernel_type`` is not recognised.
        """
        kt = shape.kernel_type
        model: nn.Module
        if kt == "layernorm":
            model = _LayerNormKernel(shape.d_model)
        elif kt == "rmsnorm":
            model = _RMSNormKernel(shape.d_model)
        elif kt == "softmax":
            model = _SoftmaxKernel()
        elif kt == "gelu":
            model = _GELUKernel()
        elif kt == "attn_qk":
            head_dim = shape.d_model // shape.n_heads
            model = _AttnQKKernel(head_dim)
        elif kt == "attn_av":
            model = _AttnAVKernel()
        elif kt == "ffn":
            model = _FFNKernel(shape.d_model, shape.ff_mult)
        else:
            raise ValueError(f"Unknown kernel_type: {kt!r}")

        model.eval()
        return model

    def _make_sample_input(self, shape: KernelShape) -> torch.Tensor:
        """Return a representative input tensor for ``shape``."""
        b = shape.batch_size
        s = shape.seq_len
        d = shape.d_model
        h = shape.n_heads
        head_dim = d // h

        kt = shape.kernel_type
        if kt in ("layernorm", "rmsnorm", "gelu", "ffn"):
            return torch.randn(b, s, d)
        if kt == "softmax":
            return torch.randn(b, h, s, s)
        if kt == "attn_qk":
            # Pack Q and K along last dim: (b, h, s, 2*head_dim)
            return torch.randn(b, h, s, 2 * head_dim)
        if kt == "attn_av":
            # Pack scores (s cols) + V (head_dim cols): (b, h, s, s+head_dim)
            return torch.randn(b, h, s, s + head_dim)
        raise ValueError(f"Unknown kernel_type: {kt!r}")

    def benchmark_shape(
        self,
        shape: KernelShape,
        prove: bool = False,
        ezkl_timeout: int = 300,
    ) -> KernelBenchmark:
        """Run proxy estimation (always) and optional actual proving for one shape.

        Parameters
        ----------
        shape:
            Kernel shape to benchmark.
        prove:
            When True, run an actual EZKL proof after the proxy estimate.
        ezkl_timeout:
            Per-step timeout in seconds for EZKL proving.

        Returns
        -------
        KernelBenchmark
            Populated result object.
        """
        try:
            from zkml_bench.metrics.zk_proxy import (  # pylint: disable=import-outside-toplevel
                compute_proxy,
            )
        except ImportError as exc:
            return KernelBenchmark(
                shape=shape,
                onnx_estimate=-1,
                ezkl_num_rows=None,
                ezkl_prove_time=None,
                error=f"compute_proxy import failed: {exc}",
            )

        try:
            model = self.make_kernel_model(shape)
            sample_input = self._make_sample_input(shape)
            proxy_result = compute_proxy(model, sample_input)
            onnx_estimate: int = int(proxy_result.constraint_estimate)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Proxy estimation failed for %s: %s", shape, exc)
            return KernelBenchmark(
                shape=shape,
                onnx_estimate=-1,
                ezkl_num_rows=None,
                ezkl_prove_time=None,
                error=f"proxy_failed: {exc}",
            )

        ezkl_num_rows: Optional[int] = None
        ezkl_prove_time: Optional[float] = None
        error: Optional[str] = None

        if prove:
            if not _EZKL_AVAILABLE:
                error = "ezkl_unavailable: EZKLProver could not be imported"
                logger.warning("Skipping EZKL proof for %s: %s", shape, error)
            else:
                ezkl_num_rows, ezkl_prove_time, error = self._run_ezkl_proof(
                    shape=shape,
                    ezkl_timeout=ezkl_timeout,
                )

        return KernelBenchmark(
            shape=shape,
            onnx_estimate=onnx_estimate,
            ezkl_num_rows=ezkl_num_rows,
            ezkl_prove_time=ezkl_prove_time,
            error=error,
        )

    def _run_ezkl_proof(  # pylint: disable=too-many-locals
        self,
        shape: KernelShape,
        ezkl_timeout: int,
    ) -> tuple[Optional[int], Optional[float], Optional[str]]:
        """Run a full EZKL proof for *shape* and return (num_rows, prove_time, error).

        Returns a 3-tuple:
        - ``num_rows``: EZKL circuit row count, or None on failure.
        - ``prove_time``: wall-clock seconds for the full prove pipeline, or None.
        - ``error``: human-readable error string, or None on success.
        """
        import hashlib  # pylint: disable=import-outside-toplevel
        import os  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel

        try:
            model = self.make_kernel_model(shape)
            sample_input = self._make_sample_input(shape)

            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, "kernel.onnx")
                torch.onnx.export(
                    model,
                    (sample_input,),
                    onnx_path,
                    opset_version=17,
                    input_names=["x"],
                    output_names=["output"],
                )
                cal_tensors = [self._make_sample_input(shape) for _ in range(8)]
                input_data = [sample_input.flatten().tolist()]
                calibration_data = [t.flatten().tolist() for t in cal_tensors]

                model_hash = hashlib.md5(  # nosec: non-security use
                    repr(shape).encode()
                ).hexdigest()
                cache_dir = os.path.join(tmpdir, "cache")

                prover = _EZKLProver()
                t_start = time.perf_counter()
                result = prover.prove(
                    onnx_path=onnx_path,
                    input_data=input_data,
                    calibration_data=calibration_data,
                    accuracy_top1=None,
                    cache_dir=cache_dir,
                    model_hash_str=model_hash,
                    timeout_setup=ezkl_timeout,
                    timeout_prove=ezkl_timeout,
                )
                t_elapsed = time.perf_counter() - t_start

                if result.available and result.error is None:
                    return int(result.num_rows), t_elapsed, None
                return None, None, f"ezkl_failed: {result.error}"
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("EZKL proof failed for %s: %s", shape, exc)
            return None, None, f"ezkl_exception: {exc}"

    def run_grid(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        self,
        seq_lens: Optional[list[int]] = None,
        d_models: Optional[list[int]] = None,
        n_heads_list: Optional[list[int]] = None,
        ff_mults: Optional[list[float]] = None,
        prove: bool = False,
        max_workers: int = 4,
    ) -> list[KernelBenchmark]:
        """Run all kernel types across the shape grid.

        With prove=False: ONNX estimates only, very fast (~seconds total).
        With prove=True: actual EZKL proofs, slow (minutes per shape).

        Parameters
        ----------
        seq_lens:
            Sequence lengths to sweep. Defaults to [64, 128, 256, 512].
        d_models:
            Model dimensions to sweep. Defaults to [256, 512, 768, 1024].
        n_heads_list:
            Attention head counts to sweep. Defaults to [4, 8, 12, 16].
        ff_mults:
            FFN expansion multipliers to sweep. Defaults to [2.0, 4.0].
        prove:
            When True, run actual EZKL proofs.
        max_workers:
            Maximum thread-pool workers; ONNX proxy is thread-safe.

        Returns
        -------
        list[KernelBenchmark]
            One entry per (kernel_type, shape) combination with valid head counts.
        """
        if seq_lens is None:
            seq_lens = [64, 128, 256, 512]
        if d_models is None:
            d_models = [256, 512, 768, 1024]
        if n_heads_list is None:
            n_heads_list = [4, 8, 12, 16]
        if ff_mults is None:
            ff_mults = [2.0, 4.0]

        shapes: list[KernelShape] = []

        for s in seq_lens:
            for d in d_models:
                for h in n_heads_list:
                    if d % h != 0:
                        continue
                    for ff in ff_mults:
                        for kt in sorted(_VALID_KERNEL_TYPES):
                            shapes.append(
                                KernelShape(
                                    kernel_type=kt,
                                    batch_size=1,
                                    seq_len=s,
                                    d_model=d,
                                    n_heads=h,
                                    ff_mult=ff,
                                )
                            )

        results: list[KernelBenchmark] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_shape = {
                pool.submit(self.benchmark_shape, shape, prove): shape
                for shape in shapes
            }
            for future in as_completed(future_to_shape):
                try:
                    results.append(future.result())
                except Exception as exc:  # pylint: disable=broad-except
                    failed_shape = future_to_shape[future]
                    logger.error("Unexpected error for shape %s: %s", failed_shape, exc)
                    results.append(
                        KernelBenchmark(
                            shape=failed_shape,
                            onnx_estimate=-1,
                            ezkl_num_rows=None,
                            ezkl_prove_time=None,
                            error=f"unhandled_exception: {exc}",
                        )
                    )

        return results

    def to_cost_table(self, benchmarks: list[KernelBenchmark]) -> dict[str, int]:
        """Build a lookup table: (kernel_type, shape_key) -> cost.

        Used by KernelDecompositionProxy.
        Shape key = ``(seq_len, d_model, n_heads, ff_mult)`` joined to the
        kernel type with underscores.

        The cost is ``ezkl_num_rows`` when available, otherwise ``onnx_estimate``.
        Entries with no valid cost (both ``-1`` / ``None``) are omitted.

        Parameters
        ----------
        benchmarks:
            List of benchmark results from :meth:`run_grid` or
            :meth:`benchmark_shape`.

        Returns
        -------
        dict[str, int]
            Mapping from ``"{kernel_type}_{seq_len}_{d_model}_{n_heads}_{ff_mult}"``
            to integer cost.
        """
        table: dict[str, int] = {}
        for bm in benchmarks:
            s = bm.shape
            key = f"{s.kernel_type}_{s.seq_len}_{s.d_model}_{s.n_heads}_{s.ff_mult}"
            cost: Optional[int] = None
            if bm.ezkl_num_rows is not None:
                cost = bm.ezkl_num_rows
            elif bm.onnx_estimate >= 0:
                cost = bm.onnx_estimate
            if cost is not None:
                table[key] = cost
        return table

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, benchmarks: list[KernelBenchmark], path: str) -> None:
        """Save benchmark results to JSON.

        Parameters
        ----------
        benchmarks:
            List of :class:`KernelBenchmark` objects to persist.
        path:
            Destination file path.
        """
        payload: dict[str, Any] = {
            "version": 1,
            "benchmarks": [_benchmark_to_dict(bm) for bm in benchmarks],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> list[KernelBenchmark]:
        """Load benchmark results from JSON.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        list[KernelBenchmark]
            Deserialised benchmark list.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        with open(path, encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)
        return [_benchmark_from_dict(d) for d in payload.get("benchmarks", [])]


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _shape_to_dict(shape: KernelShape) -> dict[str, Any]:
    """Serialise a :class:`KernelShape` to a JSON-compatible dict."""
    return {
        "kernel_type": shape.kernel_type,
        "batch_size": shape.batch_size,
        "seq_len": shape.seq_len,
        "d_model": shape.d_model,
        "n_heads": shape.n_heads,
        "ff_mult": shape.ff_mult,
    }


def _shape_from_dict(d: dict[str, Any]) -> KernelShape:
    """Deserialise a :class:`KernelShape` from a JSON-compatible dict."""
    return KernelShape(
        kernel_type=str(d["kernel_type"]),
        batch_size=int(d["batch_size"]),
        seq_len=int(d["seq_len"]),
        d_model=int(d["d_model"]),
        n_heads=int(d["n_heads"]),
        ff_mult=float(d["ff_mult"]),
    )


def _benchmark_to_dict(bm: KernelBenchmark) -> dict[str, Any]:
    """Serialise a :class:`KernelBenchmark` to a JSON-compatible dict."""
    return {
        "shape": _shape_to_dict(bm.shape),
        "onnx_estimate": bm.onnx_estimate,
        "ezkl_num_rows": bm.ezkl_num_rows,
        "ezkl_prove_time": bm.ezkl_prove_time,
        "error": bm.error,
    }


def _benchmark_from_dict(d: dict[str, Any]) -> KernelBenchmark:
    """Deserialise a :class:`KernelBenchmark` from a JSON-compatible dict."""
    return KernelBenchmark(
        shape=_shape_from_dict(d["shape"]),
        onnx_estimate=int(d["onnx_estimate"]),
        ezkl_num_rows=(
            int(d["ezkl_num_rows"]) if d.get("ezkl_num_rows") is not None else None
        ),
        ezkl_prove_time=(
            float(d["ezkl_prove_time"])
            if d.get("ezkl_prove_time") is not None
            else None
        ),
        error=d.get("error"),
    )

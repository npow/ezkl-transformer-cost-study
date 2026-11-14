"""Tests for proxy.kernels module."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional, cast

import pytest
import torch

from proxy.kernels import (
    KernelBenchmark,
    KernelHarness,
    KernelShape,
    _VALID_KERNEL_TYPES,
    _benchmark_from_dict,
    _benchmark_to_dict,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_shape(
    kernel_type: str = "layernorm",
    batch_size: int = 1,
    seq_len: int = 64,
    d_model: int = 128,
    n_heads: int = 4,
    ff_mult: float = 4.0,
) -> KernelShape:
    return KernelShape(
        kernel_type=kernel_type,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        ff_mult=ff_mult,
    )


HARNESS = KernelHarness()

# ---------------------------------------------------------------------------
# KernelShape validation
# ---------------------------------------------------------------------------


class TestKernelShapeValidation:
    def test_valid_shape_constructs(self) -> None:
        s = _make_shape()
        assert s.kernel_type == "layernorm"

    def test_invalid_kernel_type_raises(self) -> None:
        with pytest.raises(ValueError, match="kernel_type"):
            _make_shape(kernel_type="unknown_op")

    def test_d_model_not_divisible_by_n_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            _make_shape(d_model=100, n_heads=3)

    def test_zero_d_model_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            _make_shape(d_model=0)

    def test_all_valid_kernel_types_construct(self) -> None:
        for kt in _VALID_KERNEL_TYPES:
            s = _make_shape(kernel_type=kt)
            assert s.kernel_type == kt


# ---------------------------------------------------------------------------
# KernelHarness.make_kernel_model — output shapes
# ---------------------------------------------------------------------------


class TestMakeKernelModel:
    """Verify that each kernel model produces the expected output shape."""

    def _run_model(self, shape: KernelShape) -> torch.Tensor:
        model = HARNESS.make_kernel_model(shape)
        sample = HARNESS._make_sample_input(shape)
        with torch.no_grad():
            return cast(torch.Tensor, model(sample))

    def test_layernorm_output_shape(self) -> None:
        shape = _make_shape(
            kernel_type="layernorm", batch_size=2, seq_len=32, d_model=64, n_heads=4
        )
        out = self._run_model(shape)
        assert out.shape == (2, 32, 64)

    def test_rmsnorm_output_shape(self) -> None:
        shape = _make_shape(
            kernel_type="rmsnorm", batch_size=2, seq_len=32, d_model=64, n_heads=4
        )
        out = self._run_model(shape)
        assert out.shape == (2, 32, 64)

    def test_softmax_output_shape(self) -> None:
        shape = _make_shape(
            kernel_type="softmax", batch_size=2, seq_len=32, d_model=64, n_heads=4
        )
        out = self._run_model(shape)
        assert out.shape == (2, 4, 32, 32)

    def test_gelu_output_shape(self) -> None:
        shape = _make_shape(
            kernel_type="gelu", batch_size=2, seq_len=32, d_model=64, n_heads=4
        )
        out = self._run_model(shape)
        assert out.shape == (2, 32, 64)

    def test_attn_qk_output_shape(self) -> None:
        # Output should be (batch, n_heads, seq_len, seq_len)
        shape = _make_shape(
            kernel_type="attn_qk", batch_size=2, seq_len=16, d_model=64, n_heads=4
        )
        out = self._run_model(shape)
        assert out.shape == (2, 4, 16, 16)

    def test_attn_av_output_shape(self) -> None:
        # Output should be (batch, n_heads, seq_len, head_dim)
        shape = _make_shape(
            kernel_type="attn_av", batch_size=2, seq_len=16, d_model=64, n_heads=4
        )
        out = self._run_model(shape)
        head_dim = 64 // 4
        assert out.shape == (2, 4, 16, head_dim)

    def test_ffn_output_shape(self) -> None:
        shape = _make_shape(
            kernel_type="ffn",
            batch_size=2,
            seq_len=32,
            d_model=64,
            n_heads=4,
            ff_mult=4.0,
        )
        out = self._run_model(shape)
        assert out.shape == (2, 32, 64)

    def test_all_kernel_types_forward_no_error(self) -> None:
        for kt in _VALID_KERNEL_TYPES:
            shape = _make_shape(kernel_type=kt)
            model = HARNESS.make_kernel_model(shape)
            assert model is not None

    def test_model_is_in_eval_mode(self) -> None:
        for kt in _VALID_KERNEL_TYPES:
            model = HARNESS.make_kernel_model(_make_shape(kernel_type=kt))
            assert not model.training, f"{kt} model should be in eval mode"

    def test_unknown_kernel_type_raises(self) -> None:
        with pytest.raises(ValueError):
            HARNESS.make_kernel_model(_make_shape(kernel_type="layernorm"))
            # Force the error path by monkey-patching shape kernel_type
            bad_shape = _make_shape()
            object.__setattr__(bad_shape, "kernel_type", "nonexistent")
            HARNESS.make_kernel_model(bad_shape)


# ---------------------------------------------------------------------------
# KernelHarness.benchmark_shape — ONNX proxy only (prove=False)
# ---------------------------------------------------------------------------


class TestBenchmarkShape:
    def test_returns_kernel_benchmark_instance(self) -> None:
        shape = _make_shape(kernel_type="gelu")
        result = HARNESS.benchmark_shape(shape, prove=False)
        assert isinstance(result, KernelBenchmark)

    def test_onnx_estimate_is_positive_integer(self) -> None:
        shape = _make_shape(kernel_type="layernorm")
        result = HARNESS.benchmark_shape(shape, prove=False)
        assert result.error is None or result.onnx_estimate > 0
        if result.error is None:
            assert isinstance(result.onnx_estimate, int)
            assert result.onnx_estimate > 0

    def test_no_ezkl_fields_when_not_proved(self) -> None:
        shape = _make_shape(kernel_type="softmax")
        result = HARNESS.benchmark_shape(shape, prove=False)
        assert result.ezkl_num_rows is None
        assert result.ezkl_prove_time is None

    def test_all_kernels_return_positive_onnx_estimate(self) -> None:
        for kt in _VALID_KERNEL_TYPES:
            shape = _make_shape(kernel_type=kt)
            result = HARNESS.benchmark_shape(shape, prove=False)
            assert (
                result.error is None
            ), f"Kernel {kt!r} benchmark failed: {result.error}"
            assert (
                result.onnx_estimate > 0
            ), f"Kernel {kt!r} returned non-positive onnx_estimate: {result.onnx_estimate}"

    def test_shape_preserved_in_result(self) -> None:
        shape = _make_shape(kernel_type="ffn")
        result = HARNESS.benchmark_shape(shape, prove=False)
        assert result.shape is shape

    def test_ffn_estimate_larger_than_gelu(self) -> None:
        """FFN (2-layer MLP) should cost more than bare GELU on same shape."""
        shape_gelu = _make_shape(kernel_type="gelu", d_model=256, n_heads=8)
        shape_ffn = _make_shape(kernel_type="ffn", d_model=256, n_heads=8)
        r_gelu = HARNESS.benchmark_shape(shape_gelu, prove=False)
        r_ffn = HARNESS.benchmark_shape(shape_ffn, prove=False)
        if r_gelu.error is None and r_ffn.error is None:
            assert r_ffn.onnx_estimate > r_gelu.onnx_estimate


# ---------------------------------------------------------------------------
# KernelHarness.to_cost_table
# ---------------------------------------------------------------------------


class TestToCostTable:
    def _make_benchmark(
        self,
        kernel_type: str = "gelu",
        seq_len: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        ff_mult: float = 4.0,
        onnx_estimate: int = 100_000,
        ezkl_num_rows: Optional[int] = None,
        error: Optional[str] = None,
    ) -> KernelBenchmark:
        shape = KernelShape(
            kernel_type=kernel_type,
            batch_size=1,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            ff_mult=ff_mult,
        )
        return KernelBenchmark(
            shape=shape,
            onnx_estimate=onnx_estimate,
            ezkl_num_rows=ezkl_num_rows,
            ezkl_prove_time=None,
            error=error,
        )

    def test_returns_dict(self) -> None:
        bm = self._make_benchmark()
        table = HARNESS.to_cost_table([bm])
        assert isinstance(table, dict)

    def test_key_format(self) -> None:
        bm = self._make_benchmark(
            kernel_type="gelu",
            seq_len=64,
            d_model=256,
            n_heads=8,
            ff_mult=4.0,
        )
        table = HARNESS.to_cost_table([bm])
        assert "gelu_64_256_8_4.0" in table

    def test_ezkl_rows_preferred_over_onnx(self) -> None:
        bm = self._make_benchmark(onnx_estimate=100_000, ezkl_num_rows=524_288)
        table = HARNESS.to_cost_table([bm])
        key = "gelu_64_256_8_4.0"
        assert table[key] == 524_288

    def test_onnx_used_when_no_ezkl(self) -> None:
        bm = self._make_benchmark(onnx_estimate=100_000, ezkl_num_rows=None)
        table = HARNESS.to_cost_table([bm])
        key = "gelu_64_256_8_4.0"
        assert table[key] == 100_000

    def test_negative_onnx_estimate_excluded(self) -> None:
        bm = self._make_benchmark(onnx_estimate=-1, ezkl_num_rows=None)
        table = HARNESS.to_cost_table([bm])
        assert len(table) == 0

    def test_multiple_kernels_produce_multiple_keys(self) -> None:
        bms = [
            self._make_benchmark(kernel_type=kt)
            for kt in ["gelu", "layernorm", "softmax"]
        ]
        table = HARNESS.to_cost_table(bms)
        assert len(table) == 3

    def test_empty_input_returns_empty_table(self) -> None:
        table = HARNESS.to_cost_table([])
        assert table == {}

    def test_all_seven_kernels_from_benchmark_shape(self) -> None:
        bms = [
            HARNESS.benchmark_shape(_make_shape(kernel_type=kt), prove=False)
            for kt in sorted(_VALID_KERNEL_TYPES)
        ]
        table = HARNESS.to_cost_table(bms)
        assert len(table) == len(_VALID_KERNEL_TYPES)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def _make_benchmarks(self) -> list[KernelBenchmark]:
        benchmarks = []
        for kt in sorted(_VALID_KERNEL_TYPES):
            shape = _make_shape(kernel_type=kt)
            bm = HARNESS.benchmark_shape(shape, prove=False)
            benchmarks.append(bm)
        return benchmarks

    def test_save_creates_file(self) -> None:
        bms = self._make_benchmarks()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save(bms, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_round_trip_preserves_count(self) -> None:
        bms = self._make_benchmarks()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save(bms, path)
            loaded = KernelHarness.load(path)
            assert len(loaded) == len(bms)
        finally:
            os.unlink(path)

    def test_round_trip_preserves_kernel_types(self) -> None:
        bms = self._make_benchmarks()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save(bms, path)
            loaded = KernelHarness.load(path)
            orig_types = {bm.shape.kernel_type for bm in bms}
            loaded_types = {bm.shape.kernel_type for bm in loaded}
            assert orig_types == loaded_types
        finally:
            os.unlink(path)

    def test_round_trip_preserves_onnx_estimates(self) -> None:
        bms = self._make_benchmarks()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save(bms, path)
            loaded = KernelHarness.load(path)
            orig_map = {bm.shape.kernel_type: bm.onnx_estimate for bm in bms}
            loaded_map = {bm.shape.kernel_type: bm.onnx_estimate for bm in loaded}
            assert orig_map == loaded_map
        finally:
            os.unlink(path)

    def test_round_trip_preserves_shape_fields(self) -> None:
        shape = _make_shape(
            kernel_type="ffn",
            batch_size=2,
            seq_len=128,
            d_model=512,
            n_heads=8,
            ff_mult=2.0,
        )
        bm = KernelBenchmark(
            shape=shape,
            onnx_estimate=99_999,
            ezkl_num_rows=1_048_576,
            ezkl_prove_time=42.0,
            error=None,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save([bm], path)
            loaded = KernelHarness.load(path)
            assert len(loaded) == 1
            s = loaded[0].shape
            assert s.kernel_type == "ffn"
            assert s.batch_size == 2
            assert s.seq_len == 128
            assert s.d_model == 512
            assert s.n_heads == 8
            assert s.ff_mult == pytest.approx(2.0)
            assert loaded[0].onnx_estimate == 99_999
            assert loaded[0].ezkl_num_rows == 1_048_576
            assert loaded[0].ezkl_prove_time == pytest.approx(42.0)
            assert loaded[0].error is None
        finally:
            os.unlink(path)

    def test_load_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            KernelHarness.load("/tmp/_nonexistent_kernels_test_path_.json")

    def test_saved_json_is_valid(self) -> None:
        bms = self._make_benchmarks()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save(bms, path)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert "version" in data
            assert "benchmarks" in data
            assert len(data["benchmarks"]) == len(bms)
        finally:
            os.unlink(path)

    def test_round_trip_with_ezkl_fields(self) -> None:
        shape = _make_shape(kernel_type="layernorm")
        bm = KernelBenchmark(
            shape=shape,
            onnx_estimate=50_000,
            ezkl_num_rows=262_144,
            ezkl_prove_time=30.5,
            error=None,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save([bm], path)
            loaded = KernelHarness.load(path)
            assert loaded[0].ezkl_num_rows == 262_144
            assert loaded[0].ezkl_prove_time == pytest.approx(30.5)
        finally:
            os.unlink(path)

    def test_round_trip_with_error_field(self) -> None:
        shape = _make_shape(kernel_type="rmsnorm")
        bm = KernelBenchmark(
            shape=shape,
            onnx_estimate=-1,
            ezkl_num_rows=None,
            ezkl_prove_time=None,
            error="proxy_failed: some error",
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            HARNESS.save([bm], path)
            loaded = KernelHarness.load(path)
            assert loaded[0].error == "proxy_failed: some error"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# run_grid (prove=False, small grid to keep tests fast)
# ---------------------------------------------------------------------------


class TestRunGrid:
    def test_run_grid_returns_list_of_benchmarks(self) -> None:
        results = HARNESS.run_grid(
            seq_lens=[64],
            d_models=[256],
            n_heads_list=[8],
            ff_mults=[4.0],
            prove=False,
            max_workers=2,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, KernelBenchmark) for r in results)

    def test_run_grid_covers_all_kernel_types(self) -> None:
        results = HARNESS.run_grid(
            seq_lens=[64],
            d_models=[256],
            n_heads_list=[8],
            ff_mults=[4.0],
            prove=False,
            max_workers=2,
        )
        found_types = {r.shape.kernel_type for r in results}
        assert found_types == _VALID_KERNEL_TYPES

    def test_run_grid_skips_invalid_head_combinations(self) -> None:
        # d_model=100 is not divisible by n_heads=3, so 0 valid shapes.
        results = HARNESS.run_grid(
            seq_lens=[64],
            d_models=[100],
            n_heads_list=[3],
            ff_mults=[4.0],
            prove=False,
            max_workers=1,
        )
        assert results == []

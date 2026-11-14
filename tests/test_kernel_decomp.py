"""Tests for proxy.strategies.kernel_decomp — Approach C."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from proxy.features import ProxyFeatures, TransformerConfig
from proxy.strategies.kernel_decomp import KernelDecompositionProxy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 2,
    seq_len: int = 64,
    ff_mult: float = 4.0,
    activation: str = "relu",
) -> TransformerConfig:
    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        ff_mult=ff_mult,
        activation=activation,
    )


def _full_cost_table(config: TransformerConfig) -> dict:
    """Build a cost table that covers all kernels required by *config*."""
    s = config.seq_len
    d = config.d_model
    h = config.n_heads
    f = config.ff_mult
    return {
        ("attn_qk", s, d, h, f): 1_000_000,
        ("attn_av", s, d, h, f): 800_000,
        ("softmax", s, 1, h, 1.0): 200_000,
        ("layernorm", s, d, 1, 1.0): 150_000,
        ("ffn", s, d, 1, f): 3_000_000,
        ("embedding", s, d, 1, 1.0): 500_000,
    }


def _make_proxy_with_full_table(config: Optional[TransformerConfig] = None) -> KernelDecompositionProxy:
    cfg = config or _make_config()
    return KernelDecompositionProxy(cost_table=_full_cost_table(cfg), fallback_to_onnx=False)


def _make_dataset_with_records(proxy: KernelDecompositionProxy, configs: list[TransformerConfig]):
    """Create a ProxyDataset populated with synthetic labelled records."""
    from proxy.data import ProxyDataset, ProxyRecord

    dataset = ProxyDataset()
    for cfg in configs:
        features = ProxyFeatures.from_config(cfg)
        # Use the proxy's own prediction scaled by a constant as ground truth
        # to ensure fit() has signal to calibrate against.
        predicted = proxy.decompose(cfg)["total"]
        ground_truth_rows = int(predicted * 2)  # target scale = 2x log-ratio
        record = ProxyRecord(
            config=cfg,
            features=features,
            ezkl_num_rows=ground_truth_rows,
        )
        dataset.add(record)
    return dataset


# ---------------------------------------------------------------------------
# decompose — structure and values
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_returns_dict_with_required_keys(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        result = proxy.decompose(cfg)

        required_keys = {
            "attn_qk",
            "attn_av",
            "softmax",
            "layernorm",
            "ffn",
            "embedding",
            "total",
            "log_total",
            "fallback_fraction",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )

    def test_total_equals_sum_of_kernels(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        result = proxy.decompose(cfg)

        component_sum = (
            result["attn_qk"]
            + result["attn_av"]
            + result["softmax"]
            + result["layernorm"]
            + result["ffn"]
            + result["embedding"]
        )
        assert result["total"] == pytest.approx(component_sum, rel=1e-9)

    def test_log_total_matches_log_of_total(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        result = proxy.decompose(cfg)
        assert result["log_total"] == pytest.approx(math.log(result["total"]), rel=1e-9)

    def test_per_layer_costs_multiplied_by_n_layers(self) -> None:
        cfg = _make_config(n_layers=3)
        proxy = _make_proxy_with_full_table(cfg)
        result_3 = proxy.decompose(cfg)

        cfg1 = _make_config(n_layers=1)
        proxy1 = _make_proxy_with_full_table(cfg1)
        result_1 = proxy1.decompose(cfg1)

        # attn_qk for n_layers=3 should be 3× the single-layer cost.
        assert result_3["attn_qk"] == pytest.approx(3 * result_1["attn_qk"], rel=1e-9)

    def test_layernorm_doubled_per_layer(self) -> None:
        """LayerNorm appears twice per block — cost must reflect that."""
        cfg = _make_config(n_layers=1)
        s, d = cfg.seq_len, cfg.d_model
        single_ln_cost = 150_000
        table = {
            ("attn_qk", s, d, cfg.n_heads, cfg.ff_mult): 1,
            ("attn_av", s, d, cfg.n_heads, cfg.ff_mult): 1,
            ("softmax", s, 1, cfg.n_heads, 1.0): 1,
            ("layernorm", s, d, 1, 1.0): single_ln_cost,
            ("ffn", s, d, 1, cfg.ff_mult): 1,
            ("embedding", s, d, 1, 1.0): 1,
        }
        proxy = KernelDecompositionProxy(cost_table=table, fallback_to_onnx=False)
        result = proxy.decompose(cfg)
        # n_layers=1, two ln per layer → total layernorm cost = 2 * single_ln_cost
        assert result["layernorm"] == pytest.approx(2 * single_ln_cost, rel=1e-9)

    def test_fallback_fraction_zero_when_all_in_table(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        result = proxy.decompose(cfg)
        assert result["fallback_fraction"] == pytest.approx(0.0)

    def test_values_are_finite_floats(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        result = proxy.decompose(cfg)
        for key, val in result.items():
            assert math.isfinite(val), f"Non-finite value for key={key!r}: {val}"


# ---------------------------------------------------------------------------
# score — basic correctness
# ---------------------------------------------------------------------------


class TestScore:
    def test_returns_finite_float(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        s = proxy.score(cfg)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_score_equals_log_total_before_fit(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        breakdown = proxy.decompose(cfg)
        expected = breakdown["log_total"]
        assert proxy.score(cfg) == pytest.approx(expected, rel=1e-9)

    def test_larger_model_has_higher_score(self) -> None:
        small_cfg = _make_config(d_model=128, n_heads=4, n_layers=2, seq_len=64)
        big_cfg = _make_config(d_model=512, n_heads=8, n_layers=6, seq_len=128)

        small_table = _full_cost_table(small_cfg)
        big_table = _full_cost_table(big_cfg)
        # Use slightly higher costs for the big model to reflect larger circuits.
        big_table = {k: v * 10 for k, v in big_table.items()}

        small_proxy = KernelDecompositionProxy(cost_table=small_table, fallback_to_onnx=False)
        big_proxy = KernelDecompositionProxy(cost_table=big_table, fallback_to_onnx=False)

        assert big_proxy.score(big_cfg) > small_proxy.score(small_cfg)


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_uncertainty_is_one_when_cost_table_empty(self) -> None:
        """Empty table → all kernels fall back → uncertainty = 1.0."""
        cfg = _make_config()
        # Build a proxy with empty table and a mock ONNX fallback so it
        # doesn't crash when looking up costs.
        proxy = KernelDecompositionProxy(cost_table={}, fallback_to_onnx=True)

        with patch.object(proxy, "_onnx_fallback", return_value=1_000_000.0):
            u = proxy.uncertainty(cfg)

        assert u == pytest.approx(1.0)

    def test_uncertainty_is_zero_when_all_kernels_in_table(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        assert proxy.uncertainty(cfg) == pytest.approx(0.0)

    def test_uncertainty_between_zero_and_one_for_partial_table(self) -> None:
        cfg = _make_config()
        # Provide only some kernels; the rest will fall back.
        s, d, h, f = cfg.seq_len, cfg.d_model, cfg.n_heads, cfg.ff_mult
        partial_table = {
            ("attn_qk", s, d, h, f): 1_000_000,
            ("attn_av", s, d, h, f): 800_000,
            # softmax, layernorm, ffn, embedding are missing → will fall back
        }
        proxy = KernelDecompositionProxy(cost_table=partial_table, fallback_to_onnx=True)
        with patch.object(proxy, "_onnx_fallback", return_value=500_000.0):
            u = proxy.uncertainty(cfg)

        assert 0.0 < u < 1.0

    def test_fallback_raises_when_fallback_disabled(self) -> None:
        cfg = _make_config()
        proxy = KernelDecompositionProxy(cost_table={}, fallback_to_onnx=False)
        with pytest.raises(KeyError):
            proxy.uncertainty(cfg)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip_preserves_cost_table(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            proxy.save(path)
            loaded = KernelDecompositionProxy.load(path)

            # Scores should be identical.
            assert loaded.score(cfg) == pytest.approx(proxy.score(cfg), rel=1e-9)
        finally:
            os.unlink(path)

    def test_round_trip_preserves_interpolation_mode(self) -> None:
        cfg = _make_config()
        proxy = KernelDecompositionProxy(
            cost_table=_full_cost_table(cfg),
            interpolation="linear",
            fallback_to_onnx=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            proxy.save(path)
            loaded = KernelDecompositionProxy.load(path)
            assert loaded._interpolation == "linear"
        finally:
            os.unlink(path)

    def test_round_trip_preserves_scale_factor(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        proxy._scale = 2.5
        proxy._fitted = True

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            proxy.save(path)
            loaded = KernelDecompositionProxy.load(path)
            assert loaded._scale == pytest.approx(2.5)
            assert loaded._fitted is True
        finally:
            os.unlink(path)

    def test_round_trip_preserves_fallback_flag(self) -> None:
        cfg = _make_config()
        proxy = KernelDecompositionProxy(
            cost_table=_full_cost_table(cfg),
            fallback_to_onnx=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            proxy.save(path)
            loaded = KernelDecompositionProxy.load(path)
            assert loaded._fallback_to_onnx is False
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            KernelDecompositionProxy.load("/tmp/__does_not_exist_zkml__.json")


# ---------------------------------------------------------------------------
# fit — scale factor calibration
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_sets_fitted_flag(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        # Proxy is not marked fitted until fit() is called (cost_table is non-empty
        # so is_fitted() returns True, but _fitted flag is False before fit()).
        assert proxy._fitted is False
        dataset = _make_dataset_with_records(proxy, [cfg])
        proxy.fit(dataset)
        assert proxy._fitted is True

    def test_fit_changes_scale_factor(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        assert proxy._scale == pytest.approx(1.0)

        # Build a dataset where ground truth is 2× the raw total (in log space
        # we're asserting scale changes from 1.0).
        dataset = _make_dataset_with_records(proxy, [cfg, _make_config(n_layers=4)])
        proxy.fit(dataset)
        # Scale should no longer be 1.0 (it's calibrated from data).
        assert proxy._scale != pytest.approx(1.0, abs=0.01)

    def test_fit_with_empty_dataset_keeps_scale_at_one(self) -> None:
        from proxy.data import ProxyDataset

        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        empty_dataset = ProxyDataset()
        proxy.fit(empty_dataset)
        assert proxy._scale == pytest.approx(1.0)

    def test_fit_score_reflects_calibration(self) -> None:
        """After fit, score() should apply the scale factor."""
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)

        score_before = proxy.score(cfg)
        dataset = _make_dataset_with_records(proxy, [cfg])
        proxy.fit(dataset)
        score_after = proxy.score(cfg)

        # The scale changes so the score after fit differs from before.
        assert score_after != pytest.approx(score_before, abs=0.01)

    def test_fit_raises_on_non_dataset(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        with pytest.raises(TypeError):
            proxy.fit({"not": "a dataset"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# is_fitted
# ---------------------------------------------------------------------------


class TestIsFitted:
    def test_is_fitted_true_with_non_empty_table(self) -> None:
        cfg = _make_config()
        proxy = _make_proxy_with_full_table(cfg)
        assert proxy.is_fitted() is True

    def test_is_fitted_false_with_empty_table_before_fit(self) -> None:
        proxy = KernelDecompositionProxy(cost_table={}, fallback_to_onnx=True)
        assert proxy.is_fitted() is False

    def test_is_fitted_true_after_fit(self) -> None:
        from proxy.data import ProxyDataset

        proxy = KernelDecompositionProxy(cost_table={}, fallback_to_onnx=True)
        proxy.fit(ProxyDataset())
        assert proxy.is_fitted() is True


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    def _build_table_with_two_seq_lens(
        self, seq_lo: int, seq_hi: int, cost_lo: int, cost_hi: int
    ) -> dict:
        """Build a minimal table with two seq_len grid points for attn_qk."""
        return {
            ("attn_qk", seq_lo, 256, 4, 4.0): cost_lo,
            ("attn_qk", seq_hi, 256, 4, 4.0): cost_hi,
            ("attn_av", seq_lo, 256, 4, 4.0): 100,
            ("attn_av", seq_hi, 256, 4, 4.0): 100,
            ("softmax", seq_lo, 1, 4, 1.0): 100,
            ("softmax", seq_hi, 1, 4, 1.0): 100,
            ("layernorm", seq_lo, 256, 1, 1.0): 100,
            ("layernorm", seq_hi, 256, 1, 1.0): 100,
            ("ffn", seq_lo, 256, 1, 4.0): 100,
            ("ffn", seq_hi, 256, 1, 4.0): 100,
            ("embedding", seq_lo, 256, 1, 1.0): 100,
            ("embedding", seq_hi, 256, 1, 1.0): 100,
        }

    def test_nearest_snaps_to_grid(self) -> None:
        table = self._build_table_with_two_seq_lens(64, 128, 1_000, 2_000)
        proxy = KernelDecompositionProxy(
            cost_table=table, interpolation="nearest", fallback_to_onnx=False
        )
        # seq_len=70 is closer to 64 than 128, so it snaps to cost_lo.
        cfg_70 = _make_config(seq_len=70)
        cfg_64 = _make_config(seq_len=64)
        assert proxy.decompose(cfg_70)["attn_qk"] == proxy.decompose(cfg_64)["attn_qk"]

    def test_linear_interpolates_midpoint(self) -> None:
        table = self._build_table_with_two_seq_lens(64, 128, 1_000, 2_000)
        proxy = KernelDecompositionProxy(
            cost_table=table, interpolation="linear", fallback_to_onnx=False
        )
        cfg_96 = _make_config(seq_len=96)
        breakdown = proxy.decompose(cfg_96)
        # attn_qk for n_layers=2, midpoint cost = 1500, total = 1500 * n_layers
        expected_attn_qk = 1_500.0 * cfg_96.n_layers
        assert breakdown["attn_qk"] == pytest.approx(expected_attn_qk, rel=1e-9)


# ---------------------------------------------------------------------------
# from_kernel_harness_path
# ---------------------------------------------------------------------------


class TestFromKernelHarnessPath:
    def test_loads_pipe_separated_keys(self) -> None:
        import json
        import tempfile
        import os

        table_data = {
            "attn_qk|64|256|4|4.0": 1_000_000,
            "attn_av|64|256|4|4.0": 800_000,
            "softmax|64|1|4|1.0": 200_000,
            "layernorm|64|256|1|1.0": 150_000,
            "ffn|64|256|1|4.0": 3_000_000,
            "embedding|64|256|1|1.0": 500_000,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(table_data, f)
            path = f.name

        try:
            proxy = KernelDecompositionProxy.from_kernel_harness_path(path)
            assert proxy.is_fitted()
            cfg = _make_config()
            result = proxy.decompose(cfg)
            assert result["total"] > 0
        finally:
            os.unlink(path)

    def test_invalid_interpolation_raises(self) -> None:
        with pytest.raises(ValueError, match="interpolation"):
            KernelDecompositionProxy(interpolation="cubic")

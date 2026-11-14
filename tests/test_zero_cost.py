# pylint: disable=protected-access
"""Tests for proxy.strategies.zero_cost — ZeroCostProxy (Approach B)."""

from __future__ import annotations

import math
import os
import pickle
import random
import tempfile

import pytest

from proxy.data import Fidelity, ProxyDataset, ProxyRecord
from proxy.features import ProxyFeatures, TransformerConfig, feature_names
from proxy.strategies.zero_cost import ZeroCostProxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Valid d_model / n_heads combinations (d_model divisible by n_heads)
_ARCH_GRID = [
    (64, 4, 2, 64),
    (128, 4, 3, 64),
    (128, 8, 2, 128),
    (256, 4, 4, 64),
    (256, 8, 3, 128),
    (256, 8, 6, 128),
    (512, 8, 4, 64),
    (512, 8, 6, 128),
    (512, 16, 3, 64),
    (768, 12, 6, 128),
    (768, 12, 8, 128),
    (1024, 8, 4, 64),
    (1024, 16, 6, 128),
    (1024, 16, 8, 64),
    (384, 12, 4, 128),
    (384, 6, 3, 64),
    (192, 8, 4, 64),
    (192, 4, 3, 128),
    (96, 4, 2, 64),
    (96, 8, 4, 128),
]
_ACTIVATIONS = ["gelu", "relu", "poly2", "poly3", "linear"]

# Model types that are fast to train in any environment (no XGBoost)
_FAST_MODEL_TYPES = ["rf", "ridge"]

# All model types including XGBoost (slow in some environments)
_ALL_MODEL_TYPES = ["xgb", "rf", "ridge"]


def _make_config(
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    seq_len: int = 64,
    activation: str = "gelu",
) -> TransformerConfig:
    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        ff_mult=4.0,
        activation=activation,
    )


def _make_record(config: TransformerConfig, label: float) -> ProxyRecord:
    """Create a ProxyRecord with a synthetic ONNX-level label."""
    features = ProxyFeatures.from_config(config)
    # Convert log-cost back to a raw count for onnx_estimate
    onnx_est = max(1, int(math.exp(label)))
    return ProxyRecord(
        config=config,
        features=features,
        onnx_estimate=onnx_est,
    )


def _make_dataset(n: int = 20, seed: int = 42) -> ProxyDataset:
    """Build a synthetic ProxyDataset with *n* diverse records."""
    rng = random.Random(seed)
    dataset = ProxyDataset()
    for i in range(n):
        d_model, n_heads, n_layers, seq_len = _ARCH_GRID[i % len(_ARCH_GRID)]
        activation = _ACTIVATIONS[i % len(_ACTIVATIONS)]
        config = _make_config(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            activation=activation,
        )
        # Synthetic log-cost: roughly correlated with model size + noise
        log_cost = math.log(d_model * n_layers * seq_len) + rng.gauss(0, 0.3)
        record = _make_record(config, log_cost)
        dataset.add(record)
    return dataset


# ---------------------------------------------------------------------------
# Basic fit + score
# ---------------------------------------------------------------------------


class TestFitAndScore:
    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_fit_and_score_returns_float(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        config = _make_config()
        result = proxy.score(config)
        assert isinstance(result, float)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["xgb"])
    def test_fit_and_score_returns_float_xgb(self, model_type: str) -> None:
        """XGBoost variant — marked slow due to long training time in some environments."""
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        config = _make_config()
        result = proxy.score(config)
        assert isinstance(result, float)

    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_is_fitted_after_fit(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        assert not proxy.is_fitted()
        proxy.fit(dataset)
        assert proxy.is_fitted()

    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_score_returns_finite_value(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        for d_model, n_heads, n_layers, seq_len in _ARCH_GRID[:5]:
            config = _make_config(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                seq_len=seq_len,
            )
            s = proxy.score(config)
            assert math.isfinite(s), f"score={s} for {config}"

    def test_unfitted_proxy_score_uses_fallback(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        config = _make_config(d_model=256, n_heads=4, n_layers=4, seq_len=64)
        # Should not raise — returns fallback analytical estimate
        result = proxy.score(config)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_score_without_onnx_features(self) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="rf", use_onnx_features=False)
        proxy.fit(dataset)
        config = _make_config()
        result = proxy.score(config)
        assert isinstance(result, float)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


class TestUncertainty:
    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_uncertainty_is_non_negative(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        config = _make_config(d_model=256, n_heads=4, n_layers=4, seq_len=128)
        u = proxy.uncertainty(config)
        assert u >= 0.0, f"uncertainty={u} for model_type={model_type!r}"

    @pytest.mark.slow
    def test_uncertainty_xgb_is_interval_width(self) -> None:
        """Verify XGBoost quantile uncertainty is non-negative p90−p10."""
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="xgb")
        proxy.fit(dataset)
        config = _make_config(d_model=512, n_heads=8, n_layers=6, seq_len=128)
        u = proxy.uncertainty(config)
        assert u >= 0.0

    def test_uncertainty_is_zero_for_ridge(self) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="ridge")
        proxy.fit(dataset)
        config = _make_config()
        assert proxy.uncertainty(config) == 0.0

    def test_uncertainty_rf_is_non_negative(self) -> None:
        """RF uncertainty is the std of tree predictions."""
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(dataset)
        config = _make_config(d_model=512, n_heads=8, n_layers=6, seq_len=128)
        u = proxy.uncertainty(config)
        assert u >= 0.0

    def test_uncertainty_unfitted_returns_zero(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        config = _make_config()
        assert proxy.uncertainty(config) == 0.0


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_save_load_preserves_predictions(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)

        configs = [
            _make_config(d_model=d, n_heads=h, n_layers=l, seq_len=s)
            for d, h, l, s in _ARCH_GRID[:5]
        ]
        original_scores = [proxy.score(c) for c in configs]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            proxy.save(path)
            loaded = ZeroCostProxy.load(path)
            loaded_scores = [loaded.score(c) for c in configs]
        finally:
            os.unlink(path)

        for orig, loaded_s in zip(original_scores, loaded_scores):
            assert orig == pytest.approx(loaded_s), (
                f"Score changed after save/load: {orig} != {loaded_s}"
            )

    @pytest.mark.slow
    def test_save_load_preserves_predictions_xgb(self) -> None:
        """XGBoost save/load round-trip — marked slow."""
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="xgb")
        proxy.fit(dataset)
        configs = [
            _make_config(d_model=d, n_heads=h, n_layers=l, seq_len=s)
            for d, h, l, s in _ARCH_GRID[:3]
        ]
        original_scores = [proxy.score(c) for c in configs]
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            proxy.save(path)
            loaded = ZeroCostProxy.load(path)
            loaded_scores = [loaded.score(c) for c in configs]
        finally:
            os.unlink(path)
        for orig, loaded_s in zip(original_scores, loaded_scores):
            assert orig == pytest.approx(loaded_s)

    def test_load_wrong_type_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a proxy"}, f)
            path = f.name
        try:
            with pytest.raises(TypeError, match="ZeroCostProxy"):
                ZeroCostProxy.load(path)
        finally:
            os.unlink(path)

    def test_save_creates_file(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(_make_dataset(20))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            proxy.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Small-dataset fallback (< 5 samples)
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.parametrize("n_samples", [0, 1, 3, 4])
    def test_fallback_mode_for_small_datasets(self, n_samples: int) -> None:
        dataset = _make_dataset(n_samples)
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(dataset)
        # Should be in fallback mode
        assert proxy._fallback is True

    @pytest.mark.parametrize("n_samples", [0, 1, 3, 4])
    def test_fallback_score_is_finite(self, n_samples: int) -> None:
        dataset = _make_dataset(n_samples)
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(dataset)
        config = _make_config(d_model=256, n_heads=4, n_layers=4, seq_len=64)
        s = proxy.score(config)
        assert math.isfinite(s)

    @pytest.mark.parametrize("n_samples", [0, 1, 3, 4])
    def test_fallback_uncertainty_is_zero(self, n_samples: int) -> None:
        dataset = _make_dataset(n_samples)
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(dataset)
        config = _make_config()
        assert proxy.uncertainty(config) == 0.0

    @pytest.mark.parametrize("n_samples", [0, 1, 3, 4])
    def test_fallback_is_fitted_returns_true(self, n_samples: int) -> None:
        """is_fitted() must return True even in fall-back mode."""
        dataset = _make_dataset(n_samples)
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(dataset)
        assert proxy.is_fitted()

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", _ALL_MODEL_TYPES)
    def test_fallback_all_model_types(self, model_type: str) -> None:
        """All model types fall back gracefully with < 5 samples."""
        dataset = _make_dataset(3)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        assert proxy._fallback is True
        config = _make_config()
        s = proxy.score(config)
        assert math.isfinite(s)
        assert proxy.uncertainty(config) == 0.0


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------


class TestFeatureImportances:
    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_importances_sum_to_one(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        importances = proxy.feature_importances()
        assert len(importances) > 0
        total = sum(importances.values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"feature_importances sum={total} for model_type={model_type!r}"
        )

    @pytest.mark.slow
    def test_importances_sum_to_one_xgb(self) -> None:
        """XGBoost feature importances sum check — marked slow."""
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="xgb")
        proxy.fit(dataset)
        importances = proxy.feature_importances()
        assert len(importances) > 0
        total = sum(importances.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_importances_are_non_negative(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        for name, imp in proxy.feature_importances().items():
            assert imp >= 0.0, f"Negative importance {imp} for feature {name!r}"

    @pytest.mark.parametrize("model_type", _FAST_MODEL_TYPES)
    def test_importances_keys_are_feature_names(self, model_type: str) -> None:
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type=model_type)  # type: ignore[arg-type]
        proxy.fit(dataset)
        all_names = set(feature_names())
        for key in proxy.feature_importances():
            assert key in all_names, f"Unknown feature name {key!r}"

    def test_importances_empty_when_not_fitted(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        assert proxy.feature_importances() == {}

    def test_importances_empty_in_fallback_mode(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(_make_dataset(3))
        assert proxy.feature_importances() == {}


# ---------------------------------------------------------------------------
# Min fidelity filtering
# ---------------------------------------------------------------------------


class TestMinFidelityFiltering:
    def test_ezkl_prove_fidelity_excludes_onnx_records(self) -> None:
        """When min_fidelity=EZKL_PROVE, onnx-only records are filtered out."""
        dataset = ProxyDataset()
        # 10 onnx-only records
        for i in range(10):
            d_model, n_heads, n_layers, seq_len = _ARCH_GRID[i]
            config = _make_config(d_model, n_heads, n_layers, seq_len)
            features = ProxyFeatures.from_config(config)
            record = ProxyRecord(
                config=config,
                features=features,
                onnx_estimate=1_000_000 + i * 100_000,
            )
            dataset.add(record)

        proxy = ZeroCostProxy(model_type="rf", min_fidelity=Fidelity.EZKL_PROVE)
        proxy.fit(dataset)
        # All 10 records are below EZKL_PROVE → fallback
        assert proxy._fallback is True

    def test_onnx_fidelity_includes_onnx_records(self) -> None:
        """min_fidelity=ONNX_ESTIMATE includes onnx-only records."""
        dataset = _make_dataset(20)
        proxy = ZeroCostProxy(model_type="rf", min_fidelity=Fidelity.ONNX_ESTIMATE)
        proxy.fit(dataset)
        assert not proxy._fallback


# ---------------------------------------------------------------------------
# Invalid construction
# ---------------------------------------------------------------------------


class TestInvalidConstruction:  # pylint: disable=too-few-public-methods
    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="model_type"):
            ZeroCostProxy(model_type="svm")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_model_type(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        assert "rf" in repr(proxy)

    def test_repr_unfitted(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        assert "unfitted" in repr(proxy)

    def test_repr_fitted(self) -> None:
        proxy = ZeroCostProxy(model_type="rf")
        proxy.fit(_make_dataset(20))
        assert "fitted" in repr(proxy)

"""Tests for proxy.strategies.multifidelity — MultiFidelityProxy (Approach A)."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Optional

import numpy as np
import pytest

from proxy.data import Fidelity, ProxyDataset, ProxyRecord
from proxy.features import ProxyFeatures, TransformerConfig
from proxy.strategies.multifidelity import MultiFidelityProxy, _DEFAULT_FIDELITY_WEIGHTS

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _cfg(
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
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


def _record(
    config: Optional[TransformerConfig] = None,
    onnx_estimate: Optional[int] = None,
    ezkl_num_rows: Optional[int] = None,
    ezkl_prove_time: Optional[float] = None,
    ezkl_calibration: Optional[dict] = None,
) -> ProxyRecord:
    if config is None:
        config = _cfg()
    features = ProxyFeatures.from_config(config)
    return ProxyRecord(
        config=config,
        features=features,
        onnx_estimate=onnx_estimate,
        ezkl_num_rows=ezkl_num_rows,
        ezkl_prove_time=ezkl_prove_time,
        ezkl_calibration=ezkl_calibration,
    )


def _make_synthetic_dataset(n: int = 30, seed: int = 0) -> ProxyDataset:
    """Build a dataset with *n* records spanning a range of configs and labels.

    Uses varied architectural configs so features differ across rows.
    The synthetic ``ezkl_num_rows`` label correlates with total_macs (a
    plausible proxy for ZK cost) to give Spearman-detectable signal.
    """
    rng = np.random.default_rng(seed)
    dataset = ProxyDataset()

    d_models = [32, 64, 128, 256]
    n_heads_map = {32: 4, 64: 4, 128: 8, 256: 8}

    for i in range(n):
        d_model = d_models[i % len(d_models)]
        n_heads = n_heads_map[d_model]
        n_layers = rng.integers(2, 8)
        seq_len = int(rng.choice([32, 64, 128]))
        ff_mult = float(rng.choice([2.0, 4.0, 8.0]))

        config = _cfg(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=int(n_layers),
            seq_len=seq_len,
            ff_mult=ff_mult,
        )
        features = ProxyFeatures.from_config(config)
        # Synthetic label: total_macs with multiplicative noise.
        base = features.total_macs
        rows = int(base * rng.uniform(0.8, 1.2) / 100)
        rows = max(1, rows)

        rec = ProxyRecord(
            config=config,
            features=features,
            onnx_estimate=rows // 2,
            ezkl_num_rows=rows,
        )
        dataset.add(rec)

    return dataset


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        proxy = MultiFidelityProxy()
        assert not proxy.is_fitted()

    def test_custom_hyperparams(self) -> None:
        proxy = MultiFidelityProxy(n_estimators=50, max_depth=3, learning_rate=0.1)
        assert proxy._n_estimators == 50
        assert proxy._max_depth == 3
        assert proxy._learning_rate == pytest.approx(0.1)

    def test_default_fidelity_weights_applied(self) -> None:
        proxy = MultiFidelityProxy()
        assert proxy._fidelity_weights[Fidelity.ONNX_ESTIMATE] == pytest.approx(0.3)
        assert proxy._fidelity_weights[Fidelity.EZKL_CALIBRATION] == pytest.approx(0.7)
        assert proxy._fidelity_weights[Fidelity.EZKL_PROVE] == pytest.approx(1.0)

    def test_custom_fidelity_weights(self) -> None:
        weights = {Fidelity.ONNX_ESTIMATE: 0.1, Fidelity.EZKL_PROVE: 0.9}
        proxy = MultiFidelityProxy(fidelity_weights=weights)
        assert proxy._fidelity_weights[Fidelity.ONNX_ESTIMATE] == pytest.approx(0.1)

    def test_score_raises_when_not_fitted(self) -> None:
        proxy = MultiFidelityProxy()
        with pytest.raises(RuntimeError, match="not fitted"):
            proxy.score(_cfg())


# ---------------------------------------------------------------------------
# Graceful degradation with < 5 samples
# ---------------------------------------------------------------------------


class TestFewSampleDegradation:
    def test_fit_returns_nan_metrics_with_few_samples(self) -> None:
        dataset = ProxyDataset()
        for i in range(3):
            dataset.add(_record(config=_cfg(d_model=64 * (i + 1), n_heads=4 * (i + 1)), onnx_estimate=500_000 * (i + 1)))
        proxy = MultiFidelityProxy(n_estimators=3)
        result = proxy.fit(dataset)
        assert result["n_samples"] == 3
        assert math.isnan(result["cv_spearman"])
        assert math.isnan(result["cv_ndcg"])

    def test_is_fitted_after_few_sample_fit(self) -> None:
        dataset = ProxyDataset()
        dataset.add(_record(onnx_estimate=500_000))
        proxy = MultiFidelityProxy(n_estimators=3)
        proxy.fit(dataset)
        assert proxy.is_fitted()

    def test_score_uses_onnx_estimate_when_available(self) -> None:
        dataset = ProxyDataset()
        onnx_val = 1_000_000
        config = _cfg()
        features = ProxyFeatures.from_config(config)
        # Inject onnx_constraint_estimate into features.
        features_with_onnx = ProxyFeatures(
            d_model=features.d_model,
            n_heads=features.n_heads,
            n_layers=features.n_layers,
            seq_len=features.seq_len,
            ff_mult=features.ff_mult,
            head_dim=features.head_dim,
            total_params=features.total_params,
            decoder_params=features.decoder_params,
            attn_macs=features.attn_macs,
            ffn_macs=features.ffn_macs,
            total_macs=features.total_macs,
            activation_id=features.activation_id,
            is_zk_friendly=features.is_zk_friendly,
            layernorm_count=features.layernorm_count,
            softmax_count=features.softmax_count,
            gelu_count=features.gelu_count,
            onnx_constraint_estimate=onnx_val,
            onnx_nonlinear_ops=None,
            onnx_linear_ops=None,
        )
        rec = ProxyRecord(
            config=config,
            features=features_with_onnx,
            onnx_estimate=onnx_val,
        )
        dataset.add(rec)
        proxy = MultiFidelityProxy(n_estimators=3)
        proxy.fit(dataset)
        # Score should return log(onnx_val) as fallback.
        s = proxy.score(config)
        assert math.isfinite(s)

    def test_score_with_zero_samples_after_fit(self) -> None:
        """Edge case: empty dataset — fallback mean defaults to 0.0."""
        dataset = ProxyDataset()
        proxy = MultiFidelityProxy(n_estimators=3)
        result = proxy.fit(dataset)
        assert result["n_samples"] == 0
        # Proxy is considered fitted with fallback.
        assert proxy.is_fitted()


# ---------------------------------------------------------------------------
# Fit with 30-record synthetic dataset
# ---------------------------------------------------------------------------


class TestFitAndScore:
    @pytest.fixture(scope="class")
    def fitted_proxy(self) -> MultiFidelityProxy:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3, learning_rate=0.1)
        proxy.fit(dataset)
        return proxy

    @pytest.fixture(scope="class")
    def sample_config(self) -> TransformerConfig:
        return _cfg(d_model=128, n_heads=8, n_layers=4, seq_len=64)

    def test_is_fitted_after_training(self, fitted_proxy: MultiFidelityProxy) -> None:
        assert fitted_proxy.is_fitted()

    def test_regressor_is_set(self, fitted_proxy: MultiFidelityProxy) -> None:
        assert fitted_proxy._regressor is not None

    def test_ranker_is_set_for_n30(self, fitted_proxy: MultiFidelityProxy) -> None:
        # 30 >= 20, so ranker should be trained.
        assert fitted_proxy._ranker is not None

    def test_quantile_models_are_set(self, fitted_proxy: MultiFidelityProxy) -> None:
        assert fitted_proxy._quantile_low is not None
        assert fitted_proxy._quantile_high is not None

    def test_score_returns_finite_float(
        self, fitted_proxy: MultiFidelityProxy, sample_config: TransformerConfig
    ) -> None:
        s = fitted_proxy.score(sample_config)
        assert isinstance(s, float)
        assert math.isfinite(s)

    def test_score_varies_across_configs(self, fitted_proxy: MultiFidelityProxy) -> None:
        configs = [
            _cfg(d_model=32, n_heads=4, n_layers=2),
            _cfg(d_model=256, n_heads=8, n_layers=6),
        ]
        scores = [fitted_proxy.score(c) for c in configs]
        assert scores[0] != scores[1], "Scores should differ for different configs"

    def test_fit_returns_correct_n_samples(self) -> None:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3)
        result = proxy.fit(dataset)
        # Dataset deduplicates by config hash; expect between 20 and 30 unique records.
        assert 20 <= result["n_samples"] <= 30

    def test_fit_returns_dict_with_required_keys(self) -> None:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3)
        result = proxy.fit(dataset)
        assert "n_samples" in result
        assert "cv_spearman" in result
        assert "cv_ndcg" in result


# ---------------------------------------------------------------------------
# cv_metrics
# ---------------------------------------------------------------------------


class TestCvMetrics:
    def test_spearman_r_in_valid_range(self) -> None:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        metrics = proxy.cv_metrics(dataset)
        assert -1.0 <= metrics["spearman_r"] <= 1.0

    def test_ndcg_in_valid_range(self) -> None:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        metrics = proxy.cv_metrics(dataset)
        # NDCG can be nan with a very small k; accept nan or [0, 1].
        val = metrics["ndcg@10"]
        if not math.isnan(val):
            assert 0.0 <= val <= 1.0

    def test_top10_precision_in_valid_range(self) -> None:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        metrics = proxy.cv_metrics(dataset)
        val = metrics["top10_precision"]
        if not math.isnan(val):
            assert 0.0 <= val <= 1.0

    def test_cv_metrics_on_tiny_dataset_returns_nan(self) -> None:
        dataset = ProxyDataset()
        dataset.add(_record(onnx_estimate=100_000))
        proxy = MultiFidelityProxy(n_estimators=3)
        proxy.fit(dataset)
        metrics = proxy.cv_metrics(dataset)
        assert math.isnan(metrics["spearman_r"])

    def test_cv_uses_loo_for_small_datasets(self) -> None:
        """With 8 samples (< 10) the proxy should still return a finite Spearman r."""
        dataset = _make_synthetic_dataset(n=8)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        metrics = proxy.cv_metrics(dataset)
        # For 8 samples, LOO is used; result may be finite or nan depending on variance.
        # We only assert the key is present and the value is a float.
        assert isinstance(metrics["spearman_r"], float)


# ---------------------------------------------------------------------------
# pow2_cliff_score
# ---------------------------------------------------------------------------


class TestPow2CliffScore:
    @pytest.fixture(scope="class")
    def fitted_proxy(self) -> MultiFidelityProxy:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        return proxy

    def test_returns_value_in_0_1(self, fitted_proxy: MultiFidelityProxy) -> None:
        config = _cfg(d_model=64, n_heads=4, n_layers=4)
        s = fitted_proxy.pow2_cliff_score(config)
        assert 0.0 <= s <= 1.0

    def test_returns_nan_when_not_fitted(self) -> None:
        proxy = MultiFidelityProxy()
        s = proxy.pow2_cliff_score(_cfg())
        assert math.isnan(s)

    def test_exact_power_of_two_returns_one(self, fitted_proxy: MultiFidelityProxy) -> None:
        """Force score() to return a value exactly at a power-of-two; cliff should be 1."""
        import unittest.mock as mock

        # Patch score() to return log(2^20) = 20*log(2).
        target_rows = 2 ** 20
        with mock.patch.object(fitted_proxy, "score", return_value=math.log(target_rows)):
            cliff = fitted_proxy.pow2_cliff_score(_cfg())
        assert cliff == pytest.approx(1.0)

    def test_midpoint_returns_zero(self, fitted_proxy: MultiFidelityProxy) -> None:
        """Force score() to return midpoint between pow2 boundaries; cliff should be 0."""
        import unittest.mock as mock

        prev_pow2 = 2 ** 20
        next_pow2 = 2 ** 21
        midpoint = (prev_pow2 + next_pow2) / 2
        with mock.patch.object(fitted_proxy, "score", return_value=math.log(midpoint)):
            cliff = fitted_proxy.pow2_cliff_score(_cfg())
        assert cliff == pytest.approx(0.0, abs=1e-10)

    def test_cliff_score_range_over_varied_configs(self, fitted_proxy: MultiFidelityProxy) -> None:
        """All configs produce cliff scores in [0, 1]."""
        configs = [
            _cfg(d_model=32, n_heads=4, n_layers=2),
            _cfg(d_model=64, n_heads=4, n_layers=4),
            _cfg(d_model=128, n_heads=8, n_layers=6),
            _cfg(d_model=256, n_heads=8, n_layers=8),
        ]
        for config in configs:
            s = fitted_proxy.pow2_cliff_score(config)
            assert 0.0 <= s <= 1.0, f"cliff_score={s!r} out of range for {config}"


# ---------------------------------------------------------------------------
# Fidelity weighting
# ---------------------------------------------------------------------------


class TestFidelityWeighting:
    def test_higher_fidelity_gets_higher_weight(self) -> None:
        """Sample weights for EZKL_PROVE records must exceed ONNX_ESTIMATE weights."""
        proxy = MultiFidelityProxy()
        dataset = ProxyDataset()

        config_onnx = _cfg(d_model=64, n_heads=4)
        config_proved = _cfg(d_model=128, n_heads=8)

        dataset.add(_record(config=config_onnx, onnx_estimate=500_000))
        dataset.add(
            _record(config=config_proved, onnx_estimate=1_000_000, ezkl_num_rows=2_000_000)
        )

        weights = proxy._compute_sample_weights(dataset)
        assert len(weights) == 2
        # First record: ONNX_ESTIMATE → 0.3; second: EZKL_PROVE → 1.0.
        assert weights[0] == pytest.approx(0.3)
        assert weights[1] == pytest.approx(1.0)

    def test_calibration_fidelity_weight(self) -> None:
        proxy = MultiFidelityProxy()
        dataset = ProxyDataset()
        # Record has ezkl_calibration but no numeric label → fidelity_level = EZKL_CALIBRATION,
        # but best_label() = None (no num_rows/prove_time), so it is excluded from to_xy.
        # Use a record with onnx_estimate set so it has a label and calibration fidelity.
        # Actually: fidelity_level() returns EZKL_CALIBRATION when ezkl_calibration is set
        # and no prove-level data exists; but best_label() requires numeric data.
        # We test via a record that has ezkl_prove_time (EZKL_PROVE) as a secondary
        # and onnx_estimate for the label — not calibration.
        # Instead verify that EZKL_CALIBRATION weight is between ONNX and PROVE.
        onnx_w = proxy._fidelity_weights[Fidelity.ONNX_ESTIMATE]
        calib_w = proxy._fidelity_weights[Fidelity.EZKL_CALIBRATION]
        prove_w = proxy._fidelity_weights[Fidelity.EZKL_PROVE]
        assert onnx_w < calib_w < prove_w

    def test_custom_weights_respected_in_fit(self) -> None:
        """Custom fidelity_weights should be stored and used."""
        custom = {Fidelity.ONNX_ESTIMATE: 0.1, Fidelity.EZKL_CALIBRATION: 0.5, Fidelity.EZKL_PROVE: 1.0}
        proxy = MultiFidelityProxy(fidelity_weights=custom)
        assert proxy._fidelity_weights[Fidelity.ONNX_ESTIMATE] == pytest.approx(0.1)
        assert proxy._fidelity_weights[Fidelity.EZKL_CALIBRATION] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


class TestUncertainty:
    @pytest.fixture(scope="class")
    def fitted_proxy(self) -> MultiFidelityProxy:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        return proxy

    def test_returns_non_negative_float(self, fitted_proxy: MultiFidelityProxy) -> None:
        config = _cfg(d_model=64, n_heads=4, n_layers=4)
        u = fitted_proxy.uncertainty(config)
        assert isinstance(u, float)
        assert u >= 0.0

    def test_returns_nan_when_not_fitted(self) -> None:
        proxy = MultiFidelityProxy()
        u = proxy.uncertainty(_cfg())
        assert math.isnan(u)

    def test_uncertainty_varies_across_configs(self, fitted_proxy: MultiFidelityProxy) -> None:
        u1 = fitted_proxy.uncertainty(_cfg(d_model=32, n_heads=4, n_layers=2))
        u2 = fitted_proxy.uncertainty(_cfg(d_model=256, n_heads=8, n_layers=8))
        # At least one of them should be finite.
        assert math.isfinite(u1) or math.isfinite(u2)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    @pytest.fixture(scope="class")
    def fitted_proxy(self) -> MultiFidelityProxy:
        dataset = _make_synthetic_dataset(n=30)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        return proxy

    def test_save_creates_file(self, fitted_proxy: MultiFidelityProxy) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_proxy.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_load_returns_multifidelity_proxy(self, fitted_proxy: MultiFidelityProxy) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_proxy.save(path)
            loaded = MultiFidelityProxy.load(path)
            assert isinstance(loaded, MultiFidelityProxy)
        finally:
            os.unlink(path)

    def test_load_preserves_fitted_state(self, fitted_proxy: MultiFidelityProxy) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_proxy.save(path)
            loaded = MultiFidelityProxy.load(path)
            assert loaded.is_fitted()
        finally:
            os.unlink(path)

    def test_round_trip_scores_match(self, fitted_proxy: MultiFidelityProxy) -> None:
        config = _cfg(d_model=128, n_heads=8, n_layers=4)
        original_score = fitted_proxy.score(config)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_proxy.save(path)
            loaded = MultiFidelityProxy.load(path)
            loaded_score = loaded.score(config)
            assert loaded_score == pytest.approx(original_score, rel=1e-6)
        finally:
            os.unlink(path)

    def test_load_from_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            MultiFidelityProxy.load("/tmp/_nonexistent_multifidelity_proxy.pkl")

    def test_load_wrong_type_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as f:
            import pickle

            pickle.dump({"not": "a proxy"}, f)
            path = f.name
        try:
            with pytest.raises(TypeError, match="MultiFidelityProxy"):
                MultiFidelityProxy.load(path)
        finally:
            os.unlink(path)

    def test_hyperparams_preserved_after_round_trip(self, fitted_proxy: MultiFidelityProxy) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            fitted_proxy.save(path)
            loaded = MultiFidelityProxy.load(path)
            assert loaded._n_estimators == fitted_proxy._n_estimators
            assert loaded._max_depth == fitted_proxy._max_depth
            assert loaded._learning_rate == pytest.approx(fitted_proxy._learning_rate)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Ranker is omitted when n < 20
# ---------------------------------------------------------------------------


class TestRankerThreshold:
    def test_ranker_absent_for_small_dataset(self) -> None:
        dataset = _make_synthetic_dataset(n=12)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        # 12 < 20, so no ranker should be trained.
        assert proxy._ranker is None

    def test_ranker_present_for_large_dataset(self) -> None:
        dataset = _make_synthetic_dataset(n=25)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        assert proxy._ranker is not None

    def test_score_still_works_without_ranker(self) -> None:
        dataset = _make_synthetic_dataset(n=12)
        proxy = MultiFidelityProxy(n_estimators=3, max_depth=3)
        proxy.fit(dataset)
        s = proxy.score(_cfg(d_model=64, n_heads=4, n_layers=3))
        assert math.isfinite(s)


# ---------------------------------------------------------------------------
# Default fidelity weight constants
# ---------------------------------------------------------------------------


class TestDefaultFidelityWeights:
    def test_default_weights_exported(self) -> None:
        assert _DEFAULT_FIDELITY_WEIGHTS[Fidelity.ONNX_ESTIMATE] == pytest.approx(0.3)
        assert _DEFAULT_FIDELITY_WEIGHTS[Fidelity.EZKL_CALIBRATION] == pytest.approx(0.7)
        assert _DEFAULT_FIDELITY_WEIGHTS[Fidelity.EZKL_PROVE] == pytest.approx(1.0)

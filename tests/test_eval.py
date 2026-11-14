"""Tests for proxy.eval — ProxyEvaluator and ProxyEvalResult.

Slow proxy fits (XGBoost) are replaced with sklearn GBM stubs via
``unittest.mock`` to keep the test suite fast.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any
from unittest import mock

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from proxy.data import ProxyDataset, ProxyRecord
from proxy.eval import (
    ProxyEvalResult,
    ProxyEvaluator,
    _compute_calibration_error,
    _compute_ndcg_at_10,
    _compute_top_k_precision,
)
from proxy.features import ProxyFeatures, TransformerConfig
from proxy.strategies.kernel_decomp import KernelDecompositionProxy
from proxy.strategies.multifidelity import MultiFidelityProxy
from proxy.strategies.zero_cost import ZeroCostProxy


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _cfg(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    seq_len: int = 64,
    ff_mult: float = 4.0,
    activation: str = "relu",
) -> TransformerConfig:
    """Build a :class:`TransformerConfig` with sensible defaults."""
    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        ff_mult=ff_mult,
        activation=activation,
    )


def _make_synthetic_dataset(  # pylint: disable=too-many-locals
    n: int = 50, seed: int = 0
) -> ProxyDataset:
    """Build a synthetic :class:`ProxyDataset` with *n* records.

    Labels are set to ``ezkl_num_rows`` derived from ``total_macs`` with
    multiplicative noise so there is detectable Spearman correlation.
    """
    rng = np.random.default_rng(seed)
    dataset = ProxyDataset()

    d_models = [32, 64, 128, 256]
    n_heads_map = {32: 4, 64: 4, 128: 8, 256: 8}

    for i in range(n):
        d_model = d_models[i % len(d_models)]
        n_heads = n_heads_map[d_model]
        n_layers = int(rng.integers(2, 8))
        seq_len = int(rng.choice([32, 64, 128]))
        ff_mult = float(rng.choice([2.0, 4.0, 8.0]))

        config = _cfg(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            ff_mult=ff_mult,
        )
        features = ProxyFeatures.from_config(config)
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
# Sklearn GBM stub helpers
# ---------------------------------------------------------------------------


class _SklearnGBMStub:
    """sklearn GradientBoostingRegressor wrapper that mimics the XGBRegressor API.

    Used to replace the slow XGBoost regressor in tests so that proxy-fit
    calls complete in milliseconds rather than tens of seconds.
    """

    def __init__(self) -> None:
        self._gbm = GradientBoostingRegressor(
            n_estimators=5, max_depth=2, random_state=0
        )

    def fit(  # pylint: disable=invalid-name,unused-argument
        self, X: np.ndarray, y: np.ndarray, **kwargs: Any
    ) -> "_SklearnGBMStub":
        """Fit the underlying sklearn GBM."""
        self._gbm.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        """Return predictions from the sklearn GBM."""
        return self._gbm.predict(X)


def _make_fast_fit_fn(
    proxy_a: MultiFidelityProxy,
) -> Any:
    """Return a callable that replaces MultiFidelityProxy.fit with a fast GBM stub.

    The stub sets ``proxy_a._regressor`` to a sklearn GBM so that
    ``score()`` routes through the regressor branch rather than the fallback.
    """

    def _fast_fit(dataset: ProxyDataset) -> dict:  # pylint: disable=invalid-name
        features_matrix, labels = dataset.to_xy()
        stub = _SklearnGBMStub()
        stub.fit(features_matrix, labels)
        proxy_a._regressor = stub  # type: ignore[assignment]  # pylint: disable=protected-access
        proxy_a._ranker = None  # pylint: disable=protected-access
        proxy_a._quantile_low = None  # pylint: disable=protected-access
        proxy_a._quantile_high = None  # pylint: disable=protected-access
        proxy_a._fallback_mean = None  # pylint: disable=protected-access
        proxy_a._reg_min = float(labels.min())  # pylint: disable=protected-access
        proxy_a._reg_max = float(labels.max())  # pylint: disable=protected-access
        return {"n_samples": len(labels), "cv_spearman": 0.9, "cv_ndcg": 0.8}

    return _fast_fit


def _patch_proxy_a_fit(proxy_a: MultiFidelityProxy) -> Any:
    """Return a context manager patching ``proxy_a.fit`` with a fast sklearn stub.

    The fast stub replaces the internal XGBoost regressor with a tiny sklearn
    GradientBoostingRegressor so tests complete in milliseconds.
    """
    return mock.patch.object(proxy_a, "fit", side_effect=_make_fast_fit_fn(proxy_a))


# ---------------------------------------------------------------------------
# Unit tests for metric helpers
# ---------------------------------------------------------------------------


class TestTopKPrecision:
    """Unit tests for _compute_top_k_precision."""

    def test_perfect_ranking_gives_one(self) -> None:
        """Perfect predictions recover 100% of the true top-k."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _compute_top_k_precision(y, y.copy(), k=1) == pytest.approx(1.0)

    def test_reversed_ranking_gives_zero_for_k1(self) -> None:
        """Completely reversed ordering misses the true cheapest candidate."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y[::-1].copy()
        result = _compute_top_k_precision(y, y_pred, k=1)
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """One of two true top-2 is recovered gives 0.5 precision."""
        # True top-2: indices 0, 1 (values 1.0, 2.0)
        # Pred top-2: indices 0, 2 (values 1.0, 3.0) → overlap = {0} → precision = 0.5
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 3.0, 2.0, 4.0])
        result = _compute_top_k_precision(y_true, y_pred, k=2)
        assert result == pytest.approx(0.5)

    def test_k_zero_returns_nan(self) -> None:
        """k=0 produces nan rather than a division error."""
        y = np.array([1.0, 2.0])
        assert math.isnan(_compute_top_k_precision(y, y.copy(), k=0))

    def test_k_equals_n(self) -> None:
        """When k equals dataset size both sets are the same → precision 1."""
        y = np.array([3.0, 1.0, 2.0, 5.0])
        y_pred = np.array([2.0, 3.0, 1.0, 4.0])
        result = _compute_top_k_precision(y, y_pred, k=4)
        assert result == pytest.approx(1.0)

    def test_true_top10_percent_correct_identification(self) -> None:
        """Near-perfect ordering correctly identifies the cheapest candidate."""
        rng = np.random.default_rng(42)
        y_true = np.arange(10, dtype=float)
        y_pred = y_true + rng.random(10) * 0.05
        k = max(1, 10 // 10)
        result = _compute_top_k_precision(y_true, y_pred, k=k)
        assert result == pytest.approx(1.0)


class TestNdcgAt10:
    """Unit tests for _compute_ndcg_at_10."""

    def test_perfect_prediction_returns_one(self) -> None:
        """Predictions that perfectly track true costs yield NDCG@10 = 1."""
        y = np.arange(10, dtype=float)
        score = _compute_ndcg_at_10(y, y.copy())
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_returns_value_in_0_1(self) -> None:
        """Random predictions produce NDCG@10 in the valid range."""
        rng = np.random.default_rng(99)
        y_true = rng.random(30)
        y_pred = rng.random(30)
        score = _compute_ndcg_at_10(y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_too_few_samples_returns_nan(self) -> None:
        """A single sample cannot produce a meaningful NDCG."""
        y = np.array([1.0])
        assert math.isnan(_compute_ndcg_at_10(y, y.copy()))

    def test_identical_costs_return_one(self) -> None:
        """When all true costs are equal any ordering is a perfect ranking."""
        y = np.ones(20)
        score = _compute_ndcg_at_10(y, y.copy())
        assert score == pytest.approx(1.0)

    def test_larger_dataset(self) -> None:
        """NDCG@10 is computed without errors on a 100-sample dataset."""
        rng = np.random.default_rng(5)
        y_true = rng.random(100)
        y_pred = rng.random(100)
        score = _compute_ndcg_at_10(y_true, y_pred)
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0


class TestCalibrationError:
    """Unit tests for _compute_calibration_error."""

    def test_near_perfect_calibration_gives_low_error(self) -> None:
        """Near-perfect predictions produce low calibration error."""
        rng = np.random.default_rng(7)
        y_true = rng.random(100)
        y_pred = y_true + rng.random(100) * 0.01
        err = _compute_calibration_error(y_true, y_pred)
        assert err >= 0.0
        assert err < 0.2

    def test_returns_zero_for_few_samples(self) -> None:
        """Fewer than 10 samples produce a calibration error of 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert _compute_calibration_error(y, y.copy()) == pytest.approx(0.0)

    def test_output_is_non_negative(self) -> None:
        """Calibration error is always non-negative."""
        rng = np.random.default_rng(11)
        y_true = rng.random(50)
        y_pred = rng.random(50)
        err = _compute_calibration_error(y_true, y_pred)
        assert err >= 0.0

    def test_constant_prediction_has_bounded_error(self) -> None:
        """Constant predictions collapse all samples into one bin — error is finite."""
        y_true = np.arange(20, dtype=float)
        y_pred = np.ones(20) * 5.0
        err = _compute_calibration_error(y_true, y_pred)
        assert math.isfinite(err)
        assert err >= 0.0


# ---------------------------------------------------------------------------
# ProxyEvaluator construction
# ---------------------------------------------------------------------------


class TestProxyEvaluatorConstruction:
    """Tests for ProxyEvaluator.__init__ and the train/test split."""

    def test_construction_splits_dataset(self) -> None:
        """Total records are conserved across the train/test split."""
        ds = _make_synthetic_dataset(n=50)
        evaluator = ProxyEvaluator(ds, test_fraction=0.2, seed=42)
        n_train = len(evaluator._train_dataset)  # pylint: disable=protected-access
        n_test = len(evaluator._test_dataset)  # pylint: disable=protected-access
        assert n_train + n_test == len(ds)
        assert n_test >= 1

    def test_invalid_test_fraction_zero_raises(self) -> None:
        """test_fraction=0 must raise ValueError."""
        ds = _make_synthetic_dataset(n=20)
        with pytest.raises(ValueError):
            ProxyEvaluator(ds, test_fraction=0.0)

    def test_invalid_test_fraction_one_raises(self) -> None:
        """test_fraction=1 must raise ValueError."""
        ds = _make_synthetic_dataset(n=20)
        with pytest.raises(ValueError):
            ProxyEvaluator(ds, test_fraction=1.0)

    def test_test_configs_length_matches_test_dataset(self) -> None:
        """Number of test configs equals the test dataset size."""
        ds = _make_synthetic_dataset(n=50)
        evaluator = ProxyEvaluator(ds, test_fraction=0.2, seed=42)
        n_test = len(evaluator._test_dataset)  # pylint: disable=protected-access
        # pylint: disable=protected-access
        assert len(evaluator._test_configs) == n_test

    def test_different_seeds_give_different_splits(self) -> None:
        """Different seeds produce statistically different test sets."""
        ds = _make_synthetic_dataset(n=50)
        ev1 = ProxyEvaluator(ds, test_fraction=0.2, seed=0)
        ev2 = ProxyEvaluator(ds, test_fraction=0.2, seed=99)
        # pylint: disable=protected-access
        configs1 = {id(c) for c in ev1._test_configs}
        configs2 = {id(c) for c in ev2._test_configs}
        assert configs1 != configs2

    def test_train_dataset_nonempty(self) -> None:
        """Training split contains at least one record."""
        ds = _make_synthetic_dataset(n=20)
        evaluator = ProxyEvaluator(ds, test_fraction=0.2, seed=42)
        assert len(evaluator._train_dataset) > 0  # pylint: disable=protected-access

    def test_test_dataset_nonempty(self) -> None:
        """Test split contains at least one record."""
        ds = _make_synthetic_dataset(n=20)
        evaluator = ProxyEvaluator(ds, test_fraction=0.2, seed=42)
        assert len(evaluator._test_dataset) > 0  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# evaluate_all with synthetic dataset and sklearn GBM stubs
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    """Test ProxyEvaluator.evaluate_all using fast sklearn GBM stubs for proxy A."""

    @pytest.fixture(scope="class")
    def dataset(self) -> ProxyDataset:
        """50-record synthetic dataset."""
        return _make_synthetic_dataset(n=50, seed=0)

    @pytest.fixture(scope="class")
    def evaluator(self, dataset: ProxyDataset) -> ProxyEvaluator:
        """ProxyEvaluator using 80/20 train/test split."""
        return ProxyEvaluator(dataset, test_fraction=0.2, seed=42)

    def _run_evaluation(self, evaluator: ProxyEvaluator) -> list[ProxyEvalResult]:
        """Run evaluate_all with fast sklearn GBM stub for proxy A + ridge proxy B."""
        proxy_a = MultiFidelityProxy(n_estimators=2, max_depth=2)
        proxy_b = ZeroCostProxy(model_type="ridge")
        with _patch_proxy_a_fit(proxy_a):
            return evaluator.evaluate_all(proxy_a, proxy_b)

    def test_returns_list_of_results(self, evaluator: ProxyEvaluator) -> None:
        """evaluate_all returns a list."""
        results = self._run_evaluation(evaluator)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_result_types_are_proxy_eval_result(
        self, evaluator: ProxyEvaluator
    ) -> None:
        """All returned items are ProxyEvalResult instances."""
        results = self._run_evaluation(evaluator)
        for r in results:
            assert isinstance(r, ProxyEvalResult)

    def test_proxy_names_are_correct(self, evaluator: ProxyEvaluator) -> None:
        """Returned results are labelled MultiFidelity and ZeroCost."""
        results = self._run_evaluation(evaluator)
        names = [r.proxy_name for r in results]
        assert "MultiFidelity" in names
        assert "ZeroCost" in names

    def test_n_train_is_positive(self, evaluator: ProxyEvaluator) -> None:
        """Training set is non-empty for every result."""
        results = self._run_evaluation(evaluator)
        for r in results:
            assert r.n_train > 0

    def test_n_test_is_positive(self, evaluator: ProxyEvaluator) -> None:
        """Test set is non-empty for every result."""
        results = self._run_evaluation(evaluator)
        for r in results:
            assert r.n_test > 0

    def test_spearman_r_in_range(self, evaluator: ProxyEvaluator) -> None:
        """Spearman r is in [-1, 1] when it is not nan."""
        results = self._run_evaluation(evaluator)
        for r in results:
            if not math.isnan(r.spearman_r):
                assert -1.0 <= r.spearman_r <= 1.0

    def test_top10_precision_in_range(self, evaluator: ProxyEvaluator) -> None:
        """P@10% is in [0, 1] when it is not nan."""
        results = self._run_evaluation(evaluator)
        for r in results:
            if not math.isnan(r.top10_precision):
                assert 0.0 <= r.top10_precision <= 1.0

    def test_top1_precision_in_range(self, evaluator: ProxyEvaluator) -> None:
        """P@1% is in [0, 1] when it is not nan."""
        results = self._run_evaluation(evaluator)
        for r in results:
            if not math.isnan(r.top1_precision):
                assert 0.0 <= r.top1_precision <= 1.0

    def test_ndcg_at_10_in_range(self, evaluator: ProxyEvaluator) -> None:
        """NDCG@10 is in [0, 1] when it is not nan."""
        results = self._run_evaluation(evaluator)
        for r in results:
            if not math.isnan(r.ndcg_at_10):
                assert 0.0 <= r.ndcg_at_10 <= 1.0

    def test_mae_log_is_non_negative(self, evaluator: ProxyEvaluator) -> None:
        """MAE(log) is non-negative."""
        results = self._run_evaluation(evaluator)
        for r in results:
            if not math.isnan(r.mae_log):
                assert r.mae_log >= 0.0

    def test_training_time_is_non_negative(self, evaluator: ProxyEvaluator) -> None:
        """Training time is non-negative."""
        results = self._run_evaluation(evaluator)
        for r in results:
            assert r.training_time_s >= 0.0

    def test_inference_time_is_non_negative(self, evaluator: ProxyEvaluator) -> None:
        """Inference time per candidate is non-negative."""
        results = self._run_evaluation(evaluator)
        for r in results:
            assert r.inference_time_ms >= 0.0

    def test_with_proxy_c_returns_three_results(
        self, evaluator: ProxyEvaluator
    ) -> None:
        """When proxy_c is provided, three results are returned."""
        proxy_a = MultiFidelityProxy(n_estimators=2, max_depth=2)
        proxy_b = ZeroCostProxy(model_type="ridge")
        proxy_c = KernelDecompositionProxy(cost_table={}, fallback_to_onnx=True)
        with _patch_proxy_a_fit(proxy_a):
            with mock.patch.object(proxy_c, "score", return_value=10.0):
                results = evaluator.evaluate_all(proxy_a, proxy_b, proxy_c)
        assert len(results) == 3
        names = [r.proxy_name for r in results]
        assert "KernelDecomp" in names

    def test_proxy_c_none_returns_two_results(self, evaluator: ProxyEvaluator) -> None:
        """When proxy_c is None, exactly two results are returned."""
        results = self._run_evaluation(evaluator)
        assert len(results) == 2

    def test_n_train_n_test_sum_equals_dataset(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """n_train + n_test equals the total dataset size for every result."""
        results = self._run_evaluation(evaluator)
        for r in results:
            assert r.n_train + r.n_test == len(dataset)


# ---------------------------------------------------------------------------
# to_markdown_table
# ---------------------------------------------------------------------------


class TestToMarkdownTable:
    """Tests for ProxyEvaluator.to_markdown_table."""

    def _make_result(self, name: str) -> ProxyEvalResult:
        """Build a synthetic ProxyEvalResult for table formatting tests."""
        return ProxyEvalResult(
            proxy_name=name,
            n_train=40,
            n_test=10,
            spearman_r=0.85,
            spearman_p=0.001,
            top10_precision=0.70,
            top1_precision=0.50,
            ndcg_at_10=0.80,
            mae_log=0.30,
            calibration_error=0.05,
            training_time_s=1.2,
            inference_time_ms=0.5,
        )

    @pytest.fixture
    def evaluator(self) -> ProxyEvaluator:
        """Evaluator backed by a small synthetic dataset."""
        ds = _make_synthetic_dataset(n=30)
        return ProxyEvaluator(ds, test_fraction=0.2, seed=0)

    def test_returns_string(self, evaluator: ProxyEvaluator) -> None:
        """Output is a non-empty string."""
        results = [self._make_result("MultiFidelity"), self._make_result("ZeroCost")]
        table = evaluator.to_markdown_table(results)
        assert isinstance(table, str)

    def test_has_header_row(self, evaluator: ProxyEvaluator) -> None:
        """Header row contains all required column names."""
        results = [self._make_result("MultiFidelity")]
        table = evaluator.to_markdown_table(results)
        lines = table.splitlines()
        assert len(lines) >= 3, "Expected header + separator + at least one data row"
        header = lines[0]
        for col in (
            "Proxy",
            "Spearman",
            "P@10%",
            "P@1%",
            "NDCG@10",
            "MAE",
            "Train time",
        ):
            assert col in header, f"Column {col!r} missing from header"

    def test_separator_line_present(self, evaluator: ProxyEvaluator) -> None:
        """Second line is a Markdown table separator."""
        results = [self._make_result("MultiFidelity")]
        table = evaluator.to_markdown_table(results)
        lines = table.splitlines()
        sep = lines[1]
        assert "---" in sep
        assert "|" in sep

    def test_correct_number_of_data_rows(self, evaluator: ProxyEvaluator) -> None:
        """One data row is output per result."""
        results = [
            self._make_result("MultiFidelity"),
            self._make_result("ZeroCost"),
            self._make_result("KernelDecomp"),
        ]
        table = evaluator.to_markdown_table(results)
        lines = table.splitlines()
        data_rows = [
            ln
            for ln in lines
            if ln.startswith("|") and "---" not in ln and "Proxy" not in ln
        ]
        assert len(data_rows) == 3

    def test_proxy_label_mapping(self, evaluator: ProxyEvaluator) -> None:
        """Internal proxy names are mapped to paper labels (A/B/C)."""
        results = [
            self._make_result("MultiFidelity"),
            self._make_result("ZeroCost"),
            self._make_result("KernelDecomp"),
        ]
        table = evaluator.to_markdown_table(results)
        assert "MultiFidelity (A)" in table
        assert "ZeroCost (B)" in table
        assert "KernelDecomp (C)" in table

    def test_numeric_values_present_in_table(self, evaluator: ProxyEvaluator) -> None:
        """Spearman r is rendered in the table."""
        results = [self._make_result("MultiFidelity")]
        table = evaluator.to_markdown_table(results)
        assert "0.85" in table

    def test_empty_results_has_header_and_separator(
        self, evaluator: ProxyEvaluator
    ) -> None:
        """Empty results list still produces header + separator."""
        table = evaluator.to_markdown_table([])
        lines = table.splitlines()
        assert len(lines) >= 2
        assert "Proxy" in lines[0]
        assert "---" in lines[1]

    def test_all_rows_are_pipe_delimited(self, evaluator: ProxyEvaluator) -> None:
        """Every line contains at least one pipe character."""
        results = [self._make_result("MultiFidelity"), self._make_result("ZeroCost")]
        table = evaluator.to_markdown_table(results)
        for line in table.splitlines():
            assert "|" in line, f"Line missing pipe delimiter: {line!r}"


# ---------------------------------------------------------------------------
# learning_curve
# ---------------------------------------------------------------------------


class TestLearningCurve:
    """Tests for ProxyEvaluator.learning_curve."""

    @pytest.fixture(scope="class")
    def dataset(self) -> ProxyDataset:
        """60-record synthetic dataset."""
        return _make_synthetic_dataset(n=60, seed=1)

    @pytest.fixture(scope="class")
    def evaluator(self, dataset: ProxyDataset) -> ProxyEvaluator:
        """ProxyEvaluator with 80/20 split."""
        return ProxyEvaluator(dataset, test_fraction=0.2, seed=42)

    def _make_fast_cls_fit(self) -> Any:
        """Return a class-level patch function that uses sklearn GBM stubs."""

        def _fast_cls_fit(  # pylint: disable=invalid-name
            self_proxy: MultiFidelityProxy, ds: ProxyDataset
        ) -> dict:
            features_matrix, labels = ds.to_xy()
            stub = _SklearnGBMStub()
            stub.fit(features_matrix, labels)
            self_proxy._regressor = stub  # type: ignore[assignment]  # pylint: disable=protected-access
            self_proxy._ranker = None  # pylint: disable=protected-access
            self_proxy._quantile_low = None  # pylint: disable=protected-access
            self_proxy._quantile_high = None  # pylint: disable=protected-access
            self_proxy._fallback_mean = None  # pylint: disable=protected-access
            self_proxy._reg_min = (  # pylint: disable=protected-access
                float(labels.min()) if len(labels) > 0 else 0.0
            )
            self_proxy._reg_max = (  # pylint: disable=protected-access
                float(labels.max()) if len(labels) > 0 else 1.0
            )
            return {"n_samples": len(labels), "cv_spearman": 0.9, "cv_ndcg": 0.8}

        return _fast_cls_fit

    def _run_lc(
        self,
        evaluator: ProxyEvaluator,
        dataset: ProxyDataset,
        sizes: list[int],
        output_path: str | None = None,
    ) -> dict[int, float]:
        """Run learning_curve with class-level fit mock."""
        proxy_a = MultiFidelityProxy(n_estimators=2, max_depth=2)
        with mock.patch.object(MultiFidelityProxy, "fit", self._make_fast_cls_fit()):
            return evaluator.learning_curve(
                proxy_a, dataset, train_sizes=sizes, output_path=output_path
            )

    def test_returns_dict(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """learning_curve returns a dict."""
        curve = self._run_lc(evaluator, dataset, sizes=[5, 10])
        assert isinstance(curve, dict)

    def test_keys_match_train_sizes(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """Dict keys equal the requested training sizes."""
        sizes = [5, 10, 20]
        curve = self._run_lc(evaluator, dataset, sizes=sizes)
        assert set(curve.keys()) == set(sizes)

    def test_values_are_floats(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """All values in the curve dict are floats."""
        curve = self._run_lc(evaluator, dataset, sizes=[5, 10])
        for val in curve.values():
            assert isinstance(val, float)

    def test_spearman_values_in_valid_range(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """Non-nan Spearman values are in [-1, 1]."""
        curve = self._run_lc(evaluator, dataset, sizes=[5, 10])
        for val in curve.values():
            if not math.isnan(val):
                assert -1.0 <= val <= 1.0

    def test_default_train_sizes_used_when_none(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """When train_sizes is None default sizes are used."""
        proxy_a = MultiFidelityProxy(n_estimators=2, max_depth=2)
        with mock.patch.object(MultiFidelityProxy, "fit", self._make_fast_cls_fit()):
            curve = evaluator.learning_curve(proxy_a, dataset, train_sizes=None)
        assert len(curve) > 0

    def test_output_path_creates_png(
        self,
        evaluator: ProxyEvaluator,
        dataset: ProxyDataset,
        tmp_path: Any,
    ) -> None:
        """Providing output_path saves a PNG file."""
        out = str(tmp_path / "lc.png")
        self._run_lc(evaluator, dataset, sizes=[5, 10], output_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_no_output_path_does_not_raise(
        self, evaluator: ProxyEvaluator, dataset: ProxyDataset
    ) -> None:
        """Omitting output_path does not raise."""
        curve = self._run_lc(evaluator, dataset, sizes=[5], output_path=None)
        assert isinstance(curve, dict)


# ---------------------------------------------------------------------------
# plot_calibration
# ---------------------------------------------------------------------------


class TestPlotCalibration:
    """Tests for ProxyEvaluator.plot_calibration."""

    @pytest.fixture(scope="class")
    def evaluator(self) -> ProxyEvaluator:
        """Evaluator backed by a 30-record synthetic dataset."""
        ds = _make_synthetic_dataset(n=30)
        return ProxyEvaluator(ds, test_fraction=0.2, seed=0)

    def test_saves_png_file(self, evaluator: ProxyEvaluator, tmp_path: Any) -> None:
        """plot_calibration saves a non-empty PNG."""
        results = [
            ProxyEvalResult(
                proxy_name="MultiFidelity",
                n_train=24,
                n_test=6,
                spearman_r=0.8,
                spearman_p=0.05,
                top10_precision=0.6,
                top1_precision=0.5,
                ndcg_at_10=0.75,
                mae_log=0.3,
                calibration_error=0.05,
                training_time_s=1.0,
                inference_time_ms=0.1,
            )
        ]
        rng = np.random.default_rng(0)
        n = 6
        true_labels = rng.random(n).tolist()
        proxy_scores = {"MultiFidelity": rng.random(n).tolist()}
        out = str(tmp_path / "calib.png")
        evaluator.plot_calibration(results, proxy_scores, true_labels, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_mismatched_score_length_does_not_raise(
        self, evaluator: ProxyEvaluator, tmp_path: Any
    ) -> None:
        """Mismatched score list length is silently skipped without raising."""
        results = [
            ProxyEvalResult(
                proxy_name="ZeroCost",
                n_train=24,
                n_test=6,
                spearman_r=0.7,
                spearman_p=0.1,
                top10_precision=0.5,
                top1_precision=0.4,
                ndcg_at_10=0.65,
                mae_log=0.4,
                calibration_error=0.08,
                training_time_s=0.1,
                inference_time_ms=0.05,
            )
        ]
        true_labels = [0.1, 0.2, 0.3]
        proxy_scores = {"ZeroCost": [0.1, 0.2]}  # wrong length
        out = str(tmp_path / "calib_mismatch.png")
        evaluator.plot_calibration(results, proxy_scores, true_labels, out)

    def test_empty_true_labels_does_not_raise(
        self, evaluator: ProxyEvaluator, tmp_path: Any
    ) -> None:
        """Empty true_labels list does not raise."""
        out = str(tmp_path / "calib_empty.png")
        evaluator.plot_calibration([], {}, [], out)


# ---------------------------------------------------------------------------
# ProxyEvalResult dataclass
# ---------------------------------------------------------------------------


class TestProxyEvalResultDataclass:
    """Tests for the ProxyEvalResult dataclass."""

    def test_all_fields_accessible(self) -> None:
        """All twelve fields are readable from a constructed ProxyEvalResult."""
        r = ProxyEvalResult(
            proxy_name="TestProxy",
            n_train=10,
            n_test=5,
            spearman_r=0.9,
            spearman_p=0.001,
            top10_precision=0.8,
            top1_precision=0.6,
            ndcg_at_10=0.85,
            mae_log=0.1,
            calibration_error=0.05,
            training_time_s=0.5,
            inference_time_ms=0.2,
        )
        assert r.proxy_name == "TestProxy"
        assert r.n_train == 10
        assert r.n_test == 5
        assert r.spearman_r == pytest.approx(0.9)
        assert r.spearman_p == pytest.approx(0.001)
        assert r.top10_precision == pytest.approx(0.8)
        assert r.top1_precision == pytest.approx(0.6)
        assert r.ndcg_at_10 == pytest.approx(0.85)
        assert r.mae_log == pytest.approx(0.1)
        assert r.calibration_error == pytest.approx(0.05)
        assert r.training_time_s == pytest.approx(0.5)
        assert r.inference_time_ms == pytest.approx(0.2)

    def test_is_dataclass(self) -> None:
        """ProxyEvalResult is a Python dataclass."""
        assert dataclasses.is_dataclass(ProxyEvalResult)

    def test_repr_contains_proxy_name(self) -> None:
        """repr includes the proxy_name field value."""
        r = ProxyEvalResult(
            proxy_name="MultiFidelity",
            n_train=0,
            n_test=0,
            spearman_r=0.0,
            spearman_p=0.0,
            top10_precision=0.0,
            top1_precision=0.0,
            ndcg_at_10=0.0,
            mae_log=0.0,
            calibration_error=0.0,
            training_time_s=0.0,
            inference_time_ms=0.0,
        )
        assert "MultiFidelity" in repr(r)

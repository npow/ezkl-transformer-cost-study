"""Approach A: Multi-fidelity GBM proxy for ZK proving cost.

Uses a dual-objective ensemble:
  1. XGBRegressor (reg:squarederror) on log(prove_cost) — stable with few samples.
  2. XGBRanker (rank:ndcg) for ranking accuracy — NAS-optimal.

The ONNX constraint estimate is treated as a *feature*, not a label.
Ground truth labels come from real ZK proving measurements (ezkl_num_rows /
ezkl_prove_time), with ONNX estimates as a fallback.

Key non-additive effect: EZKL snaps circuits to power-of-two row counts, so
architectures near a boundary are expensive if they grow slightly. This
``pow2_cliff_score`` penalty is incorporated into uncertainty estimates.
"""

from __future__ import annotations

import logging
import math
import pickle
from typing import Any, Optional, Union

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold, LeaveOneOut
from xgboost import XGBRanker, XGBRegressor

from proxy.data import Fidelity, ProxyDataset
from proxy.features import ProxyFeatures, TransformerConfig, feature_names

logger = logging.getLogger(__name__)

# Feature index of the ONNX constraint estimate column in the feature vector.
# Derived once at module load to avoid repeated list scans.
_ONNX_CONSTRAINT_FEAT_IDX: int = feature_names().index("onnx_constraint_estimate")

# ---------------------------------------------------------------------------
# Default fidelity weights
# ---------------------------------------------------------------------------

_DEFAULT_FIDELITY_WEIGHTS: dict[Fidelity, float] = {
    Fidelity.ONNX_ESTIMATE: 0.3,
    Fidelity.EZKL_CALIBRATION: 0.7,
    Fidelity.EZKL_PROVE: 1.0,
}

# Minimum samples required to train GBM; below this threshold we fall back to
# the ONNX estimate directly.
_MIN_SAMPLES_FOR_GBM: int = 5

# Minimum samples for ranker to be worthwhile.
_MIN_SAMPLES_FOR_RANKER: int = 20

# Ensemble weights: regressor dominates; ranker refines ordering.
_ENSEMBLE_REGRESSOR_WEIGHT: float = 0.7
_ENSEMBLE_RANKER_WEIGHT: float = 0.3


# ---------------------------------------------------------------------------
# MultiFidelityProxy
# ---------------------------------------------------------------------------


class MultiFidelityProxy:
    """Approach A: GBM ranker on full feature vector with real proving cost labels.

    Key design:
    - ONNX proxy breakdown is a FEATURE, not the label.
    - Ground truth = log(ezkl_num_rows) or log(ezkl_prove_time).
    - Dual objective: regression (stable with few samples) + ranking (NAS-optimal).
    - Multi-fidelity: uses all available labels, weighted by fidelity.

    Implements ZKCostProxy protocol.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        fidelity_weights: Optional[dict[Fidelity, float]] = None,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._fidelity_weights: dict[Fidelity, float] = (
            fidelity_weights if fidelity_weights is not None else dict(_DEFAULT_FIDELITY_WEIGHTS)
        )

        # Trained model slots — None until fit() is called.
        self._regressor: Optional[XGBRegressor] = None
        self._ranker: Optional[XGBRanker] = None
        self._quantile_low: Optional[XGBRegressor] = None
        self._quantile_high: Optional[XGBRegressor] = None

        # Cached ranker output range for min-max normalisation at predict time.
        self._ranker_min: float = 0.0
        self._ranker_max: float = 1.0

        # Cached regressor output range on training data (for ranker rescaling).
        self._reg_min: float = 0.0
        self._reg_max: float = 1.0

        # Fallback: when n_samples < _MIN_SAMPLES_FOR_GBM, store mean log label.
        self._fallback_mean: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: ProxyDataset) -> dict:
        """Train on *dataset*.

        Fits the primary regressor, optional ranker, and quantile models on all
        available labelled records, weighted by fidelity.  Returns a lightweight
        training summary using in-sample Spearman correlation (not cross-validated)
        to avoid the extra cost of ``cv_metrics()``.  Call :meth:`cv_metrics`
        explicitly after training when proper held-out estimates are needed.

        Returns
        -------
        dict
            ``{'n_samples': int, 'cv_spearman': float, 'cv_ndcg': float}``

            ``cv_spearman`` and ``cv_ndcg`` are computed on the training data
            (in-sample) when ``n_samples >= _MIN_SAMPLES_FOR_GBM``; ``nan``
            otherwise.  For held-out cross-validated estimates call
            :meth:`cv_metrics` separately.
        """
        X, y, weights = self._extract_weighted(dataset)
        n_samples = len(y)
        logger.info("MultiFidelityProxy.fit: n_samples=%d", n_samples)

        if n_samples < _MIN_SAMPLES_FOR_GBM:
            self._fallback_mean = float(np.mean(y)) if n_samples > 0 else 0.0
            logger.warning(
                "Too few samples (%d < %d) for GBM; using mean fallback %.4f",
                n_samples,
                _MIN_SAMPLES_FOR_GBM,
                self._fallback_mean,
            )
            return {
                "n_samples": n_samples,
                "cv_spearman": float("nan"),
                "cv_ndcg": float("nan"),
            }

        self._fallback_mean = None
        self._fit_regressor(X, y, weights)
        self._fit_quantile_models(X, y, weights)

        if n_samples >= _MIN_SAMPLES_FOR_RANKER:
            self._fit_ranker(X, y, weights)

        # Compute lightweight in-sample metrics (no held-out CV).
        y_pred = self._regressor.predict(X)
        spearman_r, _ = spearmanr(y, y_pred)
        ndcg_val = self._compute_ndcg_at_k(y, y_pred, k=max(1, n_samples // 10))
        return {
            "n_samples": n_samples,
            "cv_spearman": float(spearman_r),
            "cv_ndcg": float(ndcg_val) if not math.isnan(ndcg_val) else float("nan"),
        }

    def score(self, config: TransformerConfig) -> float:
        """Predict log(ZK cost).  Lower = cheaper to prove."""
        if not self.is_fitted():
            raise RuntimeError("MultiFidelityProxy is not fitted. Call fit() first.")

        if self._fallback_mean is not None:
            # ONNX estimate available via features — prefer it over global mean.
            features = ProxyFeatures.from_config(config)
            vec = features.to_vector()
            onnx_val = vec[_ONNX_CONSTRAINT_FEAT_IDX]
            if onnx_val > 0:
                return float(np.log(onnx_val))
            return self._fallback_mean

        features = ProxyFeatures.from_config(config)
        x = features.to_vector().reshape(1, -1)
        reg_score = float(self._regressor.predict(x)[0])

        if self._ranker is None:
            return reg_score

        ranker_raw = float(self._ranker.predict(x)[0])
        ranker_norm = self._normalise_ranker(ranker_raw)
        return _ENSEMBLE_REGRESSOR_WEIGHT * reg_score + _ENSEMBLE_RANKER_WEIGHT * ranker_norm

    def uncertainty(self, config: TransformerConfig) -> float:
        """Uncertainty estimate as (p90 - p10) + power-of-two cliff contribution.

        Returns
        -------
        float
            Non-negative uncertainty width.
        """
        if not self.is_fitted() or self._quantile_low is None or self._fallback_mean is not None:
            return float("nan")

        features = ProxyFeatures.from_config(config)
        x = features.to_vector().reshape(1, -1)
        p10 = float(self._quantile_low.predict(x)[0])
        p90 = float(self._quantile_high.predict(x)[0])
        interval_width = max(0.0, p90 - p10)

        cliff = self.pow2_cliff_score(config)
        if math.isnan(cliff):
            return interval_width

        # Add a penalty proportional to cliff proximity (max +1 natural log unit).
        return interval_width + cliff

    def pow2_cliff_score(self, config: TransformerConfig) -> float:
        """Estimate proximity to a power-of-two circuit row boundary.

        EZKL snaps circuits to power-of-two row counts. An arch near a boundary
        (e.g., predicted 3.9 M rows when the snap-point is 4 M) would actually
        cost 8 M rows if it grows slightly. High score = near boundary.

        Returns
        -------
        float
            Value in [0, 1]. 0 = at the midpoint between two pow2 boundaries
            (safest). 1 = exactly at a power-of-two boundary (highest risk).
            Returns ``nan`` when the proxy is not fitted.
        """
        if not self.is_fitted():
            return float("nan")

        log_pred = self.score(config)
        rows = math.exp(log_pred)

        if rows <= 0:
            return float("nan")

        log2_rows = math.log2(rows)
        floor_exp = math.floor(log2_rows)
        ceil_exp = math.ceil(log2_rows)

        if floor_exp == ceil_exp:
            # Exactly a power of two — maximum cliff.
            return 1.0

        prev_pow2 = 2.0 ** floor_exp
        next_pow2 = 2.0 ** ceil_exp
        span = next_pow2 - prev_pow2

        if span == 0:
            return 1.0

        # Fractional position between prev and next pow2 in [0, 1].
        frac = (rows - prev_pow2) / span

        # Transform so that 0.5 (midpoint, safest) → 0 and 0 or 1 (boundary) → 1.
        # Uses 1 - 2*min(frac, 1-frac): score is 0 at the midpoint and 1 at either
        # power-of-two boundary (i.e. high score = near a pow2 snap-point → high risk).
        return 1.0 - 2.0 * min(frac, 1.0 - frac)

    def cv_metrics(self, dataset: ProxyDataset) -> dict:
        """Compute cross-validation metrics on *dataset*.

        Returns
        -------
        dict
            ``{'spearman_r': float, 'ndcg@10': float, 'top10_precision': float}``
            Values are ``nan`` when there are insufficient samples.
        """
        X, y, weights = self._extract_weighted(dataset)
        n_samples = len(y)

        nan_result = {
            "spearman_r": float("nan"),
            "ndcg@10": float("nan"),
            "top10_precision": float("nan"),
        }

        if n_samples < 3:
            return nan_result

        # Choose CV strategy based on dataset size.
        cv: Union[KFold, LeaveOneOut]
        if n_samples >= 20:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        elif n_samples >= 10:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv = LeaveOneOut()

        y_pred_all = np.empty_like(y)
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            w_train = weights[train_idx]

            reg = self._make_regressor()
            reg.fit(X_train, y_train, sample_weight=w_train)
            y_pred_all[test_idx] = reg.predict(X_test)

        spearman_r, _ = spearmanr(y, y_pred_all)
        ndcg_at_10 = self._compute_ndcg_at_k(y, y_pred_all, k=max(1, n_samples // 10))
        top10_prec = self._compute_top10_precision(y, y_pred_all)

        return {
            "spearman_r": float(spearman_r),
            "ndcg@10": float(ndcg_at_10),
            "top10_precision": float(top10_prec),
        }

    def is_fitted(self) -> bool:
        """Return True when the proxy has been trained (or initialised with fallback)."""
        return self._regressor is not None or self._fallback_mean is not None

    def save(self, path: str) -> None:
        """Persist the proxy to *path* using pickle.

        Parameters
        ----------
        path:
            Filesystem path for the output file.
        """
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("MultiFidelityProxy saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MultiFidelityProxy":
        """Load a previously saved proxy from *path*.

        Parameters
        ----------
        path:
            Filesystem path to a pickle file written by :meth:`save`.

        Returns
        -------
        MultiFidelityProxy
            The deserialised proxy.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        TypeError
            If the file does not contain a :class:`MultiFidelityProxy`.
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected MultiFidelityProxy, got {type(obj).__name__!r}"
            )
        logger.info("MultiFidelityProxy loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers: model construction
    # ------------------------------------------------------------------

    def _make_regressor(self) -> XGBRegressor:
        """Return an un-fitted XGBRegressor with canonical hyperparameters.

        ``nthread=1`` pins XGBoost to a single CPU thread.  Without this,
        XGBoost auto-detects the number of CPUs and spawns many threads per
        tree; on machines with high CPU counts the threading overhead dwarfs
        the actual computation for small datasets (e.g. NAS bootstrap phases).
        """
        return XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            objective="reg:squarederror",
            verbosity=0,
            random_state=42,
            nthread=1,
        )

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        self._regressor = self._make_regressor()
        self._regressor.fit(X, y, sample_weight=weights)
        train_preds = self._regressor.predict(X)
        self._reg_min = float(train_preds.min())
        self._reg_max = float(train_preds.max())
        logger.debug("Primary regressor fitted on %d samples.", len(y))

    @staticmethod
    def _log_cost_to_relevance(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Convert continuous log-cost labels to non-negative integer relevance.

        ``rank:ndcg`` in XGBoost requires non-negative integer labels.  Lower
        cost = more relevant, so we invert the ordering when binning: the
        cheapest architecture gets relevance ``n_bins - 1`` (highest), the
        most expensive gets 0.

        Parameters
        ----------
        y:
            1-D array of log-cost labels.
        n_bins:
            Number of relevance levels (default 5).

        Returns
        -------
        np.ndarray
            Integer relevance scores in ``[0, n_bins - 1]``, same length as *y*.
        """
        n = len(y)
        if n == 0:
            return np.empty(0, dtype=np.int32)

        # Rank from cheapest (rank 0) to most expensive (rank n-1).
        order = np.argsort(y)  # ascending cost
        relevance = np.empty(n, dtype=np.int32)
        # Divide the sorted positions into n_bins equal buckets.
        for i, idx in enumerate(order):
            bucket = int(i * n_bins / n)
            # Invert: cheapest bucket → highest relevance.
            relevance[idx] = n_bins - 1 - bucket
        return relevance

    def _fit_ranker(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Fit XGBRanker treating all samples as one query group.

        XGBoost ``rank:ndcg`` requires non-negative integer relevance labels.
        We convert continuous log-cost labels by bucketing into relevance tiers
        (lower cost = higher relevance).

        In XGBoost >= 2.0, ``sample_weight`` for ranking tasks is per-group,
        not per-sample.  Since we have a single query group, we pass the mean
        weight as a scalar array of length 1.
        """
        n = len(y)
        relevance = self._log_cost_to_relevance(y)
        ranker = XGBRanker(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            objective="rank:ndcg",
            verbosity=0,
            random_state=42,
            nthread=1,
        )
        group_weight = np.array([float(weights.mean())], dtype=np.float64)
        ranker.fit(X, relevance, group=[n], sample_weight=group_weight)
        self._ranker = ranker

        # Cache min/max of ranker scores on training data for normalisation.
        train_scores = self._ranker.predict(X)
        self._ranker_min = float(train_scores.min())
        self._ranker_max = float(train_scores.max())
        logger.debug(
            "Ranker fitted on %d samples; score range [%.4f, %.4f].",
            n, self._ranker_min, self._ranker_max,
        )

    def _fit_quantile_models(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        """Fit low (p10) and high (p90) quantile regressors for uncertainty."""
        common_kwargs: dict[str, Any] = dict(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            objective="reg:quantileerror",
            verbosity=0,
            random_state=42,
            nthread=1,
        )
        self._quantile_low = XGBRegressor(quantile_alpha=0.1, **common_kwargs)
        self._quantile_high = XGBRegressor(quantile_alpha=0.9, **common_kwargs)
        self._quantile_low.fit(X, y, sample_weight=weights)
        self._quantile_high.fit(X, y, sample_weight=weights)
        logger.debug("Quantile models (p10/p90) fitted on %d samples.", len(y))

    # ------------------------------------------------------------------
    # Private helpers: data extraction
    # ------------------------------------------------------------------

    def _extract_weighted(
        self, dataset: ProxyDataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, sample_weights) from *dataset*."""
        X, y = dataset.to_xy(min_fidelity=Fidelity.ONNX_ESTIMATE)
        if len(y) == 0:
            return X, y, np.empty(0, dtype=np.float64)

        weights = self._compute_sample_weights(dataset)
        return X, y, weights

    def _compute_sample_weights(self, dataset: ProxyDataset) -> np.ndarray:
        """Return per-sample weights aligned with ``dataset.to_xy()`` row order."""
        weights: list[float] = []
        for record in dataset._records:  # pylint: disable=protected-access
            label = record.best_label()
            if label is None:
                continue
            fid = record.fidelity_level()
            if fid is None:
                continue
            w = self._fidelity_weights.get(fid, 1.0)
            weights.append(w)
        return np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Private helpers: metrics
    # ------------------------------------------------------------------

    def _normalise_ranker(self, raw_score: float) -> float:
        """Min-max normalise a raw ranker score to the regressor output scale.

        Maps the ranker's training-set output range to the regressor's
        training-set output range so the ensemble blending is meaningful.
        """
        ranker_span = self._ranker_max - self._ranker_min
        reg_span = self._reg_max - self._reg_min
        if ranker_span == 0.0:
            return (self._reg_min + self._reg_max) / 2.0
        # Normalise ranker to [0, 1] then scale to regressor range.
        t = (raw_score - self._ranker_min) / ranker_span
        return self._reg_min + t * reg_span

    @staticmethod
    def _compute_ndcg_at_k(
        y_true: np.ndarray, y_pred: np.ndarray, k: int
    ) -> float:
        """Compute NDCG@k.

        sklearn's ndcg_score requires *relevance* to be non-negative; we convert
        by negating log-cost (lower cost = more relevant).
        """
        relevance = -y_true  # higher relevance = lower cost
        # Shift so minimum relevance is 0.
        relevance = relevance - relevance.min()
        try:
            return float(
                ndcg_score(relevance.reshape(1, -1), (-y_pred).reshape(1, -1), k=k)
            )
        except ValueError:
            return float("nan")

    @staticmethod
    def _compute_top10_precision(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Fraction of true top-10% that the proxy also ranks in top-10%."""
        n = len(y_true)
        if n < 2:
            return float("nan")
        k = max(1, n // 10)
        true_top = set(np.argsort(y_true)[:k])
        pred_top = set(np.argsort(y_pred)[:k])
        return len(true_top & pred_top) / k

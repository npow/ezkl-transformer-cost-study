"""Approach B: zero-cost proxy — fit op-weight regressors from architectural features.

No actual ZK proving is required.  The model re-weights the hand-crafted ZK cost
formula for transformers using only the features from :meth:`ProxyFeatures.to_vector`.

Comparison with Approach A:

* **Approach B (this module)** — runs instantly (no proving), but cannot learn
  non-additive cost effects such as power-of-two cliffs or lookup-table sharing.
* **Approach A** — requires actual ZK proving, but can capture non-additive effects.
"""

from __future__ import annotations

import logging
import math
import pickle
from typing import Literal, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

import xgboost as xgb

from proxy.data import Fidelity, ProxyDataset
from proxy.features import ProxyFeatures, TransformerConfig, feature_names

logger = logging.getLogger(__name__)

# Supported model type literals
ModelType = Literal["xgb", "ridge", "rf"]

# Number of training samples below which we fall back to a linear pass-through
_MIN_SAMPLES_FOR_FIT: int = 5

# Index of the ``onnx_constraint_estimate`` column inside the feature vector
_ONNX_CONSTRAINT_IDX: int = feature_names().index("onnx_constraint_estimate")

# Sentinel value used for missing ONNX fields (from features.py)
_MISSING_ONNX_SENTINEL: float = -1.0


class ZeroCostProxy:  # pylint: disable=too-many-instance-attributes
    """Approach B: fit op-weight regressors from architectural features only.

    No actual ZK proving required — uses ONNX proxy estimates (or even just
    analytical features) as supervision.  Fast to train but cannot learn
    non-additive cost effects.

    Implements the ZKCostProxy protocol::

        score(config) -> float          (lower = cheaper to prove)
        fit(dataset) -> None
        uncertainty(config) -> float    (prediction interval width)

    Parameters
    ----------
    model_type:
        Underlying regressor family.  One of ``'xgb'``, ``'ridge'``, ``'rf'``.
    use_onnx_features:
        When ``True`` (default) the three ``onnx_*`` columns are kept in the
        feature vector.  Set to ``False`` to use only pure analytical features
        (useful when ONNX estimates are unavailable).
    min_fidelity:
        Minimum :class:`~proxy.data.Fidelity` required for a record to be
        included in training.  Defaults to :attr:`~proxy.data.Fidelity.ONNX_ESTIMATE`.
    """

    def __init__(
        self,
        model_type: ModelType = "xgb",
        use_onnx_features: bool = True,
        min_fidelity: Fidelity = Fidelity.ONNX_ESTIMATE,
    ) -> None:
        if model_type not in ("xgb", "ridge", "rf"):
            raise ValueError(
                f"model_type must be 'xgb', 'ridge', or 'rf'; got {model_type!r}"
            )

        self.model_type: ModelType = model_type
        self.use_onnx_features: bool = use_onnx_features
        self.min_fidelity: Fidelity = min_fidelity

        # Primary regressor (fitted on median / mean objective)
        self._model: Optional[
            Union[xgb.XGBRegressor, RandomForestRegressor, Ridge]
        ] = None

        # Quantile models for XGBoost uncertainty (p10 and p90)
        self._model_low: Optional[xgb.XGBRegressor] = None
        self._model_high: Optional[xgb.XGBRegressor] = None

        # Polynomial feature transformer for Ridge
        self._poly: Optional[PolynomialFeatures] = None

        # Whether we are in fall-back (linear pass-through) mode
        self._fallback: bool = False

        # Feature column mask applied before prediction
        self._feature_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, dataset: ProxyDataset) -> None:
        """Train on all records with labels >= *min_fidelity*.

        When the filtered dataset has fewer than five records the proxy falls
        back to a linear pass-through that reads the ONNX constraint estimate
        directly (or falls back to ``total_macs`` when ONNX fields are absent).

        Parameters
        ----------
        dataset:
            Training data.  Records without a numeric label or below
            :attr:`min_fidelity` are excluded automatically.
        """
        x_mat, y_vec = dataset.to_xy(min_fidelity=self.min_fidelity)

        self._feature_mask = self._build_feature_mask()

        if len(y_vec) < _MIN_SAMPLES_FOR_FIT:
            logger.warning(
                "ZeroCostProxy.fit: only %d labelled records meet the fidelity "
                "threshold — falling back to linear pass-through.",
                len(y_vec),
            )
            self._fallback = True
            return

        self._fallback = False
        x_filtered = x_mat[:, self._feature_mask]

        if self.model_type == "xgb":
            self._fit_xgb(x_filtered, y_vec)
        elif self.model_type == "rf":
            self._fit_rf(x_filtered, y_vec)
        else:
            self._fit_ridge(x_filtered, y_vec)

    def _fit_xgb(self, x_mat: np.ndarray, y_vec: np.ndarray) -> None:
        """Fit the three XGBoost models (median + two quantiles).

        ``tree_method='hist'`` is the recommended method for XGBoost >= 2.0
        and is substantially faster than the default ``'exact'`` method,
        particularly on small datasets.  ``nthread=1`` pins XGBoost to a single
        thread; without this, XGBoost auto-detects the number of CPUs and spawns
        many threads for each tree, which creates enormous overhead on machines
        with high CPU counts when datasets are tiny (the threading cost dwarfs
        the actual computation).
        """
        common_kwargs: dict = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "nthread": 1,
        }
        self._model = xgb.XGBRegressor(
            objective="reg:squarederror",
            **common_kwargs,
        )
        self._model.fit(x_mat, y_vec)

        self._model_low = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.1,
            **common_kwargs,
        )
        self._model_low.fit(x_mat, y_vec)

        self._model_high = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.9,
            **common_kwargs,
        )
        self._model_high.fit(x_mat, y_vec)

    def _fit_rf(self, x_mat: np.ndarray, y_vec: np.ndarray) -> None:
        """Fit a Random Forest regressor."""
        self._model = RandomForestRegressor(n_estimators=100)
        self._model.fit(x_mat, y_vec)

    def _fit_ridge(self, x_mat: np.ndarray, y_vec: np.ndarray) -> None:
        """Fit a Ridge regressor with degree-2 polynomial features."""
        self._poly = PolynomialFeatures(degree=2, include_bias=False)
        x_poly = self._poly.fit_transform(x_mat)
        self._model = Ridge(alpha=1.0)
        self._model.fit(x_poly, y_vec)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def score(self, config: TransformerConfig) -> float:
        """Predict log(ZK cost).  Lower means cheaper to prove.

        Parameters
        ----------
        config:
            Architecture to score.

        Returns
        -------
        float
            Predicted log-cost.  Falls back to a linear estimate when the
            proxy is not yet fitted or was trained on too few samples.
        """
        if self._fallback or not self.is_fitted():
            return self._fallback_score(config)

        x_vec = self._extract_features(config)
        return float(self._model.predict(x_vec)[0])  # type: ignore[union-attr]

    def uncertainty(self, config: TransformerConfig) -> float:
        """Estimate prediction uncertainty (non-negative).

        * **XGBoost** — quantile regression interval width: ``p90 − p10``.
        * **RF** — standard deviation of individual tree predictions.
        * **Ridge** — always returns ``0.0`` (no uncertainty estimate).

        Parameters
        ----------
        config:
            Architecture to evaluate.

        Returns
        -------
        float
            Non-negative uncertainty estimate.
        """
        if self._fallback or not self.is_fitted():
            return 0.0

        x_vec = self._extract_features(config)

        if self.model_type == "xgb":
            return self._xgb_uncertainty(x_vec)
        if self.model_type == "rf":
            return self._rf_uncertainty(x_vec)
        # Ridge
        return 0.0

    def _xgb_uncertainty(self, x_vec: np.ndarray) -> float:
        """Return p90 − p10 from the two quantile XGBoost models."""
        if self._model_low is None or self._model_high is None:
            return 0.0
        p10 = float(self._model_low.predict(x_vec)[0])
        p90 = float(self._model_high.predict(x_vec)[0])
        return max(0.0, p90 - p10)

    def _rf_uncertainty(self, x_vec: np.ndarray) -> float:
        """Return the standard deviation of tree-level predictions."""
        rf_model = self._model
        if not isinstance(rf_model, RandomForestRegressor):
            return 0.0
        tree_preds = np.array(
            [tree.predict(x_vec)[0] for tree in rf_model.estimators_]
        )
        return float(np.std(tree_preds))

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """Return ``{feature_name: importance}`` normalised to sum to 1.0.

        Returns
        -------
        dict[str, float]
            Ordered by importance (descending).  Returns an empty dict when
            the proxy is not fitted or is in fall-back mode.

        Raises
        ------
        RuntimeError
            If the underlying model does not expose feature importances (Ridge).
        """
        if self._fallback or not self.is_fitted():
            return {}

        names = self._active_feature_names()

        if self.model_type == "xgb":
            raw = self._model.feature_importances_  # type: ignore[union-attr]
        elif self.model_type == "rf":
            raw = self._model.feature_importances_  # type: ignore[union-attr]
        else:
            # Ridge: use absolute coefficient magnitudes (no built-in importances).
            ridge_model = self._model
            if not isinstance(ridge_model, Ridge):
                return {}
            coef = ridge_model.coef_
            # Coef vector corresponds to polynomial features; map back to
            # the original feature space by summing contributions per original feature.
            if self._poly is not None:
                powers = self._poly.powers_  # shape (n_poly_features, n_orig_features)
                # Sum absolute coefficient per original feature (any poly term that
                # involves that feature contributes).
                n_orig = len(names)
                aggregated = np.zeros(n_orig)
                for i, power_row in enumerate(powers):
                    involved = power_row > 0  # boolean mask of original features used
                    aggregated += np.abs(coef[i]) * involved
                raw = aggregated
            else:
                raw = np.abs(coef)

        raw = np.asarray(raw, dtype=np.float64)
        total = raw.sum()
        if total <= 0.0:
            # Uniform fallback to avoid division by zero
            normalised = np.ones(len(raw)) / max(len(raw), 1)
        else:
            normalised = raw / total

        return {name: float(imp) for name, imp in zip(names, normalised)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted proxy to *path* using :mod:`pickle`.

        Parameters
        ----------
        path:
            Destination file path (will be created or overwritten).
        """
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("ZeroCostProxy saved to %r", path)

    @classmethod
    def load(cls, path: str) -> "ZeroCostProxy":
        """Deserialise a :class:`ZeroCostProxy` from *path*.

        Parameters
        ----------
        path:
            Path previously written by :meth:`save`.

        Returns
        -------
        ZeroCostProxy
            The loaded proxy.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        TypeError
            If the unpickled object is not a :class:`ZeroCostProxy`.
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__} at {path!r}; got {type(obj).__name__}"
            )
        logger.debug("ZeroCostProxy loaded from %r", path)
        return obj

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        """Return ``True`` if the model has been fitted (or is in fall-back mode)."""
        return self._fallback or self._model is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_mask(self) -> np.ndarray:
        """Return a boolean mask selecting the active feature columns.

        When ``use_onnx_features=False`` the three ``onnx_*`` columns are
        excluded from training and prediction.
        """
        all_names = feature_names()
        onnx_cols = frozenset(
            {"onnx_constraint_estimate", "onnx_nonlinear_ops", "onnx_linear_ops"}
        )
        mask = np.array(
            [
                True if self.use_onnx_features else (name not in onnx_cols)
                for name in all_names
            ],
            dtype=bool,
        )
        return mask

    def _active_feature_names(self) -> list[str]:
        """Return the feature names that correspond to the current mask."""
        all_names = feature_names()
        mask = self._feature_mask
        if mask is None:
            return all_names
        return [name for name, keep in zip(all_names, mask) if keep]

    def _extract_features(self, config: TransformerConfig) -> np.ndarray:
        """Extract and optionally transform features for a single config.

        Returns a 2-D array of shape ``(1, n_active_features)`` suitable for
        ``model.predict()``.  For Ridge, polynomial expansion is applied.
        """
        features = ProxyFeatures.from_config(config)
        x_vec = features.to_vector()

        if self._feature_mask is not None:
            x_vec = x_vec[self._feature_mask]

        x_2d = x_vec.reshape(1, -1)

        if self.model_type == "ridge" and self._poly is not None:
            x_2d = self._poly.transform(x_2d)

        return x_2d

    def _fallback_score(self, config: TransformerConfig) -> float:
        """Linear pass-through fallback when not enough data to fit a model.

        Prefers the ONNX constraint estimate when available; otherwise uses
        log(total_macs) as a crude analytical proxy.
        """
        features = ProxyFeatures.from_config(config)
        x_vec = features.to_vector()
        onnx_val = x_vec[_ONNX_CONSTRAINT_IDX]
        if onnx_val > 0.0:
            # Already a raw count — return log so it's on the same scale as labels
            return math.log(onnx_val)
        # Fall back to log(total_macs) for purely analytical features
        total_macs = float(features.total_macs)
        if total_macs > 0.0:
            return math.log(total_macs)
        # Last resort: return 0.0
        return 0.0

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted() else "unfitted"
        fallback_str = " [fallback]" if self._fallback else ""
        return (
            f"ZeroCostProxy("
            f"model_type={self.model_type!r}, "
            f"use_onnx_features={self.use_onnx_features}, "
            f"min_fidelity={self.min_fidelity}, "
            f"{fitted_str}{fallback_str})"
        )

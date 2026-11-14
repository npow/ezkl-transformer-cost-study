"""Evaluation harness for comparing all three proxy strategies (Table 2).

This module trains each proxy on a common train split and evaluates it on a
held-out test split, computing a standard suite of ranking-quality metrics.
The :class:`ProxyEvaluator` class produces :class:`ProxyEvalResult` dataclasses
that can be formatted as a Markdown table for inclusion in the paper.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

# matplotlib.use("Agg") must be called before importing pyplot.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import ndcg_score  # noqa: E402

from proxy.data import Fidelity, ProxyDataset, ProxyRecord  # noqa: E402
from proxy.features import TransformerConfig  # noqa: E402
from proxy.strategies.kernel_decomp import KernelDecompositionProxy  # noqa: E402
from proxy.strategies.multifidelity import MultiFidelityProxy  # noqa: E402
from proxy.strategies.zero_cost import ZeroCostProxy  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProxyEvalResult:
    """Metrics for one proxy strategy on a held-out test set."""

    proxy_name: str
    """Human-readable name: ``'MultiFidelity'``, ``'ZeroCost'``, or ``'KernelDecomp'``."""

    n_train: int
    """Number of training records used to fit the proxy."""

    n_test: int
    """Number of held-out test records used for evaluation."""

    spearman_r: float
    """Spearman rank correlation coefficient (primary metric)."""

    spearman_p: float
    """Two-tailed p-value for the Spearman correlation."""

    top10_precision: float
    """P@10%: fraction of true top-10% cheapest configs found in predicted top-10%."""

    top1_precision: float
    """P@1%: fraction of true top-1% cheapest configs found in predicted top-1%."""

    ndcg_at_10: float
    """NDCG@10: normalised discounted cumulative gain at 10 positions."""

    mae_log: float
    """Mean absolute error on log(cost) predictions."""

    calibration_error: float
    """Mean |predicted_quantile - actual_quantile| across decile bins."""

    training_time_s: float
    """Wall-clock seconds required to fit the proxy."""

    inference_time_ms: float
    """Mean inference latency in milliseconds per candidate."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_top_k_precision(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute precision at *k*: fraction of true top-*k* in predicted top-*k*.

    Parameters
    ----------
    y_true:
        Ground-truth log-costs (lower = cheaper = better).
    y_pred:
        Predicted log-costs.
    k:
        Number of candidates in the top-k set.

    Returns
    -------
    float
        Precision in ``[0, 1]``, or ``nan`` when ``k == 0``.
    """
    if k <= 0:
        return float("nan")
    true_top = set(np.argsort(y_true)[:k])
    pred_top = set(np.argsort(y_pred)[:k])
    return len(true_top & pred_top) / k


def _compute_ndcg_at_10(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute NDCG@10 using relevance scores derived from inverse normalised cost.

    Cheaper architectures (lower log-cost) receive higher relevance:
    ``relevance = 1 / (1 + normalised_cost)`` where ``normalised_cost`` is
    ``(log_cost - min) / (max - min)`` clipped to ``[0, 1]``.

    Parameters
    ----------
    y_true:
        Ground-truth log-costs.
    y_pred:
        Predicted scores (lower = predicted cheaper).

    Returns
    -------
    float
        NDCG@10 in ``[0, 1]``, or ``nan`` on error.
    """
    n = len(y_true)
    if n < 2:
        return float("nan")

    y_min, y_max = y_true.min(), y_true.max()
    span = y_max - y_min
    if span == 0.0:
        # All costs identical — perfect ranking by definition.
        return 1.0

    normalised = (y_true - y_min) / span
    relevance = 1.0 / (1.0 + normalised)  # higher for cheaper

    k = min(10, n)
    try:
        return float(
            ndcg_score(
                relevance.reshape(1, -1),
                (-y_pred).reshape(1, -1),  # negate so lower pred → higher score
                k=k,
            )
        )
    except ValueError:
        return float("nan")


def _compute_calibration_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute decile calibration error.

    Bins predicted log-costs into 10 equal-width decile bins and checks what
    fraction of true values fall into each bin. The expected fraction per bin
    is 10 %. Calibration error is the mean absolute deviation from 10 %.

    Parameters
    ----------
    y_true:
        Ground-truth log-costs.
    y_pred:
        Predicted log-costs.

    Returns
    -------
    float
        Mean |actual_fraction - 0.10| across the 10 decile bins, in
        ``[0, 1]``.  Returns ``0.0`` when fewer than 10 samples are available.
    """
    n = len(y_true)
    if n < 10:
        return 0.0

    n_bins = 10
    pred_deciles = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    # Remove duplicate bin edges to avoid zero-width bins.
    unique_edges = np.unique(pred_deciles)
    if len(unique_edges) < 2:
        return 0.0

    # Count true values in each bin.
    bin_indices = np.digitize(y_true, unique_edges[:-1], right=False)
    n_actual_bins = len(unique_edges) - 1

    errors: list[float] = []
    for b in range(1, n_actual_bins + 1):
        actual_frac = float(np.sum(bin_indices == b)) / n
        expected_frac = 1.0 / n_bins
        errors.append(abs(actual_frac - expected_frac))

    return float(np.mean(errors)) if errors else 0.0


def _fit_and_time(
    proxy: MultiFidelityProxy | ZeroCostProxy | KernelDecompositionProxy,
    dataset: ProxyDataset,
) -> float:
    """Fit *proxy* on *dataset* and return wall-clock training time in seconds."""
    t0 = time.perf_counter()
    proxy.fit(dataset)
    return time.perf_counter() - t0


def _predict_batch(
    proxy: MultiFidelityProxy | ZeroCostProxy | KernelDecompositionProxy,
    configs: list[TransformerConfig],
) -> tuple[list[float], float]:
    """Score all *configs* and return ``(scores, inference_time_ms_per_item)``.

    Parameters
    ----------
    proxy:
        A fitted proxy with a ``score(config) -> float`` method.
    configs:
        Transformer configs to score.

    Returns
    -------
    tuple[list[float], float]
        ``(scores, ms_per_item)`` where *ms_per_item* is the per-item mean
        inference latency in milliseconds.
    """
    if not configs:
        return [], float("nan")

    t0 = time.perf_counter()
    scores = [proxy.score(cfg) for cfg in configs]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return scores, elapsed_ms / len(configs)


def _compute_metrics(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    proxy_name: str,
    n_train: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_time_s: float,
    inference_time_ms: float,
) -> ProxyEvalResult:
    """Compute all evaluation metrics from ground-truth and predicted log-costs."""
    n = len(y_true)

    spearman_r, spearman_p = (float("nan"), float("nan"))
    if n >= 3:
        r_val, p_val = spearmanr(y_true, y_pred)
        spearman_r = float(r_val)
        spearman_p = float(p_val)

    k10 = max(1, n // 10)
    k1 = max(1, n // 100)

    top10 = _compute_top_k_precision(y_true, y_pred, k10)
    top1 = _compute_top_k_precision(y_true, y_pred, k1)
    ndcg10 = _compute_ndcg_at_10(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred))) if n > 0 else float("nan")
    cal_err = _compute_calibration_error(y_true, y_pred)

    return ProxyEvalResult(
        proxy_name=proxy_name,
        n_train=n_train,
        n_test=n,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        top10_precision=top10,
        top1_precision=top1,
        ndcg_at_10=ndcg10,
        mae_log=mae,
        calibration_error=cal_err,
        training_time_s=training_time_s,
        inference_time_ms=inference_time_ms,
    )


# ---------------------------------------------------------------------------
# ProxyEvaluator
# ---------------------------------------------------------------------------


class ProxyEvaluator:
    """Compare all three proxy strategies on held-out data.

    This is the paper's Table 2.

    Parameters
    ----------
    dataset:
        Full labelled dataset.  Records are split into train / test sets once
        at construction time using *seed* and *test_fraction*.
    test_fraction:
        Fraction of records to hold out for evaluation (default 0.2).
    seed:
        Random seed for reproducible train/test split (default 42).
    """

    def __init__(
        self,
        dataset: ProxyDataset,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction!r}")

        self._dataset = dataset
        self._test_fraction = test_fraction
        self._seed = seed
        self._train_dataset, self._test_dataset, self._test_configs = self._split(
            dataset, test_fraction, seed
        )

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        proxy_a: MultiFidelityProxy,
        proxy_b: ZeroCostProxy,
        proxy_c: Optional[KernelDecompositionProxy] = None,
    ) -> list[ProxyEvalResult]:
        """Train all proxies on the train split, evaluate on the test split.

        Each proxy is fitted independently on the same training dataset and
        then scored on the held-out test configs.  Results are returned in the
        order A → B → C (C is omitted when *proxy_c* is ``None``).

        Parameters
        ----------
        proxy_a:
            :class:`MultiFidelityProxy` — Approach A (GBM multi-fidelity).
        proxy_b:
            :class:`ZeroCostProxy` — Approach B (zero-cost analytical).
        proxy_c:
            :class:`KernelDecompositionProxy` — Approach C (kernel decomp).
            Optional; pass ``None`` to skip.

        Returns
        -------
        list[ProxyEvalResult]
            One result per evaluated proxy.
        """
        # Collect ground-truth labels for test configs.
        y_true = self._test_labels()
        if len(y_true) == 0:
            logger.warning(
                "ProxyEvaluator.evaluate_all: no labelled records in test split."
            )
            return []

        n_train = len(self._train_dataset)
        proxies: list[
            tuple[str, MultiFidelityProxy | ZeroCostProxy | KernelDecompositionProxy]
        ] = [
            ("MultiFidelity", proxy_a),
            ("ZeroCost", proxy_b),
        ]
        if proxy_c is not None:
            proxies.append(("KernelDecomp", proxy_c))

        results: list[ProxyEvalResult] = []
        for name, proxy in proxies:
            logger.info("Fitting proxy %r on %d training records …", name, n_train)
            train_time = _fit_and_time(proxy, self._train_dataset)
            scores, inf_ms = _predict_batch(proxy, self._test_configs)
            y_pred = np.array(scores, dtype=np.float64)
            result = _compute_metrics(
                proxy_name=name,
                n_train=n_train,
                y_true=y_true,
                y_pred=y_pred,
                training_time_s=train_time,
                inference_time_ms=inf_ms,
            )
            results.append(result)
            logger.info(
                "Proxy %r: spearman_r=%.3f  top10_prec=%.3f  ndcg@10=%.3f  "
                "train=%.2fs  inf=%.2fms",
                name,
                result.spearman_r,
                result.top10_precision,
                result.ndcg_at_10,
                result.training_time_s,
                result.inference_time_ms,
            )

        return results

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------

    def to_markdown_table(self, results: list[ProxyEvalResult]) -> str:
        """Format *results* as a Markdown table for inclusion in the paper.

        Example output::

            | Proxy | Spearman ρ | P@10% | P@1% | NDCG@10 | MAE(log) | Train time |
            |-------|------------|-------|------|---------|----------|------------|
            | MultiFidelity (A) | 0.91 | 0.82 | 0.71 | 0.87 | 0.23 | 1.2s |
            | ZeroCost (B)      | 0.78 | 0.65 | 0.44 | 0.71 | 0.41 | 0.03s |
            | KernelDecomp (C)  | 0.69 | 0.58 | 0.31 | 0.63 | 0.55 | 0.0s |

        Parameters
        ----------
        results:
            List of :class:`ProxyEvalResult` instances to display.

        Returns
        -------
        str
            Multi-line Markdown table string.
        """
        label_map: dict[str, str] = {
            "MultiFidelity": "MultiFidelity (A)",
            "ZeroCost": "ZeroCost (B)",
            "KernelDecomp": "KernelDecomp (C)",
        }
        header = (
            "| Proxy | Spearman ρ | P@10% | P@1% | NDCG@10 | MAE(log) | Train time |"
        )
        separator = (
            "|-------|------------|-------|------|---------|----------|------------|"
        )
        rows: list[str] = [header, separator]
        for r in results:
            label = label_map.get(r.proxy_name, r.proxy_name)
            train_str = self._format_time(r.training_time_s)
            rows.append(
                f"| {label} "
                f"| {r.spearman_r:.2f} "
                f"| {r.top10_precision:.2f} "
                f"| {r.top1_precision:.2f} "
                f"| {r.ndcg_at_10:.2f} "
                f"| {r.mae_log:.2f} "
                f"| {train_str} |"
            )
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Calibration plot
    # ------------------------------------------------------------------

    def plot_calibration(  # pylint: disable=too-many-locals
        self,
        results: list[ProxyEvalResult],
        proxy_scores: dict[str, list[float]],
        true_labels: list[float],
        output_path: str,
    ) -> None:
        """Plot calibration curves: predicted rank percentile vs true rank percentile.

        For each proxy the predicted scores are ranked and the rank percentile
        is plotted against the corresponding true-label rank percentile.  A
        perfect proxy produces a diagonal line.

        Parameters
        ----------
        results:
            Evaluation results (used only for proxy names).
        proxy_scores:
            ``{proxy_name: [scores]}`` — raw predicted scores per proxy.
        true_labels:
            Ground-truth log-costs (same length as each score list).
        output_path:
            Destination PNG file path.
        """
        y_true = np.array(true_labels, dtype=np.float64)
        n = len(y_true)
        if n == 0:
            logger.warning("plot_calibration: no data; skipping plot.")
            return

        true_rank_pct = np.argsort(np.argsort(y_true)) / max(n - 1, 1) * 100.0

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="Perfect proxy")

        result_names = {r.proxy_name for r in results}
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_cycle = iter(colors)
        color_map: dict[str, str] = {}

        for name in result_names:
            scores = proxy_scores.get(name)
            if scores is None or len(scores) != n:
                logger.warning(
                    "plot_calibration: scores for %r missing or length mismatch; skip.",
                    name,
                )
                continue
            pred_rank_pct = (
                np.argsort(np.argsort(np.array(scores, dtype=np.float64)))
                / max(n - 1, 1)
                * 100.0
            )
            color = color_map.setdefault(name, next(color_cycle))
            ax.scatter(
                pred_rank_pct,
                true_rank_pct,
                s=10,
                alpha=0.6,
                color=color,
                label=name,
            )

        ax.set_xlabel("Predicted rank percentile")
        ax.set_ylabel("True rank percentile")
        ax.set_title("Proxy calibration: predicted vs true rank")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Calibration plot saved to %s", output_path)

    # ------------------------------------------------------------------
    # Pareto coverage plot
    # ------------------------------------------------------------------

    def plot_pareto_coverage(  # pylint: disable=unused-argument,too-many-locals
        self,
        results: list[ProxyEvalResult],
        configs: list[TransformerConfig],
        true_costs: list[float],
        output_path: str,
    ) -> None:
        """Plot Pareto frontier coverage.

        Shows the cumulative fraction of true top-k cheap configs recovered as
        k grows from 1 to n, for each proxy.  X-axis is the true ZK cost rank
        (1 = cheapest); Y-axis is the cumulative fraction of the true top-k
        discovered by the proxy's top-k selection.

        Parameters
        ----------
        results:
            Evaluation results carrying ``proxy_name`` for labelling.
        configs:
            Transformer configs in the test set.
        true_costs:
            True log-costs aligned with *configs*.
        output_path:
            Destination PNG file path.
        """
        y_true = np.array(true_costs, dtype=np.float64)
        n = len(y_true)
        if n == 0 or not results:
            logger.warning("plot_pareto_coverage: no data; skipping plot.")
            return

        true_order = np.argsort(y_true)  # ascending cost order (cheapest first)

        fig, ax = plt.subplots(figsize=(7, 5))
        ranks = np.arange(1, n + 1)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_cycle = iter(colors)

        for r in results:
            # Obtain predicted scores for each config in the test set using the
            # proxy_name stored in the result.  Since the proxy objects are not
            # stored on the evaluator, this method accepts pre-computed results
            # and recomputes coverage from y_pred rank directly.
            # We approximate by assuming the test configs are scored in the
            # same order stored in self._test_configs; the caller must ensure
            # alignment.
            #
            # Coverage@k: |true_top_k ∩ pred_top_k| / k
            pred_scores_attr = getattr(r, "_pred_scores", None)
            if pred_scores_attr is None:
                logger.debug(
                    "plot_pareto_coverage: no cached scores for %r; "
                    "coverage curve will be flat.",
                    r.proxy_name,
                )
                ax.plot(
                    ranks,
                    np.linspace(1.0 / n, 1.0, n),
                    "--",
                    color=next(color_cycle),
                    label=f"{r.proxy_name} (no score cache)",
                )
                continue

            y_pred = np.array(pred_scores_attr, dtype=np.float64)
            pred_order = np.argsort(y_pred)
            coverage = np.array(
                [
                    len(set(true_order[:k]) & set(pred_order[:k])) / k
                    for k in range(1, n + 1)
                ],
                dtype=np.float64,
            )
            ax.plot(ranks, coverage, color=next(color_cycle), label=r.proxy_name)

        ax.set_xlabel("True ZK cost rank (1 = cheapest)")
        ax.set_ylabel("Cumulative fraction of true top-k recovered")
        ax.set_title("Pareto frontier coverage by proxy")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(1, n)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Pareto coverage plot saved to %s", output_path)

    # ------------------------------------------------------------------
    # Learning curve
    # ------------------------------------------------------------------

    def learning_curve(  # pylint: disable=too-many-locals
        self,
        proxy_a: MultiFidelityProxy,
        full_dataset: ProxyDataset,
        train_sizes: list[int] | None = None,
        output_path: Optional[str] = None,
    ) -> dict[int, float]:
        """Compute/plot Spearman r vs training-set size for Approach A.

        For each value in *train_sizes*, a new :class:`MultiFidelityProxy` with
        the same hyperparameters as *proxy_a* is fitted on a random subsample of
        *full_dataset* of the requested size, and its Spearman r on the full
        held-out test split is recorded.

        This is a key paper figure: shows how quickly the proxy improves with
        more ZK proofs.

        Parameters
        ----------
        proxy_a:
            Reference :class:`MultiFidelityProxy`.  Hyperparameters are cloned
            for each sub-proxy.
        full_dataset:
            Complete labelled dataset (train + test).
        train_sizes:
            List of training-set sizes to evaluate.  Defaults to
            ``[5, 10, 20, 50, 100]``.
        output_path:
            When provided, saves a PNG learning-curve plot to this path.

        Returns
        -------
        dict[int, float]
            ``{n_train: spearman_r}`` for each evaluated training size.
        """
        if train_sizes is None:
            train_sizes = [5, 10, 20, 50, 100]

        rng = np.random.default_rng(self._seed)
        # pylint: disable-next=invalid-name
        X_full, y_full = full_dataset.to_xy(min_fidelity=Fidelity.ONNX_ESTIMATE)
        n_full = len(y_full)

        # Build a one-off held-out split from the full dataset for LC evaluation.
        n_test = max(1, int(n_full * self._test_fraction))
        perm = rng.permutation(n_full)
        test_idx = perm[:n_test]
        X_test = X_full[test_idx]  # pylint: disable=invalid-name
        y_test = y_full[test_idx]

        curve: dict[int, float] = {}
        for n_train in sorted(set(train_sizes)):
            if n_train <= 0:
                continue
            if n_train > n_full - n_test:
                available = n_full - n_test
                logger.warning(
                    "learning_curve: requested n_train=%d but only %d records "
                    "available after test split; using %d.",
                    n_train,
                    available,
                    available,
                )
                actual_n = available
            else:
                actual_n = n_train

            train_candidates = perm[n_test:]
            chosen_idx = train_candidates[:actual_n]
            X_train = X_full[chosen_idx]  # pylint: disable=invalid-name
            y_train = y_full[chosen_idx]

            # Build a minimal ProxyDataset from the subsample so we can call fit().
            sub_dataset = _arrays_to_dataset(X_train, y_train, full_dataset)

            # pylint: disable=protected-access
            sub_proxy = MultiFidelityProxy(
                n_estimators=proxy_a._n_estimators,
                max_depth=proxy_a._max_depth,
                learning_rate=proxy_a._learning_rate,
                fidelity_weights=dict(proxy_a._fidelity_weights),
            )
            sub_proxy.fit(sub_dataset)

            # Predict on held-out test set using pre-computed feature vectors.
            y_pred_test = _predict_from_feature_matrix(sub_proxy, X_test)
            if len(y_pred_test) < 3:
                curve[n_train] = float("nan")
                continue

            r_val, _ = spearmanr(y_test, y_pred_test)
            curve[n_train] = float(r_val)
            logger.info(
                "learning_curve n_train=%d -> spearman_r=%.3f", n_train, float(r_val)
            )

        if output_path is not None:
            self._save_learning_curve_plot(curve, output_path)

        return curve

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split(
        dataset: ProxyDataset,
        test_fraction: float,
        seed: int,
    ) -> tuple[ProxyDataset, ProxyDataset, list[TransformerConfig]]:
        """Split *dataset* into train and test :class:`ProxyDataset` objects.

        Returns
        -------
        tuple[ProxyDataset, ProxyDataset, list[TransformerConfig]]
            ``(train_dataset, test_dataset, test_configs)``
        """
        # pylint: disable=protected-access
        records = list(dataset._records)
        n = len(records)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        n_test = max(1, int(n * test_fraction))
        test_indices = set(indices[:n_test].tolist())

        train_ds = ProxyDataset()
        test_ds = ProxyDataset()
        test_configs: list[TransformerConfig] = []

        for i, rec in enumerate(records):
            if i in test_indices:
                test_ds.add(rec)
                test_configs.append(rec.config)
            else:
                train_ds.add(rec)

        return train_ds, test_ds, test_configs

    def _test_labels(self) -> np.ndarray:
        """Extract ground-truth log-cost labels for the test split."""
        # pylint: disable=protected-access
        labels: list[float] = []
        for rec in self._test_dataset._records:
            label = rec.best_label()
            if label is not None:
                labels.append(label)
        return np.array(labels, dtype=np.float64)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format a duration as a short human-readable string."""
        if math.isnan(seconds):
            return "nan"
        if seconds < 0.001:
            return "0.0s"
        if seconds < 1.0:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        return f"{seconds / 60:.1f}m"

    def _save_learning_curve_plot(
        self, curve: dict[int, float], output_path: str
    ) -> None:
        """Save a learning-curve PNG to *output_path*."""
        sizes = sorted(curve.keys())
        spearman_vals = [curve[s] for s in sizes]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sizes, spearman_vals, marker="o", linewidth=1.5, markersize=5)
        ax.set_xlabel("Training set size (# proofs)")
        ax.set_ylabel("Spearman ρ (held-out test)")
        ax.set_title("Learning curve — MultiFidelity proxy (Approach A)")
        ax.set_ylim(-1.0, 1.05)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Learning curve plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Utility: build a ProxyDataset from (X, y) arrays
# ---------------------------------------------------------------------------


def _arrays_to_dataset(  # pylint: disable=invalid-name
    X: np.ndarray,
    y: np.ndarray,
    reference_dataset: ProxyDataset,
) -> ProxyDataset:
    """Create a lightweight :class:`ProxyDataset` from feature / label arrays.

    This is used by :meth:`ProxyEvaluator.learning_curve` to avoid re-fitting
    from scratch by sub-sampling already-extracted arrays.  A synthetic
    :class:`ProxyRecord` is constructed for each row — the config fields are
    reconstructed from the feature vector on a best-effort basis, but the
    important thing is that ``to_xy()`` returns the supplied arrays.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(n, n_features)``.
    y:
        Log-cost labels, shape ``(n,)``.
    reference_dataset:
        Source dataset; used to borrow :class:`TransformerConfig` objects
        whose feature vectors match the rows in *X* (by index in the
        reference dataset's record list).

    Returns
    -------
    ProxyDataset
        Minimal dataset with *n* records.
    """
    # pylint: disable=protected-access
    ref_records = reference_dataset._records

    # Build a feature-vector → record index for fast lookup.
    ds = ProxyDataset()
    n = len(y)

    for i in range(n):
        x_row = X[i]
        # Try to find a matching record in the reference dataset.
        matched_rec: Optional[ProxyRecord] = None
        for rec in ref_records:
            label = rec.best_label()
            if label is None:
                continue
            ref_vec = rec.features.to_vector()
            if np.allclose(ref_vec, x_row, rtol=1e-5, atol=1e-8):
                matched_rec = rec
                break

        if matched_rec is not None:
            # Clone the record with the known label.
            new_rec = ProxyRecord(
                config=matched_rec.config,
                features=matched_rec.features,
                onnx_estimate=matched_rec.onnx_estimate,
                ezkl_calibration=matched_rec.ezkl_calibration,
                ezkl_prove_time=matched_rec.ezkl_prove_time,
                ezkl_num_rows=matched_rec.ezkl_num_rows,
                backend=matched_rec.backend,
            )
        else:
            # Fallback: synthesise a minimal record from the feature vector.
            # Use the first record's config as a template (imperfect but
            # sufficient for the learning-curve use case).
            template = ref_records[0] if ref_records else None
            if template is None:
                continue
            new_rec = ProxyRecord(
                config=template.config,
                features=template.features,
                onnx_estimate=int(math.exp(y[i])),
            )

        ds.add(new_rec)

    return ds


def _predict_from_feature_matrix(  # pylint: disable=invalid-name
    proxy: MultiFidelityProxy,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Score rows of *feature_matrix* using the proxy's fitted regressor.

    Bypasses the ``score(config)`` call and feeds pre-computed feature vectors
    directly to the fitted regressor.  This avoids re-extracting features in
    the learning-curve loop and works with any sklearn-compatible regressor
    monkey-patched onto the proxy during testing.

    Parameters
    ----------
    proxy:
        A fitted :class:`MultiFidelityProxy`.
    feature_matrix:
        Pre-extracted feature matrix, shape ``(n, n_features)``.

    Returns
    -------
    np.ndarray
        Predicted log-costs, shape ``(n,)``.
    """
    # pylint: disable=protected-access
    if proxy._regressor is None:
        fallback = proxy._fallback_mean if proxy._fallback_mean is not None else 0.0
        return np.full(len(feature_matrix), fallback, dtype=np.float64)
    return proxy._regressor.predict(feature_matrix).astype(np.float64)

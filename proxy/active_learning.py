"""Active learning acquisition function for transformer ZK proving candidate selection.

Selects which transformer architecture configs are most worth spending ZK
proving time on by combining four signals:

* ``uncertainty``      — proxy prediction uncertainty (higher = less certain)
* ``disagreement``     — |multifidelity_score - zero_cost_score| (normalized)
* ``pareto_proximity`` — proximity to current Pareto frontier (higher = closer)
* ``pow2_cliff``       — EZKL power-of-two row boundary proximity (higher = riskier)

The composite score is a weighted sum; higher composite → higher priority for
proving.  A greedy diversity-penalized batch selector prevents the chosen batch
from containing near-duplicate architectures.

When ``use_ehvi=True`` (the default), the acquisition function switches to
Expected Hypervolume Improvement (EHVI) as the primary score.  EHVI is the
principled multi-objective Bayesian optimisation criterion and outperforms the
heuristic composite on analytical oracle benchmarks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from proxy.accuracy_proxies import decoder_params_score
from proxy.data import ProxyDataset
from proxy.features import ProxyFeatures, TransformerConfig
from proxy.strategies.kernel_decomp import KernelDecompositionProxy
from proxy.strategies.multifidelity import MultiFidelityProxy
from proxy.strategies.zero_cost import ZeroCostProxy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default acquisition weights
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "uncertainty": 0.4,
    "disagreement": 0.3,
    "pareto": 0.2,
    "cliff": 0.1,
}

# Minimum number of dataset records required to build a meaningful Pareto
# frontier; below this threshold pareto_proximity falls back to 0.5.
_MIN_PARETO_RECORDS: int = 2

# Small epsilon for cosine similarity normalisation.
_COSINE_EPS: float = 1e-10


# ---------------------------------------------------------------------------
# EHVI helpers
# ---------------------------------------------------------------------------


def _update_pareto_front(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the non-dominated subset of *points* (maximisation in both dims).

    A point ``p`` is dominated if there exists another point ``q`` such that
    ``q[0] >= p[0]`` and ``q[1] >= p[1]`` with at least one strict inequality.

    Parameters
    ----------
    points:
        Candidate points in 2-D maximisation space.

    Returns
    -------
    list[tuple[float, float]]
        Non-dominated subset; order is not guaranteed.
    """
    pareto: list[tuple[float, float]] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            # ``other`` dominates ``candidate`` when it is at least as good on
            # both axes and strictly better on at least one.
            if (
                other[0] >= candidate[0]
                and other[1] >= candidate[1]
                and (other[0] > candidate[0] or other[1] > candidate[1])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return pareto


def _compute_2d_hypervolume(
    front: list[tuple[float, float]],
    ref: tuple[float, float],
) -> float:
    """Compute the 2-D hypervolume dominated by *front* w.r.t. *ref*.

    Both objectives are in maximisation space; *ref* is the reference (nadir)
    point that must be strictly dominated by front points for them to contribute
    area.  Points that do not exceed *ref* on both axes are ignored.

    Algorithm (sweepline, O(n log n)):

    1. Filter to points that dominate *ref* on both axes.
    2. Sort by first objective (x) in **descending** order.
    3. Sweep left → right tracking the running maximum of the second objective
       (y).  At each step the slice width is ``p[0] - p_next[0]`` (or
       ``p[0] - ref[0]`` for the last point); the height is
       ``current_y_max - ref[1]``.

    Parameters
    ----------
    front:
        List of (obj1, obj2) points on the Pareto front.
    ref:
        Reference (nadir) point — the hypervolume is measured relative to it.

    Returns
    -------
    float
        2-D hypervolume indicator (>= 0).
    """
    if not front:
        return 0.0

    # Keep only points that strictly dominate the reference on both axes.
    valid = [(x, y) for x, y in front if x > ref[0] and y > ref[1]]
    if not valid:
        return 0.0

    # Sort by x descending so we sweep from the rightmost to the leftmost
    # point.  The width of each slice is determined by the gap between
    # consecutive x-values; for the final (leftmost) point the right boundary
    # extends to ref[0].
    valid.sort(key=lambda p: p[0], reverse=True)

    hv = 0.0
    current_y_max = ref[1]  # running maximum of y seen so far

    for idx, (x_val, y_val) in enumerate(valid):
        # Update the running y maximum.
        current_y_max = max(current_y_max, y_val)

        # Width: distance to the next point's x, or to ref[0] for the last.
        if idx + 1 < len(valid):
            next_x = valid[idx + 1][0]
        else:
            next_x = ref[0]

        width = x_val - next_x
        if width > 0:
            hv += width * (current_y_max - ref[1])

    return max(0.0, hv)


def hypervolume_improvement_score(  # pylint: disable=unused-argument
    config: "TransformerConfig",
    current_pareto_front: list[tuple[float, float]],
    proxy_a: "MultiFidelityProxy",
    proxy_b: "ZeroCostProxy",
) -> float:
    """Compute the hypervolume improvement if *config* were added to the front.

    Uses the analytical ZK cost formula (free) and ``decoder_params_score`` as
    proxies so that no proved data is needed.

    Algorithm:

    1. Predict ``(accuracy, zk_cost)`` for this config using the proxies.
    2. Compute the hypervolume of ``current_pareto_front``.
    3. Compute the hypervolume of ``current_pareto_front + [(accuracy, -log_zk_cost)]``.
    4. Return ``max(0, new_hv - current_hv)``.

    Parameters
    ----------
    config:
        Transformer architecture to evaluate.
    current_pareto_front:
        List of ``(accuracy, -zk_cost)`` pairs already on the front
        (maximisation space — higher is better in both dims).
    proxy_a:
        Primary multi-fidelity proxy (used for fallback cost score if the
        analytical formula is unavailable).
    proxy_b:
        Zero-cost proxy (currently unused but kept in the signature for
        API consistency and future blending).

    Returns
    -------
    float
        Hypervolume improvement >= 0.  Zero means the config is dominated by
        the current front and adds nothing.
    """
    from experiments.operator_attribution import (  # pylint: disable=import-outside-toplevel
        compute_op_breakdown,
    )

    accuracy = _decoder_params_score_safe(config)
    try:
        zk_cost = float(compute_op_breakdown(config).total_constraints)
    except Exception:  # pylint: disable=broad-except
        logger.debug(
            "compute_op_breakdown raised for config %s; falling back to proxy_a",
            config,
            exc_info=True,
        )
        zk_cost = max(1.0, _proxy_a_score_safe(proxy_a, config))

    if not current_pareto_front:
        # No front yet — use accuracy/cost ratio as a sensible fallback.
        return accuracy / max(zk_cost, 1e-10)

    # Convert to 2-D maximisation space: (accuracy, -log_zk_cost).
    log_zk = math.log(max(zk_cost, 1.0))
    new_point = (accuracy, -log_zk)

    # Reference point: slightly below the worst observed point on each axis.
    ref_acc = min(p[0] for p in current_pareto_front) - 0.1
    ref_cost = min(p[1] for p in current_pareto_front) - 0.1

    current_hv = _compute_2d_hypervolume(current_pareto_front, (ref_acc, ref_cost))
    new_front = _update_pareto_front(current_pareto_front + [new_point])
    new_hv = _compute_2d_hypervolume(new_front, (ref_acc, ref_cost))

    return max(0.0, new_hv - current_hv)


# ---------------------------------------------------------------------------
# AcquisitionScore
# ---------------------------------------------------------------------------


@dataclass
class AcquisitionScore:
    """Scored acquisition candidate.

    All component scores are normalised to [0, 1] across the candidate batch.

    Attributes
    ----------
    config:
        The transformer architecture that was scored.
    uncertainty:
        Proxy prediction uncertainty (higher = less certain → more valuable).
    disagreement:
        Normalised absolute difference between proxy_a and proxy_b scores.
    pareto_proximity:
        Proximity to the current Pareto frontier; 1 = undominated, 0 = strongly
        dominated.
    pow2_cliff:
        EZKL power-of-two row boundary risk; 1 = exactly on a snap point.
    composite:
        Weighted combination of the four component scores.
    rank:
        1-based rank among all candidates (1 = highest priority).
    """

    config: TransformerConfig
    uncertainty: float
    disagreement: float
    pareto_proximity: float
    pow2_cliff: float
    composite: float
    rank: int = field(default=0)


# ---------------------------------------------------------------------------
# ActiveLearner
# ---------------------------------------------------------------------------


class ActiveLearner:
    """Selects which transformer configs to prove next.

    Acquisition function (higher composite score → prove this next)::

        composite = w_unc  * uncertainty
                  + w_dis  * disagreement
                  + w_par  * pareto_proximity
                  + w_cli  * pow2_cliff

    Default weights: uncertainty=0.4, disagreement=0.3, pareto=0.2, cliff=0.1.

    The ``disagreement`` signal is the key novel contribution: where proxy_a
    (multi-fidelity GBM, which can capture non-additive EZKL cost effects such
    as power-of-two row snapping) and proxy_b (zero-cost, purely additive)
    disagree most is exactly where proving yields the highest information gain
    about non-additive cost phenomena.

    Parameters
    ----------
    proxy_a:
        Primary learned proxy (Approach A).  Must be a
        :class:`~proxy.strategies.multifidelity.MultiFidelityProxy`.
    proxy_b:
        Baseline zero-cost proxy (Approach B).  Must be a
        :class:`~proxy.strategies.zero_cost.ZeroCostProxy`.
    proxy_c:
        Optional kernel decomposition proxy (Approach C).  When supplied its
        ``uncertainty`` output is blended with proxy_a's uncertainty.
    weights:
        Override acquisition weights.  Accepted keys: ``'uncertainty'``,
        ``'disagreement'``, ``'pareto'``, ``'cliff'``.  Missing keys take the
        default value; keys are renormalised to sum to 1.
    backend:
        ZK backend name stored in acquisition metadata (default ``'ezkl'``).
    use_ehvi:
        When ``True`` (the default), use Expected Hypervolume Improvement as
        the primary acquisition score instead of the heuristic weighted
        composite.  The composite score is still computed and stored on each
        :class:`AcquisitionScore` for diagnostics, but ranking is driven by
        the EHVI value, which is stored in ``composite``.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        proxy_a: MultiFidelityProxy,
        proxy_b: ZeroCostProxy,
        proxy_c: Optional[KernelDecompositionProxy] = None,
        weights: Optional[dict[str, float]] = None,
        backend: str = "ezkl",
        use_ehvi: bool = True,
    ) -> None:
        self._proxy_a = proxy_a
        self._proxy_b = proxy_b
        self._proxy_c = proxy_c
        self._backend = backend
        self._use_ehvi = use_ehvi
        self._weights: dict[str, float] = dict(_DEFAULT_WEIGHTS)
        if weights is not None:
            self._weights.update(weights)
        self._normalise_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_candidates(  # pylint: disable=too-many-locals
        self,
        candidates: list[TransformerConfig],
        dataset: ProxyDataset,
    ) -> list[AcquisitionScore]:
        """Score all candidates and return sorted by composite score (descending).

        When ``use_ehvi=True``, the ranking (and ``composite`` field) is driven
        by Expected Hypervolume Improvement; the heuristic component scores are
        still computed and stored for diagnostics.

        Parameters
        ----------
        candidates:
            Transformer configs to evaluate.
        dataset:
            Existing proved dataset used to build the Pareto frontier and to
            provide context for uncertainty estimates.

        Returns
        -------
        list[AcquisitionScore]
            Sorted list (index 0 = highest priority); ``rank`` fields are
            filled in (1-based).
        """
        if not candidates:
            return []

        raw_uncertainty = self._compute_raw_uncertainty(candidates)
        raw_disagreement = self._compute_raw_disagreement(candidates)
        raw_pareto = self._compute_raw_pareto(candidates, dataset)
        raw_cliff = self._compute_raw_cliff(candidates)

        # Normalise each signal to [0, 1] across the candidate set.
        unc_norm = _minmax_normalise(raw_uncertainty)
        dis_norm = _minmax_normalise(raw_disagreement)
        par_norm = _minmax_normalise(raw_pareto)
        cli_norm = _minmax_normalise(raw_cliff)

        w_unc = self._weights["uncertainty"]
        w_dis = self._weights["disagreement"]
        w_par = self._weights["pareto"]
        w_cli = self._weights["cliff"]

        # EHVI: build the current Pareto front from the dataset once.
        pareto_front: list[tuple[float, float]] = []
        if self._use_ehvi:
            pareto_front = self._build_pareto_front(dataset)

        scores: list[AcquisitionScore] = []
        for i, config in enumerate(candidates):
            heuristic_composite = (
                w_unc * unc_norm[i]
                + w_dis * dis_norm[i]
                + w_par * par_norm[i]
                + w_cli * cli_norm[i]
            )

            if self._use_ehvi:
                ehvi = hypervolume_improvement_score(
                    config, pareto_front, self._proxy_a, self._proxy_b
                )
                composite = ehvi
            else:
                composite = heuristic_composite

            scores.append(
                AcquisitionScore(
                    config=config,
                    uncertainty=float(unc_norm[i]),
                    disagreement=float(dis_norm[i]),
                    pareto_proximity=float(par_norm[i]),
                    pow2_cliff=float(cli_norm[i]),
                    composite=float(composite),
                )
            )

        # Sort descending by composite score then assign ranks.
        scores.sort(key=lambda s: s.composite, reverse=True)
        for rank_idx, score in enumerate(scores):
            score.rank = rank_idx + 1

        return scores

    def select_batch(  # pylint: disable=too-many-locals
        self,
        candidates: list[TransformerConfig],
        dataset: ProxyDataset,
        batch_size: int = 8,
        diversity_penalty: float = 0.1,
    ) -> list[TransformerConfig]:
        """Select *batch_size* candidates for proving using diversity-penalised greedy selection.

        Algorithm
        ---------
        1. Score all candidates with :meth:`score_candidates`.
        2. Select the candidate with the highest composite score.
        3. Penalise remaining candidates by their cosine similarity to the
           already-selected set: effective score = composite −
           diversity_penalty * max_cosine_similarity_to_selected.
        4. Repeat until *batch_size* candidates are selected.

        This prevents selecting 8 nearly-identical architecture configs.

        Parameters
        ----------
        candidates:
            Pool of configs to choose from.
        dataset:
            Existing proved dataset (passed through to :meth:`score_candidates`).
        batch_size:
            Number of configs to return.  Capped at ``len(candidates)``.
        diversity_penalty:
            Weight of the cosine similarity penalty in [0, 1].  0 = no
            diversity enforcement; 1 = penalise similarity as heavily as the
            composite score itself.

        Returns
        -------
        list[TransformerConfig]
            Selected configs in order of selection (first = highest priority).
        """
        if not candidates:
            return []

        effective_batch = min(batch_size, len(candidates))

        # Compute initial scores once.
        acq_scores = self.score_candidates(candidates, dataset)
        # Build a mapping from config index in *candidates* to acquisition score.
        # Use a stable index from the original candidates list.
        config_to_original_idx: dict[int, int] = {
            id(s.config): i for i, s in enumerate(acq_scores)
        }

        # Feature vectors for diversity penalty computation.
        feature_vecs = np.stack(
            [ProxyFeatures.from_config(c).to_vector() for c in candidates],
            axis=0,
        )  # shape (n_candidates, n_features)

        # Map from original candidate index to acquisition score.
        original_idx_to_score: dict[int, AcquisitionScore] = {}
        for score in acq_scores:
            orig_idx = config_to_original_idx[id(score.config)]
            original_idx_to_score[orig_idx] = score

        remaining_indices = list(range(len(candidates)))
        selected_indices: list[int] = []

        for _ in range(effective_batch):
            if not remaining_indices:
                break

            # Compute effective composite scores with diversity penalty.
            best_idx: Optional[int] = None
            best_effective: float = float("-inf")

            for cand_idx in remaining_indices:
                base_composite = original_idx_to_score[cand_idx].composite
                if selected_indices:
                    penalty = diversity_penalty * _max_cosine_similarity(
                        feature_vecs[cand_idx], feature_vecs[selected_indices]
                    )
                else:
                    penalty = 0.0
                effective = base_composite - penalty
                if effective > best_effective:
                    best_effective = effective
                    best_idx = cand_idx

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    def update_weights(self, feedback: dict[str, bool]) -> None:
        """Update acquisition weights based on proving results.

        Applies a multiplicative +10% boost to signals flagged as useful.

        Parameters
        ----------
        feedback:
            Mapping from signal name to usefulness flag.  Recognised keys:
            ``'uncertainty_was_useful'``, ``'disagreement_was_useful'``,
            ``'pareto_was_useful'``, ``'cliff_was_useful'``.

        Examples
        --------
        >>> learner.update_weights({'uncertainty_was_useful': True,
        ...                         'disagreement_was_useful': False})
        """
        key_map = {
            "uncertainty_was_useful": "uncertainty",
            "disagreement_was_useful": "disagreement",
            "pareto_was_useful": "pareto",
            "cliff_was_useful": "cliff",
        }
        for feedback_key, weight_key in key_map.items():
            if feedback.get(feedback_key):
                self._weights[weight_key] *= 1.1
        self._normalise_weights()

    # ------------------------------------------------------------------
    # Pareto front construction
    # ------------------------------------------------------------------

    def _build_pareto_front(
        self, dataset: ProxyDataset
    ) -> list[tuple[float, float]]:
        """Build the current Pareto front from labelled records in *dataset*.

        Each record contributes a point in 2-D maximisation space:
        ``(accuracy_proxy, -log_zk_cost)``, where higher is better in both
        dimensions.  The returned list contains only the non-dominated points.

        Records without a numeric label are silently skipped.

        Parameters
        ----------
        dataset:
            Proved dataset to scan for labelled records.

        Returns
        -------
        list[tuple[float, float]]
            Non-dominated points as ``(accuracy, -log_zk_cost)`` pairs.
            Empty if fewer than :data:`_MIN_PARETO_RECORDS` labelled records
            exist.
        """
        points: list[tuple[float, float]] = []
        for record in dataset._records:  # pylint: disable=protected-access
            label = record.best_label()  # log(cost) — lower is better
            if label is None:
                continue
            accuracy = _decoder_params_score_safe(record.config)
            # Transform to maximisation space: negate log_zk_cost.
            points.append((accuracy, -label))

        if len(points) < _MIN_PARETO_RECORDS:
            return []

        return _update_pareto_front(points)

    # ------------------------------------------------------------------
    # Pareto proximity
    # ------------------------------------------------------------------

    def _pareto_proximity(
        self,
        config: TransformerConfig,
        dataset: ProxyDataset,
    ) -> float:
        """Score proximity to the current Pareto frontier in (cost, performance) space.

        Since models are untrained during NAS, we use:
        - ``decoder_params_score`` (log of non-embedding parameter count) as the
          primary accuracy proxy.  This is the best validated zero-cost accuracy
          proxy for GPT-2-class transformers on WikiText-103 (Spearman 0.967).
          ``_near_score_safe`` is kept as a module-level fallback for backwards
          compatibility and ablation use only.
        - ``proxy_a.score(config)`` as the cost proxy (lower = cheaper).

        Pareto proximity = degree to which the config is *not* dominated by any
        record in *dataset*.

        An undominated config (not dominated by any existing record) receives
        proximity 1.0.  A config dominated by k records receives a decaying
        score: ``exp(-k / n_records)`` where ``n_records`` is the number of
        labelled records.

        Parameters
        ----------
        config:
            Architecture to evaluate.
        dataset:
            Existing dataset containing reference points for the frontier.

        Returns
        -------
        float
            Pareto proximity in [0, 1].
        """
        if len(dataset) < _MIN_PARETO_RECORDS:
            return 0.5

        perf_proxy = _decoder_params_score_safe(config)
        cost_proxy = _proxy_a_score_safe(self._proxy_a, config)
        reference_points = _collect_reference_points(dataset)

        if not reference_points:
            return 0.5

        domination_count = _count_dominating(cost_proxy, perf_proxy, reference_points)
        proximity = math.exp(-domination_count / max(len(reference_points), 1))
        return float(proximity)

    # ------------------------------------------------------------------
    # Private helpers: raw signal computation
    # ------------------------------------------------------------------

    def _compute_raw_uncertainty(
        self, candidates: list[TransformerConfig]
    ) -> list[float]:
        """Return raw uncertainty scores for each candidate."""
        raw: list[float] = []
        for config in candidates:
            unc = _uncertainty_safe(self._proxy_a, config)
            if self._proxy_c is not None:
                unc_c = _uncertainty_safe(self._proxy_c, config)
                # Blend proxy_a and proxy_c uncertainties (mean).
                if math.isfinite(unc_c):
                    unc = (unc + unc_c) / 2.0 if math.isfinite(unc) else unc_c
            raw.append(unc if math.isfinite(unc) else 0.0)
        return raw

    def _compute_raw_disagreement(
        self, candidates: list[TransformerConfig]
    ) -> list[float]:
        """Return raw |score_a - score_b| disagreement for each candidate."""
        raw: list[float] = []
        for config in candidates:
            score_a = _score_safe(self._proxy_a, config)
            score_b = _score_safe(self._proxy_b, config)
            raw.append(abs(score_a - score_b))
        return raw

    def _compute_raw_pareto(
        self,
        candidates: list[TransformerConfig],
        dataset: ProxyDataset,
    ) -> list[float]:
        """Return raw Pareto proximity scores for each candidate."""
        return [self._pareto_proximity(config, dataset) for config in candidates]

    def _compute_raw_cliff(self, candidates: list[TransformerConfig]) -> list[float]:
        """Return raw pow2 cliff scores for each candidate.

        ``pow2_cliff_score`` returns ``nan`` when the proxy is unfitted or is
        operating in fallback (few-sample) mode, so no direct access to the
        private ``_fallback_mean`` attribute is needed.
        """
        raw: list[float] = []
        for config in candidates:
            cliff = self._proxy_a.pow2_cliff_score(config)
            raw.append(cliff if math.isfinite(cliff) else 0.0)
        return raw

    # ------------------------------------------------------------------
    # Private helpers: weight management
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        """Normalise weights so they sum to 1.0.  Guards against zero-sum."""
        total = sum(self._weights.values())
        if total <= 0.0:
            self._weights = dict(_DEFAULT_WEIGHTS)
            return
        self._weights = {k: v / total for k, v in self._weights.items()}

    def __repr__(self) -> str:
        fitted_a = "fitted" if self._proxy_a.is_fitted() else "unfitted"
        fitted_b = "fitted" if self._proxy_b.is_fitted() else "unfitted"
        return (
            f"ActiveLearner("
            f"proxy_a={fitted_a}, "
            f"proxy_b={fitted_b}, "
            f"backend={self._backend!r}, "
            f"use_ehvi={self._use_ehvi}, "
            f"weights={self._weights})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _collect_reference_points(dataset: ProxyDataset) -> list[tuple[float, float]]:
    """Extract (log_cost, perf_proxy) pairs from labelled records in *dataset*.

    Records without a numeric label are silently skipped.

    Parameters
    ----------
    dataset:
        Dataset to scan for labelled records.

    Returns
    -------
    list[tuple[float, float]]
        Ordered list of (log_cost, performance_proxy) tuples.
    """
    points: list[tuple[float, float]] = []
    for record in dataset._records:  # pylint: disable=protected-access
        label = record.best_label()
        if label is None:
            continue
        ref_perf = _decoder_params_score_safe(record.config)
        points.append((label, ref_perf))
    return points


def _count_dominating(
    cost: float,
    perf: float,
    reference_points: list[tuple[float, float]],
) -> int:
    """Return the number of reference points that dominate *(cost, perf)*.

    A reference point *(rc, rp)* dominates *(cost, perf)* when:
    - ``rc <= cost`` and ``rp >= perf`` (at least as cheap and as performant)
    - at least one inequality is strict (not identical)

    Parameters
    ----------
    cost:
        Log-cost of the candidate config.
    perf:
        Performance proxy of the candidate config.
    reference_points:
        List of (log_cost, perf_proxy) tuples from the existing dataset.

    Returns
    -------
    int
        Number of reference points that dominate the candidate.
    """
    count = 0
    for ref_cost, ref_perf in reference_points:
        cheaper_or_equal = ref_cost <= cost
        better_or_equal = ref_perf >= perf
        strictly_better = ref_cost < cost or ref_perf > perf
        if cheaper_or_equal and better_or_equal and strictly_better:
            count += 1
    return count


def _minmax_normalise(values: list[float]) -> np.ndarray:
    """Min-max normalise *values* to [0, 1] using sklearn's MinMaxScaler.

    When all values are identical the result is a uniform 0.5 array to
    avoid collapsing all diversity information.

    Parameters
    ----------
    values:
        Raw signal values (one per candidate).

    Returns
    -------
    np.ndarray
        Normalised values in [0, 1], shape ``(len(values),)``.
    """
    arr = np.array(values, dtype=np.float64).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    try:
        normed = scaler.fit_transform(arr).ravel()
    except ValueError:
        # All values identical — MinMaxScaler raises ValueError.
        normed = np.full(len(values), 0.5, dtype=np.float64)
    # If scaler produced a constant (range == 0), replace with 0.5.
    if np.allclose(normed, normed[0]):
        normed = np.full(len(values), 0.5, dtype=np.float64)
    return normed


def _max_cosine_similarity(
    vec: np.ndarray, selected_vecs: Union[list[int], np.ndarray]
) -> float:
    """Return the maximum cosine similarity between *vec* and any row in *selected_vecs*.

    Parameters
    ----------
    vec:
        1-D feature vector for the candidate.
    selected_vecs:
        Either a 2-D array of selected feature vectors, or a list of integer
        indices (when called from the batch loop, the caller already indexes
        into the full feature matrix).

    Returns
    -------
    float
        Max cosine similarity in [0, 1] (after clipping negative values).
    """
    if isinstance(selected_vecs, list):
        # selected_vecs is a list of raw row vectors; we expect np.ndarray rows.
        rows = np.stack(selected_vecs, axis=0)
    else:
        rows = selected_vecs  # 2-D array

    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    norm_vec = np.linalg.norm(vec)
    if norm_vec < _COSINE_EPS:
        return 0.0

    norms_rows = np.linalg.norm(rows, axis=1, keepdims=True)
    norms_rows = np.where(norms_rows < _COSINE_EPS, _COSINE_EPS, norms_rows)
    sims = (rows / norms_rows) @ (vec / norm_vec)
    return float(np.clip(sims.max(), 0.0, 1.0))


def _score_safe(
    proxy: Union[MultiFidelityProxy, ZeroCostProxy], config: TransformerConfig
) -> float:
    """Call ``proxy.score(config)`` safely, returning 0.0 on any failure."""
    try:
        val = proxy.score(config)
        return float(val) if math.isfinite(val) else 0.0
    except Exception:  # pylint: disable=broad-except
        logger.debug("score() raised for config %s", config, exc_info=True)
        return 0.0


def _uncertainty_safe(
    proxy: Union[MultiFidelityProxy, KernelDecompositionProxy],
    config: TransformerConfig,
) -> float:
    """Call ``proxy.uncertainty(config)`` safely, returning NaN on failure."""
    try:
        val = proxy.uncertainty(config)
        return float(val)
    except Exception:  # pylint: disable=broad-except
        logger.debug("uncertainty() raised for config %s", config, exc_info=True)
        return float("nan")


def _decoder_params_score_safe(config: TransformerConfig) -> float:
    """Return ``decoder_params_score`` safely, falling back to ``log(total_params)``.

    Primary accuracy proxy for GPT-2-class NAS (Spearman 0.967 on WikiText-103).
    No model or data required.

    Parameters
    ----------
    config:
        Architecture to evaluate.

    Returns
    -------
    float
        log(decoder_params) score (higher = more performant predicted).
    """
    try:
        return decoder_params_score(config)
    except Exception:  # pylint: disable=broad-except
        logger.debug(
            "decoder_params_score raised for config %s", config, exc_info=True
        )
        features = ProxyFeatures.from_config(config)
        total_params = features.total_params
        if total_params > 0:
            return math.log(total_params)
        return 0.0


def _near_score_safe(config: TransformerConfig) -> float:
    """Return a NEAR-style performance proxy for *config*.

    **Retained for backwards-compatibility and ablation studies only.**
    NEAR is not validated for GPT-2/WikiText ranking; use
    :func:`_decoder_params_score_safe` or :func:`decoder_params_score` for the
    primary accuracy axis.

    Attempts to import ``zkml_bench.nas.zero_cost_proxies.near_score``.
    Falls back to ``log(total_params)`` when that module is unavailable.

    Parameters
    ----------
    config:
        Architecture to evaluate.

    Returns
    -------
    float
        Proxy accuracy score (higher = more performant).
    """
    try:
        from zkml_bench.nas.zero_cost_proxies import (  # pylint: disable=import-outside-toplevel
            near_score,
        )

        return float(near_score(config))
    except (ImportError, Exception):  # pylint: disable=broad-except
        features = ProxyFeatures.from_config(config)
        total_params = features.total_params
        if total_params > 0:
            return math.log(total_params)
        return 0.0


def _proxy_a_score_safe(
    proxy_a: MultiFidelityProxy, config: TransformerConfig
) -> float:
    """Return proxy_a cost score safely, falling back to log(total_macs)."""
    if proxy_a.is_fitted():
        try:
            val = proxy_a.score(config)
            if math.isfinite(val):
                return val
        except Exception:  # pylint: disable=broad-except
            pass
    features = ProxyFeatures.from_config(config)
    macs = features.total_macs
    return math.log(macs) if macs > 0 else 0.0

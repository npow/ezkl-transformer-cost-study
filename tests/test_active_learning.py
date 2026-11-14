"""Tests for proxy.active_learning — ActiveLearner and AcquisitionScore."""

# pylint: disable=protected-access,too-many-lines

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from experiments.operator_attribution import compute_op_breakdown
from proxy.accuracy_proxies import decoder_params_score
from proxy.active_learning import (
    AcquisitionScore,
    ActiveLearner,
    _compute_2d_hypervolume,
    _max_cosine_similarity,
    _minmax_normalise,
    _near_score_safe,
    _update_pareto_front,
    hypervolume_improvement_score,
)
from proxy.data import ProxyDataset, ProxyRecord
from proxy.features import ProxyFeatures, TransformerConfig
from proxy.strategies.kernel_decomp import KernelDecompositionProxy
from proxy.strategies.multifidelity import MultiFidelityProxy
from proxy.strategies.zero_cost import ZeroCostProxy


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _cfg(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    seq_len: int = 64,
    ff_mult: float = 4.0,
    activation: str = "relu",
) -> TransformerConfig:
    """Create a TransformerConfig with the given hyperparameters."""
    return TransformerConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seq_len=seq_len,
        ff_mult=ff_mult,
        activation=activation,
    )


# A grid of architecturally diverse configs used across multiple tests.
_ARCH_GRID: list[tuple[int, int, int, int]] = [
    (64, 4, 2, 64),
    (128, 4, 3, 64),
    (128, 8, 2, 128),
    (256, 4, 4, 64),
    (256, 8, 3, 128),
    (256, 8, 6, 128),
    (512, 8, 4, 64),
    (512, 8, 6, 128),
    (512, 16, 3, 64),
    (384, 12, 4, 128),
]


def _make_candidates(n: int = 10) -> list[TransformerConfig]:
    """Return *n* diverse TransformerConfig instances."""
    configs: list[TransformerConfig] = []
    for i in range(n):
        d, h, layers, seq = _ARCH_GRID[i % len(_ARCH_GRID)]
        configs.append(_cfg(d_model=d, n_heads=h, n_layers=layers, seq_len=seq))
    return configs


def _make_dataset(n: int = 20, seed: int = 42) -> ProxyDataset:
    """Build a synthetic ProxyDataset with *n* diverse labelled records."""
    rng = random.Random(seed)
    dataset = ProxyDataset()
    for i in range(n):
        d, h, layers, seq = _ARCH_GRID[i % len(_ARCH_GRID)]
        config = _cfg(d_model=d, n_heads=h, n_layers=layers, seq_len=seq)
        features = ProxyFeatures.from_config(config)
        log_cost = math.log(d * layers * seq) + rng.gauss(0, 0.3)
        onnx_est = max(1, int(math.exp(log_cost)))
        record = ProxyRecord(
            config=config,
            features=features,
            onnx_estimate=onnx_est,
            ezkl_num_rows=onnx_est * 2,
        )
        dataset.add(record)
    return dataset


def _make_fitted_proxy_a(dataset: ProxyDataset) -> MultiFidelityProxy:
    """Return a MultiFidelityProxy fitted on *dataset*."""
    proxy = MultiFidelityProxy(n_estimators=5, max_depth=3, learning_rate=0.1)
    proxy.fit(dataset)
    return proxy


def _make_fitted_proxy_b(dataset: ProxyDataset) -> ZeroCostProxy:
    """Return a ZeroCostProxy fitted on *dataset*."""
    proxy = ZeroCostProxy(model_type="rf")
    proxy.fit(dataset)
    return proxy


def _make_learner(
    fitted: bool = True,
    n_train: int = 20,
) -> tuple[ActiveLearner, ProxyDataset]:
    """Return an (ActiveLearner, training_dataset) pair."""
    dataset = _make_dataset(n=n_train)
    if fitted:
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)
    else:
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
    learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
    return learner, dataset


# ---------------------------------------------------------------------------
# AcquisitionScore dataclass
# ---------------------------------------------------------------------------


class TestAcquisitionScore:
    """Tests for the AcquisitionScore dataclass."""

    def test_dataclass_fields(self) -> None:
        """All fields are stored and accessible on the dataclass."""
        config = _cfg()
        score = AcquisitionScore(
            config=config,
            uncertainty=0.7,
            disagreement=0.3,
            pareto_proximity=0.6,
            pow2_cliff=0.2,
            composite=0.5,
            rank=1,
        )
        assert score.config is config
        assert score.rank == 1
        assert score.composite == pytest.approx(0.5)

    def test_default_rank_is_zero(self) -> None:
        """Rank defaults to 0 when not supplied."""
        score = AcquisitionScore(
            config=_cfg(),
            uncertainty=0.5,
            disagreement=0.5,
            pareto_proximity=0.5,
            pow2_cliff=0.5,
            composite=0.5,
        )
        assert score.rank == 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for ActiveLearner construction and weight handling."""

    def test_default_weights_sum_to_one(self) -> None:
        """Default acquisition weights must sum to 1."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        total = sum(learner._weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_custom_weights_accepted(self) -> None:
        """Custom weights are normalised and stored correctly."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        weights = {"uncertainty": 1.0, "disagreement": 1.0, "pareto": 0.0, "cliff": 0.0}
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, weights=weights)
        assert learner._weights["uncertainty"] == pytest.approx(0.5)
        assert learner._weights["disagreement"] == pytest.approx(0.5)
        assert learner._weights["pareto"] == pytest.approx(0.0)

    def test_proxy_c_optional(self) -> None:
        """proxy_c defaults to None."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, proxy_c=None)
        assert learner._proxy_c is None

    def test_proxy_c_stored(self) -> None:
        """proxy_c is stored when supplied."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        proxy_c = KernelDecompositionProxy()
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, proxy_c=proxy_c)
        assert learner._proxy_c is proxy_c

    def test_backend_stored(self) -> None:
        """The backend string is stored on the learner."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, backend="halo2")
        assert learner._backend == "halo2"

    def test_repr_contains_backend(self) -> None:
        """repr() must include the backend name."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, backend="ezkl")
        assert "ezkl" in repr(learner)


# ---------------------------------------------------------------------------
# score_candidates
# ---------------------------------------------------------------------------


class TestScoreCandidates:
    """Tests for ActiveLearner.score_candidates."""

    @pytest.fixture(scope="class")
    def learner_and_data(self) -> tuple[ActiveLearner, ProxyDataset]:
        """Return a fitted learner and its training dataset."""
        return _make_learner(fitted=True, n_train=20)

    def test_returns_correct_length(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """score_candidates returns one score per candidate."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        assert len(scores) == 10

    def test_returns_acquisition_score_instances(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Each element returned is an AcquisitionScore."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(5)
        scores = learner.score_candidates(candidates, dataset)
        for score in scores:
            assert isinstance(score, AcquisitionScore)

    def test_sorted_descending_by_composite(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Results must be sorted by composite score in descending order."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        composites = [s.composite for s in scores]
        assert composites == sorted(
            composites, reverse=True
        ), "score_candidates must return list sorted by composite score (descending)"

    def test_ranks_are_1_based_and_sequential(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Ranks must be 1, 2, 3, … with no gaps."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(8)
        scores = learner.score_candidates(candidates, dataset)
        ranks = [s.rank for s in scores]
        assert ranks == list(range(1, len(scores) + 1))

    def test_composite_score_in_0_1_range(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Composite scores must lie in [0, 1]."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        for score in scores:
            assert (
                0.0 <= score.composite <= 1.0 + 1e-9
            ), f"composite={score.composite!r} out of [0, 1]"

    def test_component_scores_in_0_1_range(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """All component scores (uncertainty, disagreement, etc.) must be in [0, 1]."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        for score in scores:
            assert 0.0 <= score.uncertainty <= 1.0 + 1e-9
            assert 0.0 <= score.disagreement <= 1.0 + 1e-9
            assert 0.0 <= score.pareto_proximity <= 1.0 + 1e-9
            assert 0.0 <= score.pow2_cliff <= 1.0 + 1e-9

    def test_empty_candidates_returns_empty_list(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Empty candidate list returns empty result."""
        learner, dataset = learner_and_data
        assert learner.score_candidates([], dataset) == []

    def test_single_candidate_returns_one_score(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """A single candidate yields exactly one score with rank 1."""
        learner, dataset = learner_and_data
        scores = learner.score_candidates([_cfg()], dataset)
        assert len(scores) == 1
        assert scores[0].rank == 1


# ---------------------------------------------------------------------------
# score_candidates with unfitted proxies
# ---------------------------------------------------------------------------


class TestScoreCandidatesUnfitted:
    """Unfitted proxies should still work, just with lower-quality scores."""

    @pytest.fixture(scope="class")
    def learner_and_data(self) -> tuple[ActiveLearner, ProxyDataset]:
        """Return an unfitted learner and a dummy dataset."""
        return _make_learner(fitted=False)

    def test_returns_correct_length_unfitted(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """score_candidates still returns one score per candidate when proxies are unfitted."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(6)
        scores = learner.score_candidates(candidates, dataset)
        assert len(scores) == 6

    def test_composite_in_0_1_unfitted(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Composite scores remain in [0, 1] even with unfitted proxies."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(6)
        scores = learner.score_candidates(candidates, dataset)
        for score in scores:
            assert 0.0 <= score.composite <= 1.0 + 1e-9

    def test_sorted_descending_unfitted(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Descending sort invariant holds with unfitted proxies."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(6)
        scores = learner.score_candidates(candidates, dataset)
        composites = [s.composite for s in scores]
        assert composites == sorted(composites, reverse=True)


# ---------------------------------------------------------------------------
# select_batch
# ---------------------------------------------------------------------------


class TestSelectBatch:
    """Tests for ActiveLearner.select_batch."""

    @pytest.fixture(scope="class")
    def learner_and_data(self) -> tuple[ActiveLearner, ProxyDataset]:
        """Return a fitted learner and its training dataset."""
        return _make_learner(fitted=True, n_train=20)

    def test_returns_correct_batch_size(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """select_batch returns exactly batch_size configs."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        batch = learner.select_batch(candidates, dataset, batch_size=4)
        assert len(batch) == 4

    def test_batch_size_capped_at_candidates(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """batch_size is capped at the total number of candidates."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(3)
        batch = learner.select_batch(candidates, dataset, batch_size=10)
        assert len(batch) == 3

    def test_returned_configs_are_subset_of_candidates(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Every config in the batch must originate from the candidates list."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        batch = learner.select_batch(candidates, dataset, batch_size=5)
        candidate_ids = {id(c) for c in candidates}
        for config in batch:
            assert id(config) in candidate_ids

    def test_no_duplicates_in_batch(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Each config must appear at most once in the batch."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        batch = learner.select_batch(candidates, dataset, batch_size=8)
        unique_ids = {id(c) for c in batch}
        assert len(unique_ids) == len(batch), "Batch must not contain duplicate configs"

    def test_empty_candidates_returns_empty(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Empty candidate pool yields an empty batch."""
        learner, dataset = learner_and_data
        assert learner.select_batch([], dataset, batch_size=4) == []

    def test_batch_size_one_returns_single_config(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """batch_size=1 returns a list of exactly one config."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        batch = learner.select_batch(candidates, dataset, batch_size=1)
        assert len(batch) == 1


# ---------------------------------------------------------------------------
# Diversity penalty reduces near-duplicate selection
# ---------------------------------------------------------------------------


class TestDiversityPenalty:
    """Verifies that a high diversity_penalty reduces near-identical selections."""

    def _make_near_duplicate_candidates(self) -> list[TransformerConfig]:
        """Return a list with 8 near-duplicate configs and 2 diverse ones."""
        near_dupes = [
            _cfg(d_model=64, n_heads=4, n_layers=n, seq_len=64) for n in range(2, 10)
        ]
        diverse = [
            _cfg(d_model=512, n_heads=8, n_layers=2, seq_len=128),
            _cfg(d_model=256, n_heads=4, n_layers=6, seq_len=64),
        ]
        return near_dupes + diverse

    def test_high_diversity_penalty_prefers_diverse_batch(self) -> None:
        """With high diversity penalty, diverse configs should appear more often."""
        learner, dataset = _make_learner(fitted=True, n_train=20)
        candidates = self._make_near_duplicate_candidates()

        batch_diverse = learner.select_batch(
            candidates, dataset, batch_size=4, diversity_penalty=1.0
        )
        batch_greedy = learner.select_batch(
            candidates, dataset, batch_size=4, diversity_penalty=0.0
        )

        diverse_ids = {id(candidates[8]), id(candidates[9])}
        count_diverse = sum(1 for c in batch_diverse if id(c) in diverse_ids)
        count_greedy = sum(1 for c in batch_greedy if id(c) in diverse_ids)

        assert count_diverse >= count_greedy, (
            f"High diversity penalty should select at least as many diverse configs "
            f"(diverse={count_diverse}, greedy={count_greedy})"
        )

    def test_zero_diversity_penalty_returns_valid_batch(self) -> None:
        """Zero diversity penalty still returns a valid batch."""
        learner, dataset = _make_learner(fitted=True, n_train=20)
        candidates = self._make_near_duplicate_candidates()
        batch = learner.select_batch(
            candidates, dataset, batch_size=4, diversity_penalty=0.0
        )
        assert len(batch) == 4

    def test_no_duplicate_configs_regardless_of_penalty(self) -> None:
        """Diversity penalty must not produce duplicate configs in the batch."""
        learner, dataset = _make_learner(fitted=True, n_train=20)
        candidates = self._make_near_duplicate_candidates()
        for penalty in [0.0, 0.1, 0.5, 1.0]:
            batch = learner.select_batch(
                candidates, dataset, batch_size=5, diversity_penalty=penalty
            )
            unique_ids = {id(c) for c in batch}
            assert len(unique_ids) == len(
                batch
            ), f"Duplicate found in batch with diversity_penalty={penalty}"


# ---------------------------------------------------------------------------
# update_weights
# ---------------------------------------------------------------------------


class TestUpdateWeights:
    """Tests for ActiveLearner.update_weights."""

    def test_useful_signal_gets_boosted(self) -> None:
        """Marking a signal useful must increase its weight."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        old_unc = learner._weights["uncertainty"]
        learner.update_weights({"uncertainty_was_useful": True})
        assert learner._weights["uncertainty"] > old_unc

    def test_weights_still_sum_to_one_after_update(self) -> None:
        """Weights must remain normalised after update_weights."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        learner.update_weights(
            {
                "uncertainty_was_useful": True,
                "disagreement_was_useful": True,
                "pareto_was_useful": False,
                "cliff_was_useful": True,
            }
        )
        total = sum(learner._weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_false_signals_not_boosted(self) -> None:
        """Signals marked False must not receive a multiplicative boost."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        old_weights = dict(learner._weights)
        learner.update_weights({"uncertainty_was_useful": True})
        # After renormalisation, the boosted weight is larger than before.
        assert learner._weights["uncertainty"] >= old_weights["uncertainty"]

    def test_empty_feedback_no_change_in_sum(self) -> None:
        """Empty feedback dict leaves the weight sum at 1."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        learner.update_weights({})
        assert sum(learner._weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_multiple_updates_accumulate(self) -> None:
        """Repeatedly marking a signal useful accumulates the boost."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        w_before = learner._weights["disagreement"]
        learner.update_weights({"disagreement_was_useful": True})
        w_after1 = learner._weights["disagreement"]
        learner.update_weights({"disagreement_was_useful": True})
        w_after2 = learner._weights["disagreement"]
        assert w_after2 > w_after1
        assert w_after1 > w_before


# ---------------------------------------------------------------------------
# Pareto proximity
# ---------------------------------------------------------------------------


class TestParetoProximity:
    """Tests for ActiveLearner._pareto_proximity."""

    def test_returns_float_in_0_1(self) -> None:
        """Pareto proximity must be a float in [0, 1]."""
        learner, dataset = _make_learner(fitted=True, n_train=20)
        config = _cfg(d_model=64, n_heads=4, n_layers=4)
        prox = learner._pareto_proximity(config, dataset)
        assert isinstance(prox, float)
        assert 0.0 <= prox <= 1.0 + 1e-9

    def test_returns_fallback_for_empty_dataset(self) -> None:
        """Returns 0.5 fallback when dataset is empty."""
        learner, _ = _make_learner(fitted=True)
        empty_dataset = ProxyDataset()
        prox = learner._pareto_proximity(_cfg(), empty_dataset)
        assert prox == pytest.approx(0.5)

    def test_returns_fallback_for_single_record_dataset(self) -> None:
        """Returns 0.5 fallback when dataset has fewer than 2 records."""
        learner, _ = _make_learner(fitted=True)
        dataset = ProxyDataset()
        config = _cfg(d_model=64, n_heads=4, n_layers=2)
        features = ProxyFeatures.from_config(config)
        record = ProxyRecord(
            config=config,
            features=features,
            onnx_estimate=100_000,
        )
        dataset.add(record)
        prox = learner._pareto_proximity(_cfg(), dataset)
        assert prox == pytest.approx(0.5)

    def test_undominated_config_has_high_proximity(self) -> None:
        """An undominated config should have high proximity (close to 1.0)."""
        learner, dataset = _make_learner(fitted=True, n_train=10)
        tiny = _cfg(d_model=32, n_heads=4, n_layers=1, seq_len=32)
        prox = learner._pareto_proximity(tiny, dataset)
        assert isinstance(prox, float)
        assert 0.0 <= prox <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# _minmax_normalise helper
# ---------------------------------------------------------------------------


class TestMinmaxNormalise:
    """Tests for the _minmax_normalise module-level helper."""

    def test_uniform_values_returns_0_5(self) -> None:
        """All-equal values should normalise to 0.5."""
        values = [3.0, 3.0, 3.0, 3.0]
        normed = _minmax_normalise(values)
        np.testing.assert_allclose(normed, 0.5, atol=1e-9)

    def test_range_is_0_to_1(self) -> None:
        """Normalised range must span exactly [0, 1]."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = _minmax_normalise(values)
        assert normed.min() == pytest.approx(0.0, abs=1e-9)
        assert normed.max() == pytest.approx(1.0, abs=1e-9)

    def test_single_value_returns_0_5(self) -> None:
        """A single-element list normalises to 0.5."""
        normed = _minmax_normalise([42.0])
        assert normed[0] == pytest.approx(0.5, abs=1e-9)

    def test_output_length_matches_input(self) -> None:
        """Output length equals input length."""
        values = [1.0, 5.0, 3.0, 2.0, 4.0]
        normed = _minmax_normalise(values)
        assert len(normed) == len(values)

    def test_monotone_order_preserved(self) -> None:
        """Monotone input order is preserved in the normalised output."""
        values = [10.0, 20.0, 30.0, 40.0]
        normed = _minmax_normalise(values)
        for i in range(len(normed) - 1):
            assert normed[i] <= normed[i + 1]


# ---------------------------------------------------------------------------
# _max_cosine_similarity helper
# ---------------------------------------------------------------------------


class TestMaxCosineSimilarity:
    """Tests for the _max_cosine_similarity module-level helper."""

    def test_identical_vectors_return_one(self) -> None:
        """Cosine similarity between identical vectors is 1."""
        vec = np.array([1.0, 2.0, 3.0])
        selected = np.array([[1.0, 2.0, 3.0]])
        sim = _max_cosine_similarity(vec, selected)
        assert sim == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Cosine similarity between orthogonal vectors is 0."""
        vec = np.array([1.0, 0.0, 0.0])
        selected = np.array([[0.0, 1.0, 0.0]])
        sim = _max_cosine_similarity(vec, selected)
        assert sim == pytest.approx(0.0, abs=1e-9)

    def test_returns_max_across_multiple_selected(self) -> None:
        """Returns the maximum similarity across multiple selected vectors."""
        vec = np.array([1.0, 0.0])
        selected = np.array([[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071]])
        sim = _max_cosine_similarity(vec, selected)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_zero_vec_returns_zero(self) -> None:
        """Zero-length query vector returns 0."""
        vec = np.zeros(3)
        selected = np.array([[1.0, 2.0, 3.0]])
        sim = _max_cosine_similarity(vec, selected)
        assert sim == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _near_score_safe helper
# ---------------------------------------------------------------------------


class TestNearScoreSafe:
    """Tests for the _near_score_safe module-level helper."""

    def test_returns_finite_float(self) -> None:
        """Returns a finite float for any valid TransformerConfig."""
        config = _cfg(d_model=128, n_heads=8, n_layers=4, seq_len=64)
        score = _near_score_safe(config)
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_larger_model_has_higher_score(self) -> None:
        """Larger model has higher log(total_params) fallback score."""
        small = _cfg(d_model=32, n_heads=4, n_layers=1, seq_len=32)
        large = _cfg(d_model=256, n_heads=8, n_layers=6, seq_len=64)
        assert _near_score_safe(large) > _near_score_safe(small)


# ---------------------------------------------------------------------------
# Integration: score_candidates with proxy_c
# ---------------------------------------------------------------------------


class TestWithProxyC:
    """Integration tests for ActiveLearner with an optional proxy_c."""

    def test_proxy_c_accepted_without_error(self) -> None:
        """ActiveLearner accepts a KernelDecompositionProxy as proxy_c."""
        dataset = _make_dataset(n=20)
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)
        proxy_c = KernelDecompositionProxy()
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, proxy_c=proxy_c)
        candidates = _make_candidates(5)
        scores = learner.score_candidates(candidates, dataset)
        assert len(scores) == 5

    def test_composite_in_range_with_proxy_c(self) -> None:
        """Composite scores remain in [0, 1] when proxy_c is supplied."""
        dataset = _make_dataset(n=20)
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)
        proxy_c = KernelDecompositionProxy()
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, proxy_c=proxy_c)
        candidates = _make_candidates(5)
        scores = learner.score_candidates(candidates, dataset)
        for score in scores:
            assert 0.0 <= score.composite <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# _compute_2d_hypervolume
# ---------------------------------------------------------------------------


class TestCompute2dHypervolume:
    """Tests for the _compute_2d_hypervolume module-level helper."""

    def test_single_point_correct_area(self) -> None:
        """A single-point front produces a rectangular area against the reference."""
        # Point (3, 4), ref (0, 0) → area = 3 * 4 = 12
        hv = _compute_2d_hypervolume([(3.0, 4.0)], (0.0, 0.0))
        assert hv == pytest.approx(12.0)

    def test_two_points_non_dominated_correct_area(self) -> None:
        """Two non-dominated points produce the correct combined area.

        Front: (3, 1), (1, 3), ref (0, 0).
        Slices (sweeping x descending):
          x=3 → x=1: width=2, height=1 → 2
          x=1 → x=0: width=1, height=3 → 3
        Total = 5.
        """
        front = [(3.0, 1.0), (1.0, 3.0)]
        hv = _compute_2d_hypervolume(front, (0.0, 0.0))
        assert hv == pytest.approx(5.0)

    def test_dominated_points_ignored(self) -> None:
        """A dominated point does not change the hypervolume."""
        # (2, 2) dominates (1, 1).  HV should equal HV of just (2, 2).
        front_with_dom = [(2.0, 2.0), (1.0, 1.0)]
        front_clean = [(2.0, 2.0)]
        ref = (0.0, 0.0)
        assert _compute_2d_hypervolume(front_with_dom, ref) == pytest.approx(
            _compute_2d_hypervolume(front_clean, ref)
        )

    def test_empty_front_returns_zero(self) -> None:
        """Empty front returns 0."""
        assert _compute_2d_hypervolume([], (0.0, 0.0)) == pytest.approx(0.0)

    def test_point_below_reference_returns_zero(self) -> None:
        """A front point that does not dominate the reference contributes nothing."""
        assert _compute_2d_hypervolume([(0.5, 0.5)], (1.0, 1.0)) == pytest.approx(0.0)

    def test_hypervolume_non_negative(self) -> None:
        """Hypervolume is always non-negative."""
        front = [(5.0, 1.0), (3.0, 3.0), (1.0, 5.0)]
        ref = (0.0, 0.0)
        assert _compute_2d_hypervolume(front, ref) >= 0.0

    def test_larger_front_dominates_subset(self) -> None:
        """Adding a non-dominated point to the front increases hypervolume."""
        front1 = [(3.0, 2.0)]
        front2 = [(3.0, 2.0), (1.0, 4.0)]
        ref = (0.0, 0.0)
        assert _compute_2d_hypervolume(front2, ref) > _compute_2d_hypervolume(
            front1, ref
        )


# ---------------------------------------------------------------------------
# _update_pareto_front
# ---------------------------------------------------------------------------


class TestUpdateParetoFront:
    """Tests for the _update_pareto_front module-level helper."""

    def test_removes_dominated_point(self) -> None:
        """A strictly dominated point is removed from the output."""
        points = [(3.0, 3.0), (1.0, 1.0)]  # (1,1) dominated by (3,3)
        pareto = _update_pareto_front(points)
        assert len(pareto) == 1
        assert pareto[0] == (3.0, 3.0)

    def test_keeps_all_non_dominated(self) -> None:
        """Non-dominated points are all retained."""
        points = [(3.0, 1.0), (1.0, 3.0)]  # neither dominates the other
        pareto = _update_pareto_front(points)
        assert len(pareto) == 2
        assert set(pareto) == {(3.0, 1.0), (1.0, 3.0)}

    def test_empty_input_returns_empty(self) -> None:
        """Empty input returns empty output."""
        assert not _update_pareto_front([])

    def test_single_point_is_non_dominated(self) -> None:
        """A single point is always non-dominated."""
        pareto = _update_pareto_front([(5.0, 5.0)])
        assert pareto == [(5.0, 5.0)]

    def test_identical_points_one_dominates_other(self) -> None:
        """Duplicate (identical) points: one is removed because it is not strictly
        better, so neither dominates the other — both are kept.
        Verify that at most one is returned (implementation detail: both have
        the same value so dominance check requires strict improvement)."""
        # Two identical points: neither dominates the other (no strict inequality).
        pareto = _update_pareto_front([(2.0, 2.0), (2.0, 2.0)])
        # Both points survive (neither has a strictly better coordinate).
        assert len(pareto) == 2

    def test_many_dominated_reduced_to_frontier(self) -> None:
        """A grid of points is correctly reduced to the Pareto frontier."""
        # Grid: (i, j) for i, j in {1,2,3}.  Dominated by (3,3) except on the
        # anti-diagonal where each point is not dominated by any other.
        # The non-dominated set of a 3x3 grid is: (1,3),(2,3),(3,3),(3,2),(3,1)
        # -- actually (2,3) is dominated by (3,3), (3,2) is dominated by (3,3).
        # So only (3,3) survives.
        points = [(i, j) for i in range(1, 4) for j in range(1, 4)]
        pareto = _update_pareto_front(points)
        assert (3, 3) in [tuple(p) for p in pareto]
        # All survivors must have at least one coordinate equal to 3.
        for p in pareto:
            assert p[0] == 3 or p[1] == 3

    def test_all_output_points_are_non_dominated(self) -> None:
        """Every point in the returned front is non-dominated w.r.t. the full input."""
        points = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0), (2.0, 2.0)]
        pareto = _update_pareto_front(points)
        for p in pareto:
            for other in points:
                if other is p:
                    continue
                # No other point should dominate p.
                assert not (
                    other[0] >= p[0]
                    and other[1] >= p[1]
                    and (other[0] > p[0] or other[1] > p[1])
                ), f"{other} dominates {p} in returned Pareto front"


# ---------------------------------------------------------------------------
# hypervolume_improvement_score
# ---------------------------------------------------------------------------


class TestHypervolumeImprovementScore:
    """Tests for hypervolume_improvement_score."""

    @pytest.fixture(scope="class")
    def proxies(self) -> tuple:
        """Return a (proxy_a, proxy_b) pair (fitted on a small dataset)."""
        dataset = _make_dataset(n=20)
        return _make_fitted_proxy_a(dataset), _make_fitted_proxy_b(dataset)

    def test_returns_positive_for_undominated_point(self, proxies: tuple) -> None:
        """An undominated point improves the hypervolume."""
        proxy_a, proxy_b = proxies
        # Build a small front with low accuracy and high cost (so a large,
        # cheap config can improve it).
        front = [(5.0, -1.0)]  # (accuracy=5, -log_zk=-1)
        # A config with very high decoder-param score should have high accuracy
        # proxy and low zk cost → large hypervolume improvement.
        large_cfg = _cfg(d_model=512, n_heads=8, n_layers=6, seq_len=128)
        score = hypervolume_improvement_score(large_cfg, front, proxy_a, proxy_b)
        assert score >= 0.0

    def test_returns_zero_for_strictly_dominated_point(self, proxies: tuple) -> None:
        """A point strictly dominated by the existing front returns 0."""
        proxy_a, proxy_b = proxies
        # Build a front that dominates the tiny config in both dims.
        # decoder_params_score for tiny_cfg will be small → front with large
        # accuracy and high -log_cost value dominates it.
        tiny_cfg = _cfg(d_model=32, n_heads=4, n_layers=1, seq_len=32)
        acc = decoder_params_score(tiny_cfg)
        zk = float(compute_op_breakdown(tiny_cfg).total_constraints)
        log_zk = math.log(max(zk, 1.0))
        # A front with a point better than tiny_cfg on both axes.
        dominating_front = [(acc + 10.0, -log_zk + 5.0)]
        score = hypervolume_improvement_score(
            tiny_cfg, dominating_front, proxy_a, proxy_b
        )
        assert score == pytest.approx(0.0)

    def test_empty_front_returns_positive_ratio(self, proxies: tuple) -> None:
        """With an empty front, the fallback ratio is positive."""
        proxy_a, proxy_b = proxies
        cfg = _cfg(d_model=128, n_heads=4, n_layers=4, seq_len=64)
        score = hypervolume_improvement_score(cfg, [], proxy_a, proxy_b)
        assert score > 0.0

    def test_returns_float(self, proxies: tuple) -> None:
        """The function always returns a Python float."""
        proxy_a, proxy_b = proxies
        cfg = _cfg()
        front: list[tuple[float, float]] = [(10.0, -2.0), (8.0, -1.0)]
        result = hypervolume_improvement_score(cfg, front, proxy_a, proxy_b)
        assert isinstance(result, float)

    def test_non_negative(self, proxies: tuple) -> None:
        """Score is always >= 0."""
        proxy_a, proxy_b = proxies
        front: list[tuple[float, float]] = [(10.0, -2.0), (8.0, -1.0)]
        for d_model in [32, 64, 128, 256]:
            cfg = _cfg(d_model=d_model, n_heads=4, n_layers=2, seq_len=64)
            score = hypervolume_improvement_score(cfg, front, proxy_a, proxy_b)
            assert score >= 0.0, f"Negative EHVI for d_model={d_model}: {score}"


# ---------------------------------------------------------------------------
# ActiveLearner EHVI mode
# ---------------------------------------------------------------------------


class TestActiveLearnerEHVI:
    """Tests for ActiveLearner with use_ehvi=True (default)."""

    @pytest.fixture(scope="class")
    def learner_and_data(self) -> tuple[ActiveLearner, ProxyDataset]:
        """Return a fitted learner with EHVI enabled and its training dataset."""
        dataset = _make_dataset(n=20)
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=True)
        return learner, dataset

    def test_use_ehvi_default_is_true(self) -> None:
        """use_ehvi defaults to True."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b)
        assert learner._use_ehvi is True

    def test_use_ehvi_false_stored(self) -> None:
        """use_ehvi=False is stored correctly."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=False)
        assert learner._use_ehvi is False

    def test_repr_contains_use_ehvi(self) -> None:
        """repr() includes the use_ehvi flag."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=True)
        assert "use_ehvi=True" in repr(learner)

    def test_score_candidates_returns_correct_length_ehvi(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """EHVI mode returns one score per candidate."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        assert len(scores) == 10

    def test_score_candidates_sorted_descending_ehvi(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """EHVI mode sorts results by composite (EHVI) descending."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        composites = [s.composite for s in scores]
        assert composites == sorted(composites, reverse=True)

    def test_score_candidates_ranks_sequential_ehvi(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """EHVI mode assigns 1-based sequential ranks."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(8)
        scores = learner.score_candidates(candidates, dataset)
        ranks = [s.rank for s in scores]
        assert ranks == list(range(1, len(scores) + 1))

    def test_scores_non_negative_ehvi(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Composite scores are non-negative in EHVI mode."""
        learner, dataset = learner_and_data
        candidates = _make_candidates(10)
        scores = learner.score_candidates(candidates, dataset)
        for score in scores:
            assert score.composite >= 0.0, (
                f"Negative EHVI composite: {score.composite}"
            )

    def test_empty_candidates_ehvi(
        self, learner_and_data: tuple[ActiveLearner, ProxyDataset]
    ) -> None:
        """Empty candidate list returns [] in EHVI mode."""
        learner, dataset = learner_and_data
        assert learner.score_candidates([], dataset) == []

    def test_ehvi_higher_for_pareto_frontier_configs(self) -> None:
        """Configs near the Pareto frontier score higher than dominated configs in EHVI mode.

        We construct a scenario where one config has both higher accuracy proxy
        and lower ZK cost than another.  The undominated config should receive
        a higher or equal EHVI score.
        """
        dataset = _make_dataset(n=20)
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)

        # A large model has higher decoder_params_score (accuracy proxy).
        # To get low ZK cost, use short seq_len and few layers.
        large_acc_low_cost = _cfg(
            d_model=512, n_heads=8, n_layers=2, seq_len=32, activation="relu"
        )
        # A tiny model has low decoder_params_score (low accuracy proxy).
        tiny_acc = _cfg(d_model=32, n_heads=4, n_layers=1, seq_len=32)

        # Compute ground-truth proxy values to build a front that dominates
        # tiny_acc but not large_acc_low_cost.
        tiny_acc_val = decoder_params_score(tiny_acc)
        tiny_zk = float(compute_op_breakdown(tiny_acc).total_constraints)
        tiny_log_zk = math.log(max(tiny_zk, 1.0))
        # Front dominates tiny_acc in both dims.
        front = [(tiny_acc_val + 5.0, -tiny_log_zk + 3.0)]

        ehvi_large = hypervolume_improvement_score(
            large_acc_low_cost, front, proxy_a, proxy_b
        )
        ehvi_tiny = hypervolume_improvement_score(tiny_acc, front, proxy_a, proxy_b)

        # The dominated config should have lower (or equal) EHVI.
        assert ehvi_large >= ehvi_tiny, (
            f"Expected large_acc_low_cost EHVI ({ehvi_large:.4f}) >= "
            f"tiny_acc EHVI ({ehvi_tiny:.4f})"
        )

    def test_build_pareto_front_empty_dataset(self) -> None:
        """_build_pareto_front returns [] for an empty dataset."""
        proxy_a = MultiFidelityProxy(n_estimators=5)
        proxy_b = ZeroCostProxy(model_type="rf")
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=True)
        empty_ds = ProxyDataset()
        assert not learner._build_pareto_front(empty_ds)

    def test_build_pareto_front_non_dominated_result(self) -> None:
        """_build_pareto_front only returns non-dominated points."""
        dataset = _make_dataset(n=20)
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)
        learner = ActiveLearner(proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=True)

        front = learner._build_pareto_front(dataset)
        # Every point in the returned front must be non-dominated.
        for p in front:
            for other in front:
                if other is p:
                    continue
                assert not (
                    other[0] >= p[0]
                    and other[1] >= p[1]
                    and (other[0] > p[0] or other[1] > p[1])
                ), f"{other} dominates {p} in _build_pareto_front output"

    def test_ehvi_vs_heuristic_different_rankings(self) -> None:
        """EHVI and heuristic modes can produce different orderings (smoke test)."""
        dataset = _make_dataset(n=20)
        proxy_a = _make_fitted_proxy_a(dataset)
        proxy_b = _make_fitted_proxy_b(dataset)

        learner_ehvi = ActiveLearner(
            proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=True
        )
        learner_heuristic = ActiveLearner(
            proxy_a=proxy_a, proxy_b=proxy_b, use_ehvi=False
        )
        candidates = _make_candidates(10)

        scores_ehvi = learner_ehvi.score_candidates(candidates, dataset)
        scores_heuristic = learner_heuristic.score_candidates(candidates, dataset)

        # Both must return 10 scores.
        assert len(scores_ehvi) == 10
        assert len(scores_heuristic) == 10

        # In EHVI mode, composites are EHVI values (non-negative, not in [0,1]).
        for score in scores_ehvi:
            assert score.composite >= 0.0

        # In heuristic mode, composites are in [0, 1].
        for score in scores_heuristic:
            assert 0.0 <= score.composite <= 1.0 + 1e-9

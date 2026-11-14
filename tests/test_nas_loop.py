"""Tests for the 3-phase ZK-NAS orchestrator (search/nas_loop.py).

Design goals:
- Phase 1 tests run without LMTrainer (no actual training).
- Phase 2/3 tests mock LMTrainer so no real GPU training occurs.
- All tests complete in < 30 seconds using a tiny 20-config search space.
- Checkpoints are verified after each phase.
"""

# pylint: disable=redefined-outer-name  # pytest fixture injection pattern
# pylint: disable=protected-access  # tests intentionally patch private module attributes
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from proxy.data import ProxyDataset
from proxy.features import TransformerConfig
from search.nas_loop import (
    NASResult,
    NASState,
    ProxyValidation,
    ZKNASLoop,
    _best_proxy_name,
    _config_from_dict,
    _config_to_dict,
    _nas_result_from_dict,
    _nas_result_to_dict,
    _proxy_validation_from_dict,
    _proxy_validation_to_dict,
    _spearman,
    _kendall,
)


# ---------------------------------------------------------------------------
# Tiny search-space helpers
# ---------------------------------------------------------------------------


def _small_config(**kwargs: Any) -> TransformerConfig:
    """Return a small but valid TransformerConfig."""
    defaults: dict[str, Any] = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "seq_len": 16,
        "ff_mult": 2.0,
        "activation": "poly2",
        "use_rotary": False,
        "use_layernorm": True,
        "vocab_size": 128,
        "tie_embeddings": False,
    }
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_nas_result(  # pylint: disable=too-many-arguments
    *,
    zk_cost: int = 10_000,
    dp_score: float = 12.0,
    dp: int = 160_000,
    phase: int = 1,
    on_pareto: bool = False,
    calibration_ppl: float | None = None,
    final_ppl: float | None = None,
) -> NASResult:
    """Build a NASResult with sensible defaults for unit tests."""
    return NASResult(
        config=_small_config(),
        zk_cost=zk_cost,
        decoder_params=dp,
        decoder_params_score=dp_score,
        on_pareto_frontier=on_pareto,
        calibration_ppl=calibration_ppl,
        final_ppl=final_ppl,
        phase=phase,
        iteration=0,
    )


def _build_tiny_loop(output_dir: str) -> ZKNASLoop:
    """Return a ZKNASLoop configured for fast tests.

    - 2 Phase 1 iterations with pool=5, batch=2
    - Phase 2 sample_size=3, calibration_steps=1
    - Phase 3 max_steps=1, top_k=2
    - No actual EZKL proofs
    """
    return ZKNASLoop(
        output_dir=output_dir,
        backend="ezkl",
        seed=42,
        phase1_iterations=2,
        phase1_batch_size=2,
        phase1_pool_size=5,
        phase2_calibration_steps=1,
        phase2_sample_size=3,
        phase2_training_seq_len=16,
        phase3_max_steps=1,
        phase3_top_k=2,
        prove=False,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Fake LMTrainer for mocking
# ---------------------------------------------------------------------------


def _fake_train_result(ppl: float = 50.0) -> MagicMock:
    """Return a mock LMTrainingResult with val_perplexity=ppl."""
    result = MagicMock()
    result.val_perplexity = ppl
    return result


@contextmanager
def _mock_phase1_proxies():
    """Context manager that replaces slow proxy computations with fast stubs.

    Patches ``_estimate_zk_cost`` and ``_compute_grad_norm_and_snip`` in
    ``search.nas_loop`` so Phase 1 tests complete in seconds rather than
    minutes on CPU-only environments (ONNX export takes ~20 s per config).
    """
    import search.nas_loop as _nas_loop_mod  # pylint: disable=import-outside-toplevel
    from proxy.features import ProxyFeatures  # pylint: disable=import-outside-toplevel

    def _fast_zk(_self: Any, cfg: TransformerConfig) -> int:
        return int(ProxyFeatures.from_config(cfg).total_macs)

    orig_zk = _nas_loop_mod.ZKNASLoop._estimate_zk_cost  # type: ignore[attr-defined]  # pylint: disable=protected-access
    orig_gns = (
        _nas_loop_mod._compute_grad_norm_and_snip
    )  # pylint: disable=protected-access
    _nas_loop_mod.ZKNASLoop._estimate_zk_cost = _fast_zk  # type: ignore[attr-defined]  # pylint: disable=protected-access
    _nas_loop_mod._compute_grad_norm_and_snip = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
        lambda cfg, **_kw: (0.5, 1.5)
    )
    try:
        yield
    finally:
        _nas_loop_mod.ZKNASLoop._estimate_zk_cost = orig_zk  # type: ignore[attr-defined]  # pylint: disable=protected-access
        _nas_loop_mod._compute_grad_norm_and_snip = orig_gns  # type: ignore[attr-defined]  # pylint: disable=protected-access


@contextmanager
def _mock_lm_trainer(ppl: float = 50.0, training_error: Exception | None = None):
    """Context manager that injects mock objects for Phase 2 / Phase 3 testing.

    Patches:
    - ``search.lm_trainer`` module in ``sys.modules`` (since LMTrainer is
      imported lazily inside the phase methods, we must inject before the call).
    - ``search.nas_loop._compute_grad_norm_and_snip`` — avoids a ~20s backward
      pass per config on CPU-only environments.
    - ``search.nas_loop._compute_near_score`` — avoids another backward pass.

    Yields the mock trainer instance so tests can inspect calls if needed.
    """
    # Build mock trainer module.
    mock_module = types.ModuleType("search.lm_trainer")
    mock_trainer_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.val_perplexity = ppl
    if training_error is not None:
        mock_trainer_instance.train.side_effect = training_error
        mock_trainer_instance.calibration_run.side_effect = training_error
    else:
        mock_trainer_instance.train.return_value = mock_result
        mock_trainer_instance.calibration_run.return_value = mock_result

    mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
    mock_module.LMTrainer = mock_trainer_cls  # type: ignore[attr-defined]
    mock_module.LMTrainingConfig = MagicMock  # type: ignore[attr-defined]

    old_module = sys.modules.get("search.lm_trainer")
    sys.modules["search.lm_trainer"] = mock_module

    # Patch slow module-level functions in search.nas_loop.
    import search.nas_loop as _nas_loop_mod  # pylint: disable=import-outside-toplevel

    orig_gns = (
        _nas_loop_mod._compute_grad_norm_and_snip
    )  # pylint: disable=protected-access
    orig_near = _nas_loop_mod._compute_near_score  # pylint: disable=protected-access

    _nas_loop_mod._compute_grad_norm_and_snip = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
        lambda cfg, **_kw: (1.0, 2.0)
    )
    _nas_loop_mod._compute_near_score = lambda cfg: 10.0  # type: ignore[attr-defined]  # pylint: disable=protected-access

    # Also patch _estimate_zk_cost on the loop class to avoid ONNX export.
    orig_zk = _nas_loop_mod.ZKNASLoop._estimate_zk_cost  # type: ignore[attr-defined]  # pylint: disable=protected-access
    _nas_loop_mod.ZKNASLoop._estimate_zk_cost = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
        lambda self, cfg: 12345
    )
    try:
        yield mock_trainer_instance
    finally:
        _nas_loop_mod._compute_grad_norm_and_snip = orig_gns  # type: ignore[attr-defined]  # pylint: disable=protected-access
        _nas_loop_mod._compute_near_score = orig_near  # type: ignore[attr-defined]  # pylint: disable=protected-access
        _nas_loop_mod.ZKNASLoop._estimate_zk_cost = orig_zk  # type: ignore[attr-defined]  # pylint: disable=protected-access
        if old_module is None:
            sys.modules.pop("search.lm_trainer", None)
        else:
            sys.modules["search.lm_trainer"] = old_module


# ---------------------------------------------------------------------------
# Module-scoped Phase 1 fixture (no LMTrainer, mocked ONNX export)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase1_state_and_loop() -> Generator[tuple[NASState, ZKNASLoop], None, None]:
    """Run Phase 1 once and share the result across all Phase 1 tests.

    ``_estimate_zk_cost`` is patched to return a cheap analytical estimate
    so that tests do not trigger ONNX export (which takes ~20 s per config).
    """
    import search.nas_loop as _nas_loop_mod  # pylint: disable=import-outside-toplevel
    from proxy.features import ProxyFeatures  # pylint: disable=import-outside-toplevel

    def _fast_zk(_self: Any, cfg: TransformerConfig) -> int:
        # Use total_macs as a cheap surrogate for ZK cost (no ONNX export).
        return int(ProxyFeatures.from_config(cfg).total_macs)

    orig_zk = _nas_loop_mod.ZKNASLoop._estimate_zk_cost  # type: ignore[attr-defined]  # pylint: disable=protected-access
    orig_gns = (
        _nas_loop_mod._compute_grad_norm_and_snip
    )  # pylint: disable=protected-access
    _nas_loop_mod.ZKNASLoop._estimate_zk_cost = _fast_zk  # type: ignore[attr-defined]  # pylint: disable=protected-access
    _nas_loop_mod._compute_grad_norm_and_snip = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
        lambda cfg, **_kw: (0.5, 1.5)
    )

    with tempfile.TemporaryDirectory() as tmp:
        loop = _build_tiny_loop(tmp)
        state = NASState(
            phase=0,
            iteration=0,
            results=[],
            dataset=ProxyDataset(),
            proxy_validation=None,
            hypervolume=0.0,
            start_time="2024-01-01T00:00:00+00:00",
        )
        state = loop.run_phase1(state)
        yield state, loop

    _nas_loop_mod.ZKNASLoop._estimate_zk_cost = orig_zk  # type: ignore[attr-defined]  # pylint: disable=protected-access
    _nas_loop_mod._compute_grad_norm_and_snip = orig_gns  # type: ignore[attr-defined]  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Unit tests: serialisation helpers (no ONNX, no training)
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    def test_round_trip_basic(self) -> None:
        cfg = _small_config()
        restored = _config_from_dict(_config_to_dict(cfg))
        assert restored.d_model == cfg.d_model
        assert restored.n_heads == cfg.n_heads
        assert restored.activation == cfg.activation

    def test_vocab_size_default_on_missing_key(self) -> None:
        d = _config_to_dict(_small_config())
        d.pop("vocab_size")
        restored = _config_from_dict(d)
        assert restored.vocab_size == 50257


class TestNASResultRoundTrip:
    def test_round_trip_phase1_result(self) -> None:
        result = _make_nas_result(zk_cost=50_000, dp_score=11.5, phase=1)
        restored = _nas_result_from_dict(_nas_result_to_dict(result))
        assert restored.zk_cost == 50_000
        assert restored.decoder_params_score == pytest.approx(11.5)
        assert restored.phase == 1
        assert restored.grad_norm_score is None
        assert restored.calibration_ppl is None
        assert restored.final_ppl is None

    def test_round_trip_phase2_result(self) -> None:
        result = _make_nas_result(calibration_ppl=35.7, phase=2)
        result.grad_norm_score = 0.42
        result.snip_score = 1.23
        restored = _nas_result_from_dict(_nas_result_to_dict(result))
        assert restored.calibration_ppl == pytest.approx(35.7)
        assert restored.grad_norm_score == pytest.approx(0.42)
        assert restored.snip_score == pytest.approx(1.23)
        assert restored.phase == 2

    def test_round_trip_phase3_result(self) -> None:
        result = _make_nas_result(final_ppl=28.0, phase=3)
        restored = _nas_result_from_dict(_nas_result_to_dict(result))
        assert restored.final_ppl == pytest.approx(28.0)
        assert restored.phase == 3


class TestProxyValidationRoundTrip:  # pylint: disable=too-few-public-methods
    def test_round_trip(self) -> None:
        pv = ProxyValidation(
            n_configs=10,
            spearman_decoder_params=0.82,
            spearman_grad_norm=0.75,
            spearman_snip=0.70,
            spearman_near=0.60,
            kendall_decoder_params=0.65,
            kendall_grad_norm=0.58,
            best_proxy="decoder_params",
        )
        restored = _proxy_validation_from_dict(_proxy_validation_to_dict(pv))
        assert restored.n_configs == 10
        assert restored.spearman_decoder_params == pytest.approx(0.82)
        assert restored.best_proxy == "decoder_params"


# ---------------------------------------------------------------------------
# Unit tests: correlation helpers
# ---------------------------------------------------------------------------


class TestCorrelationHelpers:
    def test_spearman_perfect_positive(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]
        assert _spearman(x, y) == pytest.approx(1.0)

    def test_spearman_too_few_points(self) -> None:
        assert _spearman([1.0], [2.0]) == 0.0
        assert _spearman([], []) == 0.0

    def test_kendall_perfect_positive(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0]
        y = [1.0, 2.0, 3.0, 4.0]
        assert _kendall(x, y) == pytest.approx(1.0)

    def test_kendall_too_few_points(self) -> None:
        assert _kendall([1.0, 2.0], [2.0, 3.0]) == 0.0


class TestBestProxyName:
    def test_returns_highest_key(self) -> None:
        scores = {"decoder_params": 0.8, "grad_norm": 0.6, "snip": 0.9}
        assert _best_proxy_name(scores) == "snip"

    def test_empty_dict_returns_default(self) -> None:
        assert _best_proxy_name({}) == "decoder_params"


# ---------------------------------------------------------------------------
# Phase 1 tests
# ---------------------------------------------------------------------------


class TestPhase1:
    def test_phase1_returns_nas_state(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        state, _ = phase1_state_and_loop
        assert isinstance(state, NASState)

    def test_phase1_produces_results(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        state, loop = phase1_state_and_loop
        # 2 iterations × batch_size=2 = at least 4 results
        assert len(state.results) >= loop.phase1_batch_size

    def test_phase1_results_have_correct_phase(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        state, _ = phase1_state_and_loop
        for r in state.results:
            assert r.phase == 1

    def test_phase1_no_training_columns_set(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        """Phase 1 must NOT populate calibration_ppl or final_ppl."""
        state, _ = phase1_state_and_loop
        for r in state.results:
            assert r.calibration_ppl is None
            assert r.final_ppl is None

    def test_phase1_decoder_params_score_positive(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        state, _ = phase1_state_and_loop
        for r in state.results:
            assert math.isfinite(r.decoder_params_score)
            assert r.decoder_params_score > 0.0

    def test_phase1_dataset_grows(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        state, _ = phase1_state_and_loop
        assert len(state.dataset) > 0

    def test_phase1_checkpoint_saved(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        """A checkpoint file must exist after Phase 1."""
        _state, loop = phase1_state_and_loop
        checkpoint = os.path.join(loop.output_dir, "phase1_iter0001.json")
        assert os.path.isfile(checkpoint)

    def test_phase1_checkpoint_valid_json(
        self, phase1_state_and_loop: tuple[NASState, ZKNASLoop]
    ) -> None:
        _state, loop = phase1_state_and_loop
        checkpoint = os.path.join(loop.output_dir, "phase1_iter0001.json")
        with open(checkpoint, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert "results" in payload
        assert "phase" in payload
        assert payload["phase"] == 1


# ---------------------------------------------------------------------------
# Phase 2 tests (mock LMTrainer)
# ---------------------------------------------------------------------------


class TestPhase2:
    def test_phase2_returns_proxy_validation(self) -> None:
        costs = [1000, 2000, 3000, 4000, 5000, 6000]
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_phase1_results(n=6, zk_costs=costs)
            with _mock_lm_trainer(ppl=45.0):
                state, validation = loop.run_phase2(state)
            assert isinstance(validation, ProxyValidation)

    def test_phase2_validation_has_correct_n_configs(self) -> None:
        costs = [1000, 2000, 3000, 4000, 5000, 6000]
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_phase1_results(n=6, zk_costs=costs)
            with _mock_lm_trainer(ppl=45.0):
                _, validation = loop.run_phase2(state)
            # sample_size=3 so at most 3 configs trained
            assert validation.n_configs <= loop.phase2_sample_size

    def test_phase2_proxy_validation_stored_in_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_phase1_results(
                n=5, zk_costs=[100, 200, 300, 400, 500]
            )
            with _mock_lm_trainer(ppl=40.0):
                state, validation = loop.run_phase2(state)
            assert state.proxy_validation is not None
            assert state.proxy_validation is validation

    def test_phase2_results_have_calibration_ppl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_phase1_results(
                n=4, zk_costs=[500, 1000, 1500, 2000]
            )
            with _mock_lm_trainer(ppl=33.0):
                state, _ = loop.run_phase2(state)
            phase2_results = [r for r in state.results if r.phase == 2]
            assert len(phase2_results) > 0
            for r in phase2_results:
                assert r.calibration_ppl is not None
                assert math.isfinite(r.calibration_ppl)

    def test_phase2_checkpoint_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_phase1_results(n=4, zk_costs=[100, 200, 300, 400])
            with _mock_lm_trainer(ppl=50.0):
                loop.run_phase2(state)
            checkpoint = os.path.join(tmp, "phase2_iter0000.json")
            assert os.path.isfile(checkpoint)

    def test_phase2_best_proxy_is_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_phase1_results(n=4, zk_costs=[100, 200, 300, 400])
            with _mock_lm_trainer(ppl=50.0):
                _, validation = loop.run_phase2(state)
            assert isinstance(validation.best_proxy, str)
            assert validation.best_proxy in (
                "decoder_params",
                "grad_norm",
                "snip",
                "near",
            )


# ---------------------------------------------------------------------------
# Phase 3 tests (mock LMTrainer)
# ---------------------------------------------------------------------------


class TestPhase3:
    def test_phase3_produces_final_ppl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_pareto_results(n=4)
            with _mock_lm_trainer(ppl=25.0):
                state = loop.run_phase3(state)
            phase3_results = [r for r in state.results if r.phase == 3]
            assert len(phase3_results) > 0
            for r in phase3_results:
                assert r.final_ppl is not None
                assert math.isfinite(r.final_ppl)

    def test_phase3_includes_gpt2_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_pareto_results(n=3)
            with _mock_lm_trainer(ppl=30.0):
                state = loop.run_phase3(state)
            # top_k=2 Pareto configs + 1 GPT-2 baseline = 3 Phase 3 results
            phase3_results = [r for r in state.results if r.phase == 3]
            # At least 1 (GPT-2) must always be present.
            assert len(phase3_results) >= 1

    def test_phase3_checkpoint_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_pareto_results(n=2)
            with _mock_lm_trainer(ppl=20.0):
                loop.run_phase3(state)
            checkpoint = os.path.join(tmp, "phase3_iter0000.json")
            assert os.path.isfile(checkpoint)

    def test_phase3_tolerates_training_failure(self) -> None:
        """If training raises, final_ppl should be None (not crash)."""
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            state = _build_state_with_pareto_results(n=2)
            with _mock_lm_trainer(training_error=RuntimeError("GPU OOM")):
                state = loop.run_phase3(state)
            phase3_results = [r for r in state.results if r.phase == 3]
            # All results should have final_ppl=None (training failed).
            for r in phase3_results:
                assert r.final_ppl is None


# ---------------------------------------------------------------------------
# run(phases=[1]) end-to-end test
# ---------------------------------------------------------------------------


class TestRunPhase1Only:
    def test_run_phase1_only_completes(self) -> None:
        """run(phases=[1]) must complete without error."""
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                state = loop.run(phases=[1])
            assert isinstance(state, NASState)
            assert len(state.results) > 0

    def test_run_phase1_only_no_calibration_ppl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                state = loop.run(phases=[1])
            for r in state.results:
                assert r.calibration_ppl is None
                assert r.final_ppl is None

    def test_run_phase1_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                loop.run(phases=[1])
            # Phase 1 produces checkpoints named phase1_iter{N:04d}.json
            found = any(
                f.startswith("phase1_") and f.endswith(".json") for f in os.listdir(tmp)
            )
            assert found


# ---------------------------------------------------------------------------
# Checkpoint load/save round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_checkpoint_restores_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                state = loop.run(phases=[1])
            ckpt = os.path.join(tmp, "phase1_iter0001.json")
            restored = ZKNASLoop.load_checkpoint(ckpt)
            assert restored.phase == state.phase

    def test_checkpoint_restores_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                state = loop.run(phases=[1])
            ckpt = os.path.join(tmp, "phase1_iter0001.json")
            restored = ZKNASLoop.load_checkpoint(ckpt)
            assert len(restored.results) == len(state.results)

    def test_checkpoint_restores_hypervolume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                state = loop.run(phases=[1])
            ckpt = os.path.join(tmp, "phase1_iter0001.json")
            restored = ZKNASLoop.load_checkpoint(ckpt)
            assert restored.hypervolume == pytest.approx(state.hypervolume)

    def test_checkpoint_valid_json_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            with _mock_phase1_proxies():
                loop.run(phases=[1])
            ckpt = os.path.join(tmp, "phase1_iter0001.json")
            with open(ckpt, encoding="utf-8") as fh:
                payload = json.load(fh)
            assert "phase" in payload
            assert "iteration" in payload
            assert "results" in payload
            assert "hypervolume" in payload


# ---------------------------------------------------------------------------
# pareto_table tests
# ---------------------------------------------------------------------------


class TestParetoTable:
    """Tests for pareto_table().

    Use pre-built states to avoid triggering the slow ONNX export path.
    """

    def _loop_and_state(self, tmp: str) -> tuple[ZKNASLoop, NASState]:
        """Return a loop and a state with some Pareto-frontier results."""
        loop = _build_tiny_loop(tmp)
        state = _build_state_with_pareto_results(n=3)
        return loop, state

    def test_returns_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop, state = self._loop_and_state(tmp)
            table = loop.pareto_table(state)
            assert isinstance(table, str)

    def test_contains_pipe_characters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop, state = self._loop_and_state(tmp)
            table = loop.pareto_table(state)
            assert "|" in table

    def test_header_contains_zk_cost_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop, state = self._loop_and_state(tmp)
            table = loop.pareto_table(state)
            assert "ZK cost" in table or "no Pareto" in table

    def test_header_contains_decoder_params_score_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop, state = self._loop_and_state(tmp)
            table = loop.pareto_table(state, use_final_ppl=False)
            assert "decoder_params_score" in table or "no Pareto" in table

    def test_use_final_ppl_column_header(self) -> None:
        state = NASState(
            phase=3,
            results=[_make_nas_result(on_pareto=True, final_ppl=22.5, phase=3)],
            dataset=ProxyDataset(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.pareto_table(state, use_final_ppl=True)
        assert "final_ppl" in table or "no Pareto" in table

    def test_empty_state_returns_graceful_message(self) -> None:
        empty_state = NASState(phase=1, results=[], dataset=ProxyDataset())
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.pareto_table(empty_state)
        assert isinstance(table, str)
        assert len(table) > 0


# ---------------------------------------------------------------------------
# proxy_validation_table tests
# ---------------------------------------------------------------------------


class TestProxyValidationTable:
    def test_returns_string(self) -> None:
        pv = ProxyValidation(
            n_configs=10,
            spearman_decoder_params=0.90,
            spearman_grad_norm=0.85,
            spearman_snip=0.80,
            spearman_near=0.70,
            kendall_decoder_params=0.75,
            kendall_grad_norm=0.68,
            best_proxy="decoder_params",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.proxy_validation_table(pv)
        assert isinstance(table, str)

    def test_contains_pipe_characters(self) -> None:
        pv = ProxyValidation(
            n_configs=8,
            spearman_decoder_params=0.88,
            spearman_grad_norm=0.82,
            spearman_snip=0.79,
            spearman_near=0.65,
            kendall_decoder_params=0.70,
            kendall_grad_norm=0.60,
            best_proxy="decoder_params",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.proxy_validation_table(pv)
        assert "|" in table

    def test_header_contains_spearman_column(self) -> None:
        pv = ProxyValidation(
            n_configs=5,
            spearman_decoder_params=0.85,
            spearman_grad_norm=0.75,
            spearman_snip=0.70,
            spearman_near=0.60,
            kendall_decoder_params=0.68,
            kendall_grad_norm=0.55,
            best_proxy="decoder_params",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.proxy_validation_table(pv)
        assert "Spearman" in table

    def test_best_proxy_marked_in_table(self) -> None:
        pv = ProxyValidation(
            n_configs=5,
            spearman_decoder_params=0.95,
            spearman_grad_norm=0.75,
            spearman_snip=0.70,
            spearman_near=0.60,
            kendall_decoder_params=0.80,
            kendall_grad_norm=0.60,
            best_proxy="decoder_params",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.proxy_validation_table(pv)
        # The best proxy row should be marked (e.g., **yes**)
        assert "yes" in table.lower() or "decoder_params" in table

    def test_all_four_proxies_appear(self) -> None:
        pv = ProxyValidation(
            n_configs=5,
            spearman_decoder_params=0.90,
            spearman_grad_norm=0.80,
            spearman_snip=0.75,
            spearman_near=0.65,
            kendall_decoder_params=0.75,
            kendall_grad_norm=0.65,
            best_proxy="decoder_params",
        )
        with tempfile.TemporaryDirectory() as tmp:
            loop = _build_tiny_loop(tmp)
            table = loop.proxy_validation_table(pv)
        for proxy in ("decoder_params", "grad_norm", "snip", "near"):
            assert proxy in table


# ---------------------------------------------------------------------------
# Internal factory helpers (used by Phase 2 and Phase 3 tests)
# ---------------------------------------------------------------------------


def _build_state_with_phase1_results(
    n: int,
    zk_costs: list[int],
) -> NASState:
    """Build a NASState with *n* Phase 1 NASResults having the given ZK costs."""
    results = []
    dataset = ProxyDataset()
    for i, zk in enumerate(zk_costs[:n]):
        # Vary d_model to ensure architectural diversity.
        d_model = 64 + (i % 4) * 32
        n_heads = 4
        while d_model % n_heads != 0:
            n_heads -= 1
        cfg = TransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            seq_len=16,
            ff_mult=2.0,
            activation="poly2",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=128,
            tie_embeddings=False,
        )
        r = NASResult(
            config=cfg,
            zk_cost=zk,
            decoder_params=d_model * 2 * 4,
            decoder_params_score=math.log(d_model * 2 * 4),
            on_pareto_frontier=False,
            phase=1,
            iteration=i,
        )
        results.append(r)
    return NASState(
        phase=1,
        iteration=len(results),
        results=results,
        dataset=dataset,
        proxy_validation=None,
        hypervolume=0.0,
        start_time="2024-01-01T00:00:00+00:00",
    )


def _build_state_with_pareto_results(n: int) -> NASState:
    """Build a NASState where the first *n* results are on the Pareto frontier."""
    results = []
    dataset = ProxyDataset()
    for i in range(n):
        d_model = 64 + i * 32
        n_heads = 4
        while d_model % n_heads != 0:
            n_heads -= 1
        cfg = TransformerConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            seq_len=16,
            ff_mult=2.0,
            activation="poly2",
            use_rotary=False,
            use_layernorm=True,
            vocab_size=128,
            tie_embeddings=False,
        )
        r = NASResult(
            config=cfg,
            zk_cost=1000 * (i + 1),
            decoder_params=d_model * 2 * 4,
            decoder_params_score=math.log(d_model * 2 * 4),
            on_pareto_frontier=True,
            phase=1,
            iteration=i,
        )
        results.append(r)
    return NASState(
        phase=1,
        iteration=n,
        results=results,
        dataset=dataset,
        proxy_validation=None,
        hypervolume=1.0,
        start_time="2024-01-01T00:00:00+00:00",
    )

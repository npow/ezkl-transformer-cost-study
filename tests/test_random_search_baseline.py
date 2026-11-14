"""Tests for experiments/run_random_search_baseline.py.

All tests run entirely on CPU in a few seconds — no training, no ONNX export,
no GPU required.  The fast path uses ``n_configs=20`` to keep the hypervolume
computation tractable in CI.
"""

# pylint: disable=redefined-outer-name  # pytest fixture injection pattern

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — ensures first-party modules are importable when tests are
# run from any working directory.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parents[1]))

# pylint: disable=wrong-import-position
from experiments.run_random_search_baseline import (  # noqa: E402
    compute_search_efficiency,
    load_zknas_curve,
    plot_hypervolume_comparison,
    run_hand_designed_baselines,
    run_random_search,
)

# pylint: enable=wrong-import-position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_random_results() -> dict:
    """Run random search with 20 configs once; reuse across all tests in module."""
    return run_random_search(n_configs=20, seed=99, seq_len_fixed=512)


# ---------------------------------------------------------------------------
# run_random_search tests
# ---------------------------------------------------------------------------


class TestRunRandomSearch:
    """Tests for :func:`run_random_search`."""

    def test_returns_dict_with_required_keys(self, small_random_results: dict) -> None:
        """Returned dict must contain exactly the six documented keys."""
        required = {
            "configs",
            "zk_costs",
            "dp_scores",
            "hypervolume_curve",
            "final_hypervolume",
            "pareto_size",
        }
        assert required == set(small_random_results.keys())

    def test_configs_length(self, small_random_results: dict) -> None:
        """configs list must have exactly n_configs entries."""
        assert len(small_random_results["configs"]) == 20

    def test_zk_costs_length(self, small_random_results: dict) -> None:
        """zk_costs list must have exactly n_configs entries."""
        assert len(small_random_results["zk_costs"]) == 20

    def test_dp_scores_length(self, small_random_results: dict) -> None:
        """dp_scores list must have exactly n_configs entries."""
        assert len(small_random_results["dp_scores"]) == 20

    def test_hypervolume_curve_length(self, small_random_results: dict) -> None:
        """hypervolume_curve must have exactly n_configs entries."""
        assert len(small_random_results["hypervolume_curve"]) == 20

    def test_hypervolume_curve_indices(self, small_random_results: dict) -> None:
        """First element of each tuple must be the 1-based config index."""
        for i, (n_eval, _) in enumerate(small_random_results["hypervolume_curve"]):
            assert n_eval == i + 1

    def test_hypervolume_non_decreasing(self, small_random_results: dict) -> None:
        """Hypervolume must be non-decreasing as more configs are evaluated."""
        hvs = [hv for _, hv in small_random_results["hypervolume_curve"]]
        for i in range(1, len(hvs)):
            assert hvs[i] >= hvs[i - 1] - 1e-6, (
                f"Hypervolume decreased at step {i}: {hvs[i - 1]} → {hvs[i]}"
            )

    def test_final_hypervolume_matches_curve(self, small_random_results: dict) -> None:
        """final_hypervolume must match the last entry in hypervolume_curve."""
        last_hv = small_random_results["hypervolume_curve"][-1][1]
        assert abs(small_random_results["final_hypervolume"] - last_hv) < 1e-6

    def test_pareto_size_positive(self, small_random_results: dict) -> None:
        """pareto_size must be at least 1 (one config always on frontier)."""
        assert small_random_results["pareto_size"] >= 1

    def test_all_seq_lens_fixed(self, small_random_results: dict) -> None:
        """All sampled configs must have seq_len=512."""
        for cfg in small_random_results["configs"]:
            assert cfg.seq_len == 512

    def test_zk_costs_positive(self, small_random_results: dict) -> None:
        """All ZK constraint counts must be strictly positive."""
        for zk in small_random_results["zk_costs"]:
            assert zk > 0

    def test_dp_scores_finite(self, small_random_results: dict) -> None:
        """All decoder_params scores must be finite floats."""
        for dp in small_random_results["dp_scores"]:
            assert math.isfinite(dp)

    def test_different_seed_gives_different_configs(self) -> None:
        """Two different seeds must produce at least one distinct config."""
        r1 = run_random_search(n_configs=5, seed=1, seq_len_fixed=512)
        r2 = run_random_search(n_configs=5, seed=2, seq_len_fixed=512)
        configs_1 = [(c.d_model, c.n_layers) for c in r1["configs"]]
        configs_2 = [(c.d_model, c.n_layers) for c in r2["configs"]]
        assert configs_1 != configs_2

    def test_reproducibility(self) -> None:
        """Same seed must produce identical results."""
        r1 = run_random_search(n_configs=5, seed=77, seq_len_fixed=512)
        r2 = run_random_search(n_configs=5, seed=77, seq_len_fixed=512)
        assert r1["zk_costs"] == r2["zk_costs"]
        assert r1["dp_scores"] == r2["dp_scores"]


# ---------------------------------------------------------------------------
# run_hand_designed_baselines tests
# ---------------------------------------------------------------------------


class TestRunHandDesignedBaselines:
    """Tests for :func:`run_hand_designed_baselines`."""

    @pytest.fixture(scope="class")
    def baselines(self) -> dict:
        """Evaluate hand-designed baselines once; share across class tests."""
        return run_hand_designed_baselines()

    def test_returns_four_entries(self, baselines: dict) -> None:
        """Must return exactly four hand-designed strategies."""
        assert len(baselines) == 4

    def test_expected_strategy_names(self, baselines: dict) -> None:
        """Returned keys must exactly match the four expected strategy names."""
        expected = {
            "width-only (12L/256d)",
            "depth-only (768d/4L)",
            "proportional (512d/6L)",
            "arch-optimal (384d/8L/relu)",
        }
        assert set(baselines.keys()) == expected

    def test_each_entry_has_required_keys(self, baselines: dict) -> None:
        """Every baseline entry must have config, zk_cost, and dp_score."""
        for name, entry in baselines.items():
            assert "config" in entry, f"Missing 'config' in {name!r}"
            assert "zk_cost" in entry, f"Missing 'zk_cost' in {name!r}"
            assert "dp_score" in entry, f"Missing 'dp_score' in {name!r}"

    def test_zk_costs_positive(self, baselines: dict) -> None:
        """ZK cost must be strictly positive for every baseline."""
        for name, entry in baselines.items():
            assert entry["zk_cost"] > 0, f"zk_cost not positive for {name!r}"

    def test_dp_scores_finite(self, baselines: dict) -> None:
        """decoder_params score must be a finite float for every baseline."""
        for name, entry in baselines.items():
            assert math.isfinite(entry["dp_score"]), (
                f"dp_score not finite for {name!r}"
            )

    def test_all_seq_len_512(self, baselines: dict) -> None:
        """All baseline configs must have seq_len=512 (iso-context)."""
        for name, entry in baselines.items():
            assert entry["config"].seq_len == 512, (
                f"Expected seq_len=512 for {name!r}"
            )


# ---------------------------------------------------------------------------
# load_zknas_curve tests
# ---------------------------------------------------------------------------


class TestLoadZknasCurve:
    """Tests for :func:`load_zknas_curve`."""

    def test_returns_empty_for_nonexistent_dir(self) -> None:
        """Non-existent directory must return an empty list without raising."""
        curve = load_zknas_curve(phase1_dir=Path("/nonexistent/path/xyz"))
        assert curve == []

    def test_loads_from_real_phase1_dir(self) -> None:
        """If Phase 1 results exist, the curve must have 20 entries."""
        phase1_dir = Path(__file__).parents[1] / "experiments" / "results" / "nas_phase1"
        if not phase1_dir.exists():
            pytest.skip("Phase 1 results directory not present")
        curve = load_zknas_curve(phase1_dir=phase1_dir)
        assert len(curve) == 20

    def test_curve_is_sorted(self) -> None:
        """Returned curve must be sorted by n_evaluated in ascending order."""
        phase1_dir = Path(__file__).parents[1] / "experiments" / "results" / "nas_phase1"
        if not phase1_dir.exists():
            pytest.skip("Phase 1 results directory not present")
        curve = load_zknas_curve(phase1_dir=phase1_dir)
        ns = [n for n, _ in curve]
        assert ns == sorted(ns)

    def test_loads_from_synthetic_dir(self, tmp_path: Path) -> None:
        """Synthetic checkpoint files must be parsed correctly."""
        for i in range(3):
            payload = {"iteration": i, "hypervolume": float((i + 1) * 100)}
            (tmp_path / f"phase1_iter{i:04d}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        curve = load_zknas_curve(phase1_dir=tmp_path, batch_size=8)
        assert len(curve) == 3
        assert curve[0] == (8, 100.0)
        assert curve[1] == (16, 200.0)
        assert curve[2] == (24, 300.0)


# ---------------------------------------------------------------------------
# compute_search_efficiency tests
# ---------------------------------------------------------------------------


class TestComputeSearchEfficiency:
    """Tests for :func:`compute_search_efficiency`."""

    def test_returns_n_when_reached(self) -> None:
        """Returns n_evaluated at the first point where hv >= threshold."""
        curve = [(10, 50.0), (20, 100.0), (30, 150.0)]
        result = compute_search_efficiency(100.0, curve)
        assert result == 20

    def test_returns_none_when_never_reached(self) -> None:
        """Returns None when no point in the curve reaches the threshold."""
        curve = [(10, 50.0), (20, 80.0)]
        result = compute_search_efficiency(200.0, curve)
        assert result is None

    def test_exact_match(self) -> None:
        """Exact equality with the threshold must return that n_evaluated."""
        curve = [(5, 99.9), (10, 100.0)]
        result = compute_search_efficiency(100.0, curve)
        assert result == 10

    def test_first_point_sufficient(self) -> None:
        """Returns the very first n_evaluated when the first point is enough."""
        curve = [(1, 500.0), (2, 600.0)]
        result = compute_search_efficiency(100.0, curve)
        assert result == 1


# ---------------------------------------------------------------------------
# plot_hypervolume_comparison tests
# ---------------------------------------------------------------------------


class TestPlotHypervolumeComparison:
    """Tests for :func:`plot_hypervolume_comparison`."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """Output PNG must exist and be non-empty after plotting."""
        zknas_curve = [(8, 1000.0), (16, 1500.0), (24, 1800.0)]
        random_curve = [(1, 100.0), (8, 800.0), (16, 1200.0), (24, 1600.0)]
        out = str(tmp_path / "comparison.png")
        plot_hypervolume_comparison(zknas_curve, random_curve, out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_no_error_with_empty_zknas_curve(self, tmp_path: Path) -> None:
        """Plot must still be produced even when ZK-NAS curve is empty."""
        random_curve = [(1, 100.0), (2, 200.0)]
        out = str(tmp_path / "comparison_empty_zk.png")
        plot_hypervolume_comparison([], random_curve, out)
        assert Path(out).exists()

    def test_no_error_with_empty_random_curve(self, tmp_path: Path) -> None:
        """Empty random_curve should skip plotting gracefully (no exception)."""
        out = str(tmp_path / "comparison_empty_rs.png")
        plot_hypervolume_comparison([(8, 1000.0)], [], out)
        # File should not be created when random_curve is empty.
        assert not Path(out).exists()

"""Tests for experiments/proxy_validation.py."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure experiments/ is importable as a module directory.
_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.proxy_validation import (
    _top10_precision,
    markdown_table,
    plot_proxy_correlations,
    run_proxy_validation,
    stratified_sample,
)
from proxy.features import TransformerConfig
from search.transformer_space import TransformerSearchSpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(n: int = 10) -> dict[str, Any]:
    """Build a synthetic results dict with *n* configs."""
    rng = np.random.RandomState(0)  # pylint: disable=no-member
    ppls = (rng.rand(n) * 200 + 20).tolist()  # PPL in [20, 220]
    return {
        "n_configs": n,
        "calibration_steps": 5,
        "proxies": {
            "decoder_params": {"spearman": 0.92, "kendall": 0.78, "top10_prec": 0.80},
            "grad_norm": {"spearman": 0.88, "kendall": 0.74, "top10_prec": 0.70},
            "snip": {"spearman": 0.89, "kendall": 0.76, "top10_prec": 0.72},
            "near": {"spearman": 0.71, "kendall": 0.55, "top10_prec": 0.55},
        },
        "best_proxy": "decoder_params",
        "configs": [],
        "scores": {
            "decoder_params": (rng.rand(n) * 10).tolist(),
            "grad_norm": (rng.rand(n) * 1000).tolist(),
            "snip": (rng.rand(n) * 500).tolist(),
            "near": (rng.rand(n) * 5).tolist(),
        },
        "ppls": ppls,
    }


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------


class TestStratifiedSample:
    def test_returns_correct_n(self) -> None:
        """stratified_sample returns exactly n configs."""
        space = TransformerSearchSpace(seed=0)
        configs = stratified_sample(space, n=20, seed=7)
        assert len(configs) == 20

    def test_all_valid_configs(self) -> None:
        """All returned configs pass is_valid()."""
        space = TransformerSearchSpace(seed=0)
        configs = stratified_sample(space, n=30, seed=42)
        for cfg in configs:
            assert space.is_valid(cfg), f"Invalid config: {cfg}"

    def test_tier_distribution(self) -> None:
        """~20% small, ~60% medium, ~20% large (within tolerance)."""
        space = TransformerSearchSpace(seed=1)
        n = 50
        configs = stratified_sample(space, n=n, seed=0)

        small = sum(1 for c in configs if c.d_model <= 384)
        large = sum(1 for c in configs if c.d_model > 768)

        # Allow ±5 absolute units of tolerance on a sample of 50.
        assert 3 <= small <= 17, f"Expected ~10 small, got {small}"
        assert 3 <= large <= 17, f"Expected ~10 large, got {large}"

    def test_zk_friendly_minimum(self) -> None:
        """At least 10 configs have ZK-friendly activations when n >= 20."""
        space = TransformerSearchSpace(seed=2)
        configs = stratified_sample(space, n=50, seed=11)
        zk_friendly = {"poly2", "poly3", "linear"}
        zk_count = sum(1 for c in configs if c.activation in zk_friendly)
        assert zk_count >= 10, f"Expected >= 10 ZK-friendly, got {zk_count}"

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed produces the same configs."""
        space1 = TransformerSearchSpace(seed=99)
        space2 = TransformerSearchSpace(seed=99)
        c1 = stratified_sample(space1, n=15, seed=99)
        c2 = stratified_sample(space2, n=15, seed=99)
        for a, b in zip(c1, c2):
            assert a.d_model == b.d_model
            assert a.n_layers == b.n_layers
            assert a.activation == b.activation


# ---------------------------------------------------------------------------
# markdown_table
# ---------------------------------------------------------------------------


class TestMarkdownTable:
    def test_returns_string(self) -> None:
        """markdown_table returns a non-empty string."""
        results = _make_results()
        table = markdown_table(results)
        assert isinstance(table, str)
        assert len(table) > 0

    def test_contains_header_columns(self) -> None:
        """Table contains the required column headers."""
        results = _make_results()
        table = markdown_table(results)
        for col in ("Proxy", "Spearman", "Kendall", "P@10%", "Notes"):
            assert col in table, f"Missing column header: {col!r}"

    def test_contains_all_proxy_rows(self) -> None:
        """Table contains a row for each proxy."""
        results = _make_results()
        table = markdown_table(results)
        for display_name in (
            "Decoder params",
            "GradNorm",
            "SNIP",
            "NEAR (ablation)",
        ):
            assert display_name in table, f"Missing proxy row: {display_name!r}"

    def test_separator_line_present(self) -> None:
        """Table contains a markdown separator line (|---|)."""
        results = _make_results()
        table = markdown_table(results)
        assert "---" in table

    def test_numeric_values_present(self) -> None:
        """Spearman values from the results dict appear in the table."""
        results = _make_results()
        table = markdown_table(results)
        # decoder_params spearman = 0.92 → should appear as 0.92
        assert "0.92" in table

    def test_notes_present(self) -> None:
        """Note strings appear in the table."""
        results = _make_results()
        table = markdown_table(results)
        assert "No model needed" in table
        assert "1 backward pass" in table
        assert "Not recommended" in table

    def test_sorted_by_spearman(self) -> None:
        """Best-Spearman proxy appears before worst-Spearman proxy in the table."""
        results = _make_results()
        table = markdown_table(results)
        lines = [l for l in table.splitlines() if "|" in l and "---" not in l and "Proxy" not in l]
        # First data row should be the proxy with highest |spearman|
        assert "Decoder params" in lines[0], f"Expected decoder_params first, got: {lines[0]}"


# ---------------------------------------------------------------------------
# _top10_precision (internal helper, tested via public API)
# ---------------------------------------------------------------------------


class TestTop10Precision:
    def test_perfect_ranking(self) -> None:
        """Perfect proxy → top10_prec = 1.0."""
        ppls = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        # Perfect proxy: higher score = lower PPL
        scores = [100.0 - p for p in ppls]  # 90,80,...,0
        p10 = _top10_precision(scores, ppls, higher_is_better=True)
        assert p10 == pytest.approx(1.0)

    def test_random_ranking_low_precision(self) -> None:
        """Random permutation yields precision typically below 1.0."""
        rng = np.random.RandomState(7)  # pylint: disable=no-member
        n = 100
        ppls = rng.rand(n).tolist()
        scores = rng.rand(n).tolist()  # random — no correlation
        p10 = _top10_precision(scores, ppls, higher_is_better=True)
        assert 0.0 <= p10 <= 1.0

    def test_single_element(self) -> None:
        """With a single element, returns nan."""
        p10 = _top10_precision([1.0], [50.0])
        assert math.isnan(p10)


# ---------------------------------------------------------------------------
# plot_proxy_correlations
# ---------------------------------------------------------------------------


class TestPlotProxyCorrelations:
    def test_saves_png_file(self, tmp_path: Path) -> None:
        """plot_proxy_correlations saves a PNG file at the given path."""
        results = _make_results(n=10)
        out = str(tmp_path / "test_plot.png")
        plot_proxy_correlations(results, out)
        assert os.path.exists(out), "PNG file was not created"
        assert os.path.getsize(out) > 0, "PNG file is empty"

    def test_png_magic_bytes(self, tmp_path: Path) -> None:
        """The saved file starts with the PNG magic bytes."""
        results = _make_results(n=10)
        out = str(tmp_path / "test_plot.png")
        plot_proxy_correlations(results, out)
        with open(out, "rb") as fh:
            header = fh.read(8)
        assert header == b"\x89PNG\r\n\x1a\n", "File does not look like a PNG"


# ---------------------------------------------------------------------------
# run_proxy_validation — smoke test with mocked LMTrainer
# ---------------------------------------------------------------------------


class TestRunProxyValidation:
    """Smoke test: n_configs=3, calibration_steps=5.

    Both ``LMTrainer`` and ``_compute_proxy_scores`` are mocked so no real model
    builds or WikiText-2 downloads happen during the test.
    """

    @staticmethod
    def _make_mock_trainer() -> MagicMock:
        """Return a mock LMTrainer instance whose calibration_run returns a fixed result."""
        mock_result = MagicMock()
        mock_result.val_perplexity = 150.0

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.calibration_run.return_value = mock_result
        return mock_trainer_instance

    @staticmethod
    def _proxy_scores_side_effect(
        config: TransformerConfig,
        device: str,  # noqa: ARG004 — required by _compute_proxy_scores signature
        seq_len: int,  # noqa: ARG004
    ) -> tuple[float, float, float, float, MagicMock]:
        """Return synthetic proxy scores + a mock model without building a real model."""
        _ = device, seq_len  # unused but required by the mocked function signature
        dp = float(config.d_model * config.n_layers)
        gn = float(config.n_layers * 100.0)
        sn = float(config.d_model * 0.5)
        nr = 0.0
        mock_model = MagicMock()
        return dp, gn, sn, nr, mock_model

    def _run_smoke(self, tmp_path: Path, seed: int) -> dict[str, Any]:
        """Helper: run validation with all heavy ops mocked, return results dict."""
        mock_trainer = self._make_mock_trainer()
        with patch(
            "experiments.proxy_validation.LMTrainer",
            return_value=mock_trainer,
        ), patch(
            "experiments.proxy_validation._compute_proxy_scores",
            side_effect=self._proxy_scores_side_effect,
        ):
            return run_proxy_validation(
                n_configs=3,
                calibration_steps=5,
                seq_len=128,
                output_dir=str(tmp_path),
                seed=seed,
                device="cpu",
            )

    def test_smoke_returns_dict(self, tmp_path: Path) -> None:
        """run_proxy_validation returns a dict with all required top-level keys."""
        results = self._run_smoke(tmp_path, seed=0)
        required_keys = {
            "n_configs",
            "calibration_steps",
            "proxies",
            "best_proxy",
            "configs",
            "scores",
            "ppls",
        }
        assert required_keys <= set(results.keys()), (
            f"Missing keys: {required_keys - set(results.keys())}"
        )

    def test_smoke_proxy_stats_keys(self, tmp_path: Path) -> None:
        """Each proxy entry in results['proxies'] has spearman, kendall, top10_prec."""
        results = self._run_smoke(tmp_path, seed=1)
        for proxy_name in ("decoder_params", "grad_norm", "snip", "near"):
            assert proxy_name in results["proxies"], f"Missing proxy: {proxy_name}"
            stats = results["proxies"][proxy_name]
            for stat_key in ("spearman", "kendall", "top10_prec"):
                assert stat_key in stats, f"Missing stat {stat_key!r} in {proxy_name}"

    def test_smoke_saves_json(self, tmp_path: Path) -> None:
        """run_proxy_validation writes results.json to output_dir."""
        self._run_smoke(tmp_path, seed=2)
        json_path = tmp_path / "results.json"
        assert json_path.exists(), "results.json was not created"
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "proxies" in data

    def test_smoke_saves_markdown(self, tmp_path: Path) -> None:
        """run_proxy_validation writes table1.md to output_dir."""
        self._run_smoke(tmp_path, seed=3)
        md_path = tmp_path / "table1.md"
        assert md_path.exists(), "table1.md was not created"
        content = md_path.read_text(encoding="utf-8")
        assert "Proxy" in content

    def test_smoke_saves_plot(self, tmp_path: Path) -> None:
        """run_proxy_validation writes proxy_correlation.png to output_dir."""
        self._run_smoke(tmp_path, seed=4)
        png_path = tmp_path / "proxy_correlation.png"
        assert png_path.exists(), "proxy_correlation.png was not created"
        assert png_path.stat().st_size > 0

    def test_smoke_best_proxy_is_valid(self, tmp_path: Path) -> None:
        """best_proxy is one of the four known proxy names."""
        results = self._run_smoke(tmp_path, seed=5)
        assert results["best_proxy"] in {"decoder_params", "grad_norm", "snip", "near"}

    def test_smoke_configs_length(self, tmp_path: Path) -> None:
        """results['configs'] has the same length as results['ppls']."""
        results = self._run_smoke(tmp_path, seed=6)
        assert len(results["configs"]) == len(results["ppls"])
        assert len(results["ppls"]) <= 3  # smoke run, so at most 3 configs

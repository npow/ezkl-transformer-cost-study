"""pytest configuration and fixtures for zkml-nas tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (XGBoost training with full n_estimators=200). "
        "Run with --run-slow to include.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow-marked tests unless --run-slow is passed."""
    if config.getoption("--run-slow", default=False):
        return
    skip_slow = pytest.mark.skip(reason="Skipped slow XGBoost tests; use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-slow flag."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g., full XGBoost training with 200 estimators).",
    )

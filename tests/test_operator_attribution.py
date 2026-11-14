"""Tests for experiments/operator_attribution.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from experiments.operator_attribution import (  # noqa: E402  pylint: disable=C0413
    OpBreakdown,
    breakdown_table,
    compute_op_breakdown,
)
from proxy.features import TransformerConfig  # noqa: E402  pylint: disable=C0413


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gpt2_config() -> TransformerConfig:
    """Standard GPT-2 small config."""
    return TransformerConfig(
        d_model=768,
        n_heads=12,
        n_layers=12,
        seq_len=1024,
        ff_mult=4.0,
        activation="gelu",
        use_rotary=False,
        use_layernorm=True,
        vocab_size=50257,
        tie_embeddings=True,
    )


@pytest.fixture()
def linear_attn_config() -> TransformerConfig:
    """GPT-2 with linear (no softmax) attention."""
    return TransformerConfig(
        d_model=768,
        n_heads=12,
        n_layers=12,
        seq_len=1024,
        ff_mult=4.0,
        activation="linear",
        use_rotary=False,
        use_layernorm=True,
        vocab_size=50257,
        tie_embeddings=True,
    )


# ---------------------------------------------------------------------------
# compute_op_breakdown — structural validity
# ---------------------------------------------------------------------------


class TestComputeOpBreakdown:
    """Tests for compute_op_breakdown()."""

    def test_returns_op_breakdown_instance(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        result = compute_op_breakdown(gpt2_config)
        assert isinstance(result, OpBreakdown)

    def test_total_constraints_positive(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        assert bd.total_constraints > 0

    def test_all_constraint_fields_nonnegative(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        for field in (
            "attn_qkv_proj_constraints",
            "attn_score_constraints",
            "attn_value_constraints",
            "attn_out_proj_constraints",
            "ffn_up_constraints",
            "ffn_down_constraints",
            "embedding_constraints",
            "softmax_constraints",
            "activation_constraints",
            "layernorm_constraints",
            "rope_constraints",
        ):
            assert getattr(bd, field) >= 0, f"{field} should be >= 0"

    def test_fractions_sum_to_one(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        assert abs(bd.linear_fraction + bd.nonlinear_fraction - 1.0) < 1e-9

    def test_linear_fraction_in_unit_interval(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        assert 0.0 <= bd.linear_fraction <= 1.0

    def test_total_equals_sum_of_parts(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        parts_sum = (
            bd.attn_qkv_proj_constraints
            + bd.attn_score_constraints
            + bd.attn_value_constraints
            + bd.attn_out_proj_constraints
            + bd.ffn_up_constraints
            + bd.ffn_down_constraints
            + bd.embedding_constraints
            + bd.softmax_constraints
            + bd.activation_constraints
            + bd.layernorm_constraints
            + bd.rope_constraints
        )
        assert parts_sum == bd.total_constraints

    # ------------------------------------------------------------------
    # Key paper claim: matmuls dominate for GPT-2 with GELU
    # ------------------------------------------------------------------

    def test_linear_fraction_gt_95_for_gpt2_gelu(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        """Matmuls should account for >95% of ZK cost in standard GPT-2."""
        bd = compute_op_breakdown(gpt2_config)
        assert (
            bd.linear_fraction > 0.95
        ), f"Expected linear_fraction > 0.95 for GPT-2 (GELU); got {bd.linear_fraction:.4f}"

    # ------------------------------------------------------------------
    # Linear attention: no softmax
    # ------------------------------------------------------------------

    def test_softmax_zero_for_linear_attention(
        self,
        linear_attn_config: TransformerConfig,  # pylint: disable=redefined-outer-name
    ) -> None:
        """Linear attention models must have zero softmax cost."""
        bd = compute_op_breakdown(linear_attn_config)
        assert bd.softmax_constraints == 0

    # ------------------------------------------------------------------
    # RoPE models have positive rope_constraints
    # ------------------------------------------------------------------

    def test_rope_constraints_positive_when_use_rotary(self) -> None:
        cfg = TransformerConfig(
            d_model=512,
            n_heads=8,
            n_layers=4,
            seq_len=256,
            ff_mult=4.0,
            activation="poly2",
            use_rotary=True,
            use_layernorm=False,
            vocab_size=50257,
            tie_embeddings=True,
        )
        bd = compute_op_breakdown(cfg)
        assert bd.rope_constraints > 0

    def test_rope_constraints_zero_when_no_rotary(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        assert bd.rope_constraints == 0

    # ------------------------------------------------------------------
    # to_dict
    # ------------------------------------------------------------------

    def test_to_dict_is_json_serialisable(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        serialised = json.dumps(bd.to_dict())
        assert isinstance(serialised, str) and len(serialised) > 10

    def test_to_dict_contains_expected_keys(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        d = bd.to_dict()
        for key in (
            "total_constraints",
            "linear_fraction",
            "nonlinear_fraction",
            "softmax_constraints",
            "layernorm_constraints",
        ):
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# breakdown_table
# ---------------------------------------------------------------------------


class TestBreakdownTable:
    """Tests for breakdown_table()."""

    @pytest.fixture()
    def sample_breakdowns(
        self,
        gpt2_config: TransformerConfig,  # pylint: disable=redefined-outer-name
        linear_attn_config: TransformerConfig,  # pylint: disable=redefined-outer-name
    ) -> dict[str, OpBreakdown]:
        return {
            "GPT-2 baseline": compute_op_breakdown(gpt2_config),
            "Linear attn": compute_op_breakdown(linear_attn_config),
        }

    def test_returns_string(
        self,
        sample_breakdowns: dict[
            str, OpBreakdown
        ],  # pylint: disable=redefined-outer-name
    ) -> None:
        result = breakdown_table(sample_breakdowns)
        assert isinstance(result, str)

    def test_contains_expected_column_headers(
        self,
        sample_breakdowns: dict[
            str, OpBreakdown
        ],  # pylint: disable=redefined-outer-name
    ) -> None:
        result = breakdown_table(sample_breakdowns)
        for col in (
            "Architecture",
            "Total",
            "MatMul%",
            "Softmax%",
            "Activation%",
            "LayerNorm%",
            "RoPE%",
        ):
            assert col in result, f"Missing column header: {col!r}"

    def test_contains_architecture_names(
        self,
        sample_breakdowns: dict[
            str, OpBreakdown
        ],  # pylint: disable=redefined-outer-name
    ) -> None:
        result = breakdown_table(sample_breakdowns)
        for name in sample_breakdowns:
            assert name in result, f"Architecture name {name!r} not found in table"

    def test_row_count_matches_input(
        self,
        sample_breakdowns: dict[
            str, OpBreakdown
        ],  # pylint: disable=redefined-outer-name
    ) -> None:
        result = breakdown_table(sample_breakdowns)
        # header row + separator + one data row per architecture
        lines = [ln for ln in result.splitlines() if ln.strip()]
        n_data_rows = len(lines) - 2  # minus header and separator
        assert n_data_rows == len(sample_breakdowns)

    def test_separator_line_present(
        self,
        sample_breakdowns: dict[
            str, OpBreakdown
        ],  # pylint: disable=redefined-outer-name
    ) -> None:
        result = breakdown_table(sample_breakdowns)
        assert "|---|" in result

    def test_single_architecture_works(
        self, gpt2_config: TransformerConfig  # pylint: disable=redefined-outer-name
    ) -> None:
        bd = compute_op_breakdown(gpt2_config)
        result = breakdown_table({"GPT-2": bd})
        assert "GPT-2" in result
        assert "%" in result

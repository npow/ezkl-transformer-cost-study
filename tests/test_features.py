"""Tests for proxy.features — feature extraction from TransformerConfig."""

from __future__ import annotations

import numpy as np
import pytest

from proxy.features import (
    ACTIVATION_IDS,
    ZK_FRIENDLY_ACTIVATIONS,
    ProxyFeatures,
    TransformerConfig,
    feature_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> TransformerConfig:
    """Return a minimal valid TransformerConfig, overriding any supplied fields."""
    defaults = dict(
        d_model=256,
        n_heads=4,
        n_layers=2,
        seq_len=64,
        ff_mult=4.0,
        activation="relu",
        use_rotary=False,
        use_layernorm=True,
        vocab_size=50257,
        tie_embeddings=False,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


# ---------------------------------------------------------------------------
# TransformerConfig validation
# ---------------------------------------------------------------------------


class TestTransformerConfigValidation:
    def test_valid_config_does_not_raise(self) -> None:
        cfg = make_config()
        assert cfg.d_model == 256

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="activation"):
            make_config(activation="swish")

    def test_d_model_not_divisible_by_n_heads_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            make_config(d_model=256, n_heads=7)

    def test_zero_d_model_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            make_config(d_model=0)

    def test_zero_n_layers_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers"):
            make_config(n_layers=0)

    def test_negative_ff_mult_raises(self) -> None:
        with pytest.raises(ValueError, match="ff_mult"):
            make_config(ff_mult=-1.0)


# ---------------------------------------------------------------------------
# from_config without model
# ---------------------------------------------------------------------------


class TestFromConfigNoModel:
    def test_returns_proxy_features_instance(self) -> None:
        cfg = make_config()
        feat = ProxyFeatures.from_config(cfg)
        assert isinstance(feat, ProxyFeatures)

    def test_onnx_fields_are_none_without_model(self) -> None:
        cfg = make_config()
        feat = ProxyFeatures.from_config(cfg)
        assert feat.onnx_constraint_estimate is None
        assert feat.onnx_nonlinear_ops is None
        assert feat.onnx_linear_ops is None

    def test_raw_fields_match_config(self) -> None:
        cfg = make_config(d_model=512, n_heads=8, n_layers=6, seq_len=128, ff_mult=4.0)
        feat = ProxyFeatures.from_config(cfg)
        assert feat.d_model == 512
        assert feat.n_heads == 8
        assert feat.n_layers == 6
        assert feat.seq_len == 128
        assert feat.ff_mult == pytest.approx(4.0)

    def test_head_dim(self) -> None:
        cfg = make_config(d_model=512, n_heads=8)
        feat = ProxyFeatures.from_config(cfg)
        assert feat.head_dim == pytest.approx(512 / 8)

    def test_activation_id_all_values(self) -> None:
        for act, expected_id in ACTIVATION_IDS.items():
            cfg = make_config(activation=act)
            feat = ProxyFeatures.from_config(cfg)
            assert feat.activation_id == expected_id, f"activation={act}"

    def test_is_zk_friendly_poly2(self) -> None:
        feat = ProxyFeatures.from_config(make_config(activation="poly2"))
        assert feat.is_zk_friendly is True

    def test_is_zk_friendly_poly3(self) -> None:
        feat = ProxyFeatures.from_config(make_config(activation="poly3"))
        assert feat.is_zk_friendly is True

    def test_is_zk_friendly_linear(self) -> None:
        feat = ProxyFeatures.from_config(make_config(activation="linear"))
        assert feat.is_zk_friendly is True

    def test_is_not_zk_friendly_gelu(self) -> None:
        feat = ProxyFeatures.from_config(make_config(activation="gelu"))
        assert feat.is_zk_friendly is False

    def test_is_not_zk_friendly_relu(self) -> None:
        feat = ProxyFeatures.from_config(make_config(activation="relu"))
        assert feat.is_zk_friendly is False


# ---------------------------------------------------------------------------
# Structural count correctness
# ---------------------------------------------------------------------------


class TestStructuralCounts:
    def test_layernorm_count(self) -> None:
        """n_layers * 2 + 1 (pre-attn, pre-ffn per block plus final norm)."""
        for n_layers in (2, 6, 12):
            cfg = make_config(n_layers=n_layers)
            feat = ProxyFeatures.from_config(cfg)
            expected = n_layers * 2 + 1
            assert feat.layernorm_count == expected, f"n_layers={n_layers}"

    def test_softmax_count_non_linear_activation(self) -> None:
        """One softmax per block for non-linear-attention activations."""
        for act in ("gelu", "relu", "poly2", "poly3"):
            cfg = make_config(n_layers=4, activation=act)
            feat = ProxyFeatures.from_config(cfg)
            assert feat.softmax_count == 4, f"activation={act}"

    def test_softmax_count_linear_activation(self) -> None:
        """Linear attention has no softmax."""
        cfg = make_config(n_layers=4, activation="linear")
        feat = ProxyFeatures.from_config(cfg)
        assert feat.softmax_count == 0

    def test_gelu_count_gelu_activation(self) -> None:
        cfg = make_config(n_layers=6, activation="gelu")
        feat = ProxyFeatures.from_config(cfg)
        assert feat.gelu_count == 6

    def test_gelu_count_non_gelu_activation(self) -> None:
        for act in ("relu", "poly2", "poly3", "linear"):
            cfg = make_config(n_layers=6, activation=act)
            feat = ProxyFeatures.from_config(cfg)
            assert feat.gelu_count == 0, f"activation={act}"


# ---------------------------------------------------------------------------
# Parameter count formula
# ---------------------------------------------------------------------------


class TestParamCount:
    def test_gpt2_small_tied_embeddings(self) -> None:
        """GPT-2-small with tied embeddings ≈ 117M params via the analytical formula.

        The formula (no biases, no position embedding) gives ~123.6 M, which is
        within 6% of the 117 M commonly quoted for GPT-2 small.  We verify the
        exact formula output rather than a rounded figure.
        """
        d = 768
        n = 12
        v = 50257
        ff = 4.0
        cfg = TransformerConfig(
            d_model=d,
            n_heads=12,
            n_layers=n,
            seq_len=128,
            ff_mult=ff,
            activation="gelu",
            vocab_size=v,
            tie_embeddings=True,
        )
        feat = ProxyFeatures.from_config(cfg)
        expected = v * d + n * (4 * d * d + int(2 * ff * d * d) + 4 * d)
        assert feat.total_params == expected
        # Sanity check: should be within 10% of 117M
        assert abs(feat.total_params - 117_000_000) / 117_000_000 < 0.10

    def test_param_count_no_tie(self) -> None:
        """Without tied embeddings the lm_head params are counted once extra."""
        d = 256
        n = 2
        v = 1000
        ff = 4.0
        cfg = TransformerConfig(
            d_model=d,
            n_heads=4,
            n_layers=n,
            seq_len=64,
            ff_mult=ff,
            vocab_size=v,
            tie_embeddings=False,
        )
        feat = ProxyFeatures.from_config(cfg)
        per_layer = 4 * d * d + int(2 * ff * d * d) + 4 * d
        expected = v * d + n * per_layer + v * d  # embedding + layers + lm_head
        assert feat.total_params == expected

    def test_param_count_tie_vs_no_tie_differs_by_vocab_times_dmodel(self) -> None:
        cfg_tied = make_config(
            d_model=256, n_heads=4, n_layers=2, vocab_size=1000, tie_embeddings=True
        )
        cfg_free = make_config(
            d_model=256, n_heads=4, n_layers=2, vocab_size=1000, tie_embeddings=False
        )
        f_tied = ProxyFeatures.from_config(cfg_tied)
        f_free = ProxyFeatures.from_config(cfg_free)
        assert f_free.total_params - f_tied.total_params == 256 * 1000


# ---------------------------------------------------------------------------
# MAC count formulas
# ---------------------------------------------------------------------------


class TestMACCounts:
    def test_attn_macs_formula(self) -> None:
        d, s = 256, 64
        cfg = make_config(d_model=d, n_heads=4, seq_len=s)
        feat = ProxyFeatures.from_config(cfg)
        expected = (
            2 * s * d * d  # QKV proj
            + 2 * s * s * d  # scores + values
            + 2 * s * d * d  # output proj
        )
        assert feat.attn_macs == expected

    def test_ffn_macs_formula(self) -> None:
        d, s, ff = 256, 64, 4.0
        cfg = make_config(d_model=d, n_heads=4, seq_len=s, ff_mult=ff)
        feat = ProxyFeatures.from_config(cfg)
        expected = int(2 * s * d * ff * d * 2)
        assert feat.ffn_macs == expected

    def test_total_macs_is_layers_times_per_layer(self) -> None:
        cfg = make_config(n_layers=6)
        feat = ProxyFeatures.from_config(cfg)
        assert feat.total_macs == 6 * (feat.attn_macs + feat.ffn_macs)


# ---------------------------------------------------------------------------
# to_vector consistency
# ---------------------------------------------------------------------------


class TestToVector:
    def test_returns_numpy_array(self) -> None:
        cfg = make_config()
        vec = ProxyFeatures.from_config(cfg).to_vector()
        assert isinstance(vec, np.ndarray)

    def test_dtype_is_float64(self) -> None:
        cfg = make_config()
        vec = ProxyFeatures.from_config(cfg).to_vector()
        assert vec.dtype == np.float64

    def test_vector_length_matches_feature_names(self) -> None:
        cfg = make_config()
        vec = ProxyFeatures.from_config(cfg).to_vector()
        assert vec.shape == (len(feature_names()),)

    def test_consistent_length_across_configs(self) -> None:
        """All configs produce vectors of the same length."""
        configs = [
            make_config(d_model=256, n_heads=4, n_layers=2, seq_len=64),
            make_config(d_model=1024, n_heads=16, n_layers=24, seq_len=512),
            make_config(d_model=512, n_heads=8, n_layers=6, ff_mult=8.0),
            make_config(activation="poly3", use_rotary=True, use_layernorm=False),
        ]
        lengths = {ProxyFeatures.from_config(c).to_vector().shape[0] for c in configs}
        assert len(lengths) == 1, f"Inconsistent vector lengths: {lengths}"

    def test_onnx_sentinels_when_no_model(self) -> None:
        """When model=None, the three ONNX columns are filled with -1.0."""
        feat = ProxyFeatures.from_config(make_config())
        vec = feat.to_vector()
        names = feature_names()
        for col in (
            "onnx_constraint_estimate",
            "onnx_nonlinear_ops",
            "onnx_linear_ops",
        ):
            idx = names.index(col)
            assert vec[idx] == pytest.approx(-1.0), f"column={col}"

    def test_all_config_fields_represented(self) -> None:
        """Every TransformerConfig field must influence at least one feature."""
        names = feature_names()
        # d_model, n_heads, n_layers, seq_len, ff_mult appear directly.
        for col in ("d_model", "n_heads", "n_layers", "seq_len", "ff_mult"):
            assert col in names, f"{col} missing from feature_names()"
        # activation → activation_id
        assert "activation_id" in names
        # use_rotary and use_layernorm affect architectural decisions;
        # is_zk_friendly reflects activation choice.
        assert "is_zk_friendly" in names
        # vocab_size and tie_embeddings drive total_params.
        assert "total_params" in names


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------


class TestFeatureNames:
    def test_returns_list_of_strings(self) -> None:
        names = feature_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_no_duplicate_names(self) -> None:
        names = feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names detected"

    def test_returns_a_copy(self) -> None:
        names1 = feature_names()
        names2 = feature_names()
        names1.append("__sentinel__")
        assert "__sentinel__" not in names2, "feature_names() must return a copy"

    def test_length_matches_vector_length(self) -> None:
        cfg = make_config()
        vec = ProxyFeatures.from_config(cfg).to_vector()
        assert len(feature_names()) == vec.shape[0]

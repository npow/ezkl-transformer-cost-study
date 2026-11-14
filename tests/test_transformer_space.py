"""Tests for search.transformer_space — ZK-NAS transformer search space."""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
import torch

from proxy.features import TransformerConfig
from search.transformer_space import (
    SEARCH_SPACE,
    LinearActivation,
    Poly2Activation,
    Poly3Activation,
    RMSNorm,
    TransformerBlock,
    TransformerSearchSpace,
    ZKTransformer,
    _ZK_FRIENDLY_ACTIVATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_config(**kwargs) -> TransformerConfig:
    """Return a small valid config, overriding any supplied fields."""
    defaults = dict(
        d_model=64,
        n_heads=4,
        n_layers=2,
        seq_len=16,
        ff_mult=2.0,
        activation="gelu",
        use_rotary=False,
        use_layernorm=True,
        vocab_size=128,
        tie_embeddings=False,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


# ---------------------------------------------------------------------------
# Activation modules
# ---------------------------------------------------------------------------


class TestActivationModules:
    def test_poly2_output(self) -> None:
        act = Poly2Activation()
        x = torch.tensor([2.0, -3.0])
        result = act(x)
        expected = torch.tensor([4.0, 9.0])
        torch.testing.assert_close(result, expected)

    def test_poly3_output(self) -> None:
        act = Poly3Activation()
        x = torch.tensor([2.0, -1.0])
        result = act(x)
        # f(2) = 8 + 2 = 10, f(-1) = -1 + (-1) = -2
        expected = torch.tensor([10.0, -2.0])
        torch.testing.assert_close(result, expected)

    def test_linear_output(self) -> None:
        act = LinearActivation()
        x = torch.tensor([3.0, -5.0, 0.0])
        torch.testing.assert_close(act(x), x)

    def test_poly2_is_nn_module(self) -> None:
        assert isinstance(Poly2Activation(), torch.nn.Module)

    def test_poly3_is_nn_module(self) -> None:
        assert isinstance(Poly3Activation(), torch.nn.Module)

    def test_linear_is_nn_module(self) -> None:
        assert isinstance(LinearActivation(), torch.nn.Module)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self) -> None:
        norm = RMSNorm(d_model=32)
        x = torch.randn(2, 8, 32)
        assert norm(x).shape == x.shape

    def test_output_is_normalised(self) -> None:
        """After RMSNorm, x / rms should be ~unit-rms (before weight scaling)."""
        torch.manual_seed(0)
        norm = RMSNorm(d_model=64)
        x = torch.randn(4, 16, 64) * 5.0
        out = norm(x)
        # With weight=1, RMS of output should be ~1 per token
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert (rms - 1.0).abs().max().item() < 0.05


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    @pytest.mark.parametrize("activation", ["gelu", "relu", "poly2", "poly3", "linear"])
    def test_output_shape_all_activations(self, activation: str) -> None:
        cfg = _small_config(activation=activation)
        block = TransformerBlock(cfg)
        x = torch.randn(2, cfg.seq_len, cfg.d_model)
        out = block(x)
        assert out.shape == x.shape

    def test_rmsnorm_variant(self) -> None:
        cfg = _small_config(use_layernorm=False)
        block = TransformerBlock(cfg)
        x = torch.randn(1, cfg.seq_len, cfg.d_model)
        out = block(x)
        assert out.shape == x.shape

    def test_rotary_variant(self) -> None:
        cfg = _small_config(use_rotary=True)
        block = TransformerBlock(cfg)
        x = torch.randn(1, cfg.seq_len, cfg.d_model)
        out = block(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# ZKTransformer forward pass
# ---------------------------------------------------------------------------


class TestZKTransformerForward:
    @pytest.mark.parametrize("activation", ["gelu", "relu", "poly2", "poly3", "linear"])
    def test_output_shape_all_activations(self, activation: str) -> None:
        cfg = _small_config(activation=activation)
        model = ZKTransformer(cfg)
        model.eval()
        ids = torch.zeros(2, cfg.seq_len, dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)

    def test_output_shape_rotary(self) -> None:
        cfg = _small_config(use_rotary=True)
        model = ZKTransformer(cfg)
        model.eval()
        ids = torch.zeros(1, cfg.seq_len, dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, cfg.seq_len, cfg.vocab_size)

    def test_output_shape_rmsnorm(self) -> None:
        cfg = _small_config(use_layernorm=False)
        model = ZKTransformer(cfg)
        model.eval()
        ids = torch.zeros(1, cfg.seq_len, dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, cfg.seq_len, cfg.vocab_size)

    def test_tied_embeddings_weight_sharing(self) -> None:
        cfg = _small_config(tie_embeddings=True)
        model = ZKTransformer(cfg)
        assert model.lm_head.weight is model.token_emb.weight

    def test_untied_embeddings(self) -> None:
        cfg = _small_config(tie_embeddings=False)
        model = ZKTransformer(cfg)
        assert model.lm_head.weight is not model.token_emb.weight

    def test_get_sample_input_shape(self) -> None:
        cfg = _small_config()
        model = ZKTransformer(cfg)
        sample = model.get_sample_input(batch_size=3)
        assert sample.shape == (3, cfg.seq_len)
        assert sample.dtype == torch.long

    def test_deterministic_eval(self) -> None:
        """Eval-mode forward should be deterministic."""
        cfg = _small_config()
        model = ZKTransformer(cfg)
        model.eval()
        ids = torch.zeros(1, cfg.seq_len, dtype=torch.long)
        with torch.no_grad():
            out1 = model(ids)
            out2 = model(ids)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ["gelu", "poly2"])
def test_onnx_export(activation: str) -> None:
    """ZKTransformer should be exportable to ONNX without error."""
    onnx = pytest.importorskip("onnx")
    cfg = _small_config(activation=activation, use_rotary=False)
    model = ZKTransformer(cfg)
    model.eval()
    sample = model.get_sample_input(batch_size=1)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (sample,),
        buf,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    )
    buf.seek(0)
    onnx_model = onnx.load(buf)
    onnx.checker.check_model(onnx_model)


# ---------------------------------------------------------------------------
# TransformerSearchSpace
# ---------------------------------------------------------------------------


class TestTransformerSearchSpaceValidity:
    def test_is_valid_gpt2_baseline(self) -> None:
        ss = TransformerSearchSpace()
        cfg = ss.gpt2_baseline()
        assert ss.is_valid(cfg)

    def test_is_valid_rejects_bad_d_model_n_heads(self) -> None:
        # TransformerConfig.__post_init__ raises for d_model not divisible by n_heads;
        # is_valid should also return False for such configs (tested via object mutation).
        ss = TransformerSearchSpace()
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(d_model=256, n_heads=7, n_layers=2, seq_len=64)
        # Verify is_valid returns False for a mutated config with bad divisibility
        cfg = TransformerConfig(d_model=256, n_heads=8, n_layers=2, seq_len=64)
        cfg.__dict__["n_heads"] = 7  # bypass validation to test is_valid directly
        assert not ss.is_valid(cfg)

    def test_is_valid_rejects_out_of_space_d_model(self) -> None:
        ss = TransformerSearchSpace()
        # Build config bypassing __post_init__ d_model check
        cfg = TransformerConfig(
            d_model=512, n_heads=8, n_layers=2, seq_len=64, vocab_size=50257
        )
        cfg_bad = TransformerConfig.__new__(TransformerConfig)
        cfg_bad.__dict__.update(cfg.__dict__)
        cfg_bad.__dict__["d_model"] = 999  # not in search space
        assert not ss.is_valid(cfg_bad)


class TestRandomConfig:
    def test_random_config_is_valid(self) -> None:
        ss = TransformerSearchSpace(seed=0)
        for _ in range(50):
            cfg = ss.random_config()
            assert ss.is_valid(cfg), f"Invalid config sampled: {cfg}"

    def test_random_config_has_correct_vocab(self) -> None:
        ss = TransformerSearchSpace(vocab_size=32000, seed=7)
        cfg = ss.random_config()
        assert cfg.vocab_size == 32000

    def test_random_config_divisibility(self) -> None:
        ss = TransformerSearchSpace(seed=1)
        for _ in range(20):
            cfg = ss.random_config()
            assert cfg.d_model % cfg.n_heads == 0


class TestGpt2Baseline:
    def test_d_model(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.gpt2_baseline().d_model == 768

    def test_n_heads(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.gpt2_baseline().n_heads == 12

    def test_n_layers(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.gpt2_baseline().n_layers == 12

    def test_activation(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.gpt2_baseline().activation == "gelu"

    def test_seq_len(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.gpt2_baseline().seq_len == 1024

    def test_vocab_size(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.gpt2_baseline().vocab_size == 50257


class TestBuildModel:
    def test_returns_zk_transformer(self) -> None:
        ss = TransformerSearchSpace()
        cfg = _small_config()
        model = ss.build_model(cfg)
        assert isinstance(model, ZKTransformer)

    def test_model_is_in_eval_mode(self) -> None:
        ss = TransformerSearchSpace()
        cfg = _small_config()
        model = ss.build_model(cfg)
        assert not model.training

    def test_output_shape_matches_config(self) -> None:
        ss = TransformerSearchSpace()
        cfg = _small_config(
            d_model=128, n_heads=4, n_layers=2, seq_len=32, vocab_size=256
        )
        model = ss.build_model(cfg)
        ids = torch.zeros(1, cfg.seq_len, dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, cfg.seq_len, cfg.vocab_size)


class TestGridConfigs:
    def test_returns_list(self) -> None:
        ss = TransformerSearchSpace()
        cfgs = ss.grid_configs(max_configs=10)
        assert isinstance(cfgs, list)

    def test_max_configs_respected(self) -> None:
        ss = TransformerSearchSpace()
        cfgs = ss.grid_configs(max_configs=50)
        assert len(cfgs) <= 50

    def test_all_valid(self) -> None:
        ss = TransformerSearchSpace()
        for cfg in ss.grid_configs(max_configs=200):
            assert ss.is_valid(cfg), f"Invalid config in grid: {cfg}"

    def test_no_duplicates(self) -> None:
        ss = TransformerSearchSpace()
        cfgs = ss.grid_configs(max_configs=200)
        keys = [
            (
                c.d_model,
                c.n_heads,
                c.n_layers,
                c.seq_len,
                c.ff_mult,
                c.activation,
                c.use_rotary,
                c.use_layernorm,
                c.tie_embeddings,
            )
            for c in cfgs
        ]
        assert len(keys) == len(set(keys)), "Duplicate configs in grid"

    def test_covers_zk_friendly(self) -> None:
        ss = TransformerSearchSpace()
        cfgs = ss.grid_configs(max_configs=500)
        zk_acts = {
            c.activation for c in cfgs if c.activation in _ZK_FRIENDLY_ACTIVATIONS
        }
        assert len(zk_acts) >= 2, "Grid should include ZK-friendly activations"


class TestZkFriendlyCandidates:
    def test_returns_n_configs(self) -> None:
        ss = TransformerSearchSpace(seed=42)
        cfgs = ss.zk_friendly_candidates(n=20)
        assert len(cfgs) == 20

    def test_all_zk_friendly_activations(self) -> None:
        ss = TransformerSearchSpace(seed=0)
        cfgs = ss.zk_friendly_candidates(n=50)
        for cfg in cfgs:
            assert (
                cfg.activation in _ZK_FRIENDLY_ACTIVATIONS
            ), f"Non-ZK-friendly activation {cfg.activation!r} in zk_friendly_candidates"

    def test_small_d_model_bias(self) -> None:
        ss = TransformerSearchSpace(seed=3)
        cfgs = ss.zk_friendly_candidates(n=50)
        large_d = sum(1 for c in cfgs if c.d_model > 512)
        # Most configs should use d_model ≤ 512
        assert large_d == 0, "zk_friendly_candidates should not return d_model > 512"

    def test_all_valid(self) -> None:
        ss = TransformerSearchSpace(seed=5)
        for cfg in ss.zk_friendly_candidates(n=30):
            assert ss.is_valid(cfg)


class TestEstimateZkCost:
    def test_returns_dict_with_required_keys(self) -> None:
        ss = TransformerSearchSpace()
        cfg = _small_config(activation="poly2")
        cost = ss.estimate_zk_cost(cfg)
        for key in ("constraint_estimate", "nonlinear_ops", "linear_ops"):
            assert key in cost, f"Missing key {key!r} in estimate_zk_cost result"

    def test_constraint_estimate_positive_poly2(self) -> None:
        ss = TransformerSearchSpace()
        cfg = _small_config(
            activation="poly2", d_model=64, n_heads=4, n_layers=2, seq_len=16
        )
        cost = ss.estimate_zk_cost(cfg)
        assert (
            cost["constraint_estimate"] > 0
        ), "constraint_estimate should be positive for a non-trivial model"

    def test_constraint_estimate_positive_gelu(self) -> None:
        ss = TransformerSearchSpace()
        cfg = _small_config(
            activation="gelu", d_model=64, n_heads=4, n_layers=2, seq_len=16
        )
        cost = ss.estimate_zk_cost(cfg)
        assert cost["constraint_estimate"] > 0

    def test_gpt2_baseline_higher_cost_than_poly2_small(self) -> None:
        """GPT-2 baseline should have strictly higher ZK cost than a tiny poly2 model."""
        ss = TransformerSearchSpace()
        gpt2_cost = ss.estimate_zk_cost(ss.gpt2_baseline())
        small_poly2_cfg = _small_config(
            activation="poly2",
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=16,
            vocab_size=50257,
        )
        small_cost = ss.estimate_zk_cost(small_poly2_cfg)
        assert (
            gpt2_cost["constraint_estimate"] > small_cost["constraint_estimate"]
        ), "GPT-2 baseline should have higher ZK cost than a small poly2 model"

    def test_no_exception_on_failure(self) -> None:
        """estimate_zk_cost should warn rather than raise on proxy failure."""
        ss = TransformerSearchSpace()
        cfg = _small_config()
        # Should not raise regardless of zkml_bench availability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cost = ss.estimate_zk_cost(cfg)
        assert isinstance(cost, dict)


class TestNConfigs:
    def test_n_configs_positive(self) -> None:
        ss = TransformerSearchSpace()
        assert ss.n_configs() > 0

    def test_n_configs_accounts_for_divisibility(self) -> None:
        """n_configs should exclude (d_model, n_heads) pairs where d_model % n_heads != 0."""
        ss = TransformerSearchSpace()
        total = ss.n_configs()
        valid_pairs = sum(
            1
            for d in SEARCH_SPACE["d_model"]
            for h in SEARCH_SPACE["n_heads"]
            if d % h == 0
        )
        other = (
            len(SEARCH_SPACE["n_layers"])
            * len(SEARCH_SPACE["seq_len"])
            * len(SEARCH_SPACE["ff_mult"])
            * len(SEARCH_SPACE["activation"])
            * len(SEARCH_SPACE["use_rotary"])
            * len(SEARCH_SPACE["use_layernorm"])
            * len(SEARCH_SPACE["tie_embeddings"])
        )
        assert total == valid_pairs * other

    def test_n_configs_large(self) -> None:
        """The search space should be meaningfully large (> 10_000 configs)."""
        ss = TransformerSearchSpace()
        assert ss.n_configs() > 10_000


import pytest as _pytest_causal
import torch as _torch_causal


@_pytest_causal.mark.parametrize("activation", ["gelu", "relu", "poly2", "linear"])
def test_model_is_causal(activation: str) -> None:
    """Changing a future token must not change logits at earlier positions."""
    from proxy.features import TransformerConfig as _Cfg
    from search.transformer_space import ZKTransformer as _Model

    cfg = _Cfg(64, 4, 2, 16, 4.0, activation, False, True, 100, True)
    model = _Model(cfg).eval()
    x = _torch_causal.randint(0, 100, (1, 16))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 100
    with _torch_causal.no_grad():
        a, b = model(x), model(y)
    assert _torch_causal.allclose(a[:, :-1], b[:, :-1], atol=1e-6)


@_pytest_causal.mark.parametrize("rotary", [False, True])
@_pytest_causal.mark.parametrize("linear_attn", [None, True, False])
@_pytest_causal.mark.parametrize("activation", ["gelu", "relu", "linear"])
def test_model_is_causal_all_variants(activation: str, linear_attn, rotary: bool) -> None:
    """Causality must hold with rotary on/off and with attention decoupled from the FFN
    (the decomposition variants used in the paper)."""
    from proxy.features import TransformerConfig as _Cfg
    from search.transformer_space import ZKTransformer as _Model

    cfg = _Cfg(64, 4, 2, 16, 4.0, activation, rotary, True, 100, True)
    model = _Model(cfg).eval()
    if linear_attn is not None:
        for blk in model.blocks:
            blk.attn.use_linear_attn = linear_attn
    x = _torch_causal.randint(0, 100, (2, 16))
    for pos in (15, 8):  # perturb the last token and a middle token
        y = x.clone()
        y[:, pos] = (y[:, pos] + 1) % 100
        with _torch_causal.no_grad():
            a, b = model(x), model(y)
        assert _torch_causal.allclose(a[:, :pos], b[:, :pos], atol=1e-6)

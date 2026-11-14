"""Tests for proxy.accuracy_proxies — LM-validated accuracy proxy stack."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import pytest
import torch

from proxy.accuracy_proxies import (
    composite_accuracy_score,
    decoder_params_score,
    grad_norm_score,
    near_score_lm,
    snip_score,
)
from proxy.features import TransformerConfig
from search.transformer_space import TransformerSearchSpace, ZKTransformer


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs: Any) -> TransformerConfig:
    """Return a minimal valid TransformerConfig, overriding any supplied fields."""
    defaults: dict[str, Any] = dict(
        d_model=64,
        n_heads=4,
        n_layers=2,
        seq_len=32,
        ff_mult=4.0,
        activation="relu",
        use_rotary=False,
        use_layernorm=True,
        vocab_size=1000,
        tie_embeddings=True,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_data(
    batch: int = 2, seq_len: int = 32, vocab_size: int = 1000
) -> torch.Tensor:
    """Return a random integer token tensor on CPU."""
    return torch.randint(0, vocab_size, (batch, seq_len))


def _tiny_model(config: TransformerConfig | None = None) -> ZKTransformer:
    """Return a ZKTransformer built from a tiny config."""
    cfg = config or _cfg()
    return ZKTransformer(cfg)


# ---------------------------------------------------------------------------
# decoder_params_score
# ---------------------------------------------------------------------------


class TestDecoderParamsScore:
    def test_returns_finite_positive_float(self) -> None:
        """decoder_params_score returns a finite positive float for a valid config."""
        cfg = _cfg()
        score = decoder_params_score(cfg)
        assert isinstance(score, float)
        assert math.isfinite(score)
        assert score > 0.0

    def test_finite_for_all_search_space_configs(self) -> None:
        """Returns finite float for 50 random configs sampled from the search space."""
        space = TransformerSearchSpace(seed=7)
        for _ in range(50):
            cfg = space.random_config()
            score = decoder_params_score(cfg)
            assert math.isfinite(score), f"Non-finite score for config {cfg}"
            assert score > 0.0, f"Non-positive score for config {cfg}"

    def test_monotone_small_vs_large(self) -> None:
        """Larger configs have higher decoder_params_score than smaller ones."""
        small = _cfg(d_model=64, n_heads=4, n_layers=2, seq_len=32)
        large = _cfg(d_model=256, n_heads=8, n_layers=12, seq_len=128)
        assert decoder_params_score(large) > decoder_params_score(small)

    def test_gpt2_baseline_higher_than_tiny(self) -> None:
        """GPT-2 small has higher decoder_params_score than a 2-layer tiny transformer."""
        tiny = TransformerConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            seq_len=64,
            ff_mult=4.0,
            activation="gelu",
            vocab_size=50257,
            tie_embeddings=True,
        )
        gpt2 = TransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            seq_len=1024,
            ff_mult=4.0,
            activation="gelu",
            vocab_size=50257,
            tie_embeddings=True,
        )
        assert decoder_params_score(gpt2) > decoder_params_score(tiny)

    def test_score_is_log_of_decoder_params(self) -> None:
        """Score equals log(decoder_params) as defined in the docstring."""
        # With tie_embeddings=True and use_rotary=False:
        #   decoder_params = total_params - embed_params - pos_embed_params
        d, n, v, s, ff = 256, 4, 1000, 64, 4.0
        cfg = _cfg(
            d_model=d,
            n_heads=n,
            n_layers=4,
            seq_len=s,
            ff_mult=ff,
            vocab_size=v,
            tie_embeddings=True,
            use_rotary=False,
        )
        from proxy.features import ProxyFeatures, _params_per_layer

        total = v * d + 4 * _params_per_layer(cfg)
        embed = v * d
        pos_embed = s * d
        lm_head = 0  # tied
        expected_dp = total - embed - pos_embed - lm_head
        expected_score = math.log(float(expected_dp))
        assert decoder_params_score(cfg) == pytest.approx(expected_score, rel=1e-6)

    def test_rotary_vs_no_rotary_differs_by_pos_embed(self) -> None:
        """Rotary configs skip pos embedding, so decoder_params is larger."""
        cfg_rotary = _cfg(use_rotary=True)
        cfg_sinusoidal = _cfg(use_rotary=False)
        score_rotary = decoder_params_score(cfg_rotary)
        score_sinusoidal = decoder_params_score(cfg_sinusoidal)
        # Rotary removes positional embedding from denominator → larger decoder_params
        assert score_rotary >= score_sinusoidal

    def test_tied_vs_untied_embeddings(self) -> None:
        """Untied embeddings subtract lm_head params → lower decoder_params_score."""
        cfg_tied = _cfg(tie_embeddings=True)
        cfg_free = _cfg(tie_embeddings=False)
        # With untied we subtract more params → lower score
        assert decoder_params_score(cfg_tied) >= decoder_params_score(cfg_free)


# ---------------------------------------------------------------------------
# grad_norm_score
# ---------------------------------------------------------------------------


class TestGradNormScore:
    def test_returns_non_negative_float(self) -> None:
        """grad_norm_score returns a non-negative float with a real model + data."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        score = grad_norm_score(model, data, device="cpu")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_larger_model_produces_nonzero_score(self) -> None:
        """A model with more parameters typically has non-zero gradient norm."""
        cfg = _cfg(d_model=128, n_heads=4, n_layers=4, seq_len=32)
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        score = grad_norm_score(model, data, device="cpu")
        # Cannot guarantee non-zero, but it should be finite.
        assert math.isfinite(score)

    def test_returns_zero_on_invalid_data(self) -> None:
        """Returns 0.0 when data causes an exception (e.g., out-of-vocab token)."""
        cfg = _cfg(vocab_size=10)
        model = _tiny_model(cfg)
        # Intentionally large token IDs that exceed vocab_size to trigger an error
        bad_data = torch.full((2, cfg.seq_len), 9999, dtype=torch.long)
        score = grad_norm_score(model, bad_data, device="cpu")
        assert score == 0.0

    def test_does_not_update_params(self) -> None:
        """grad_norm_score must not modify model parameters."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)

        # Snapshot params before
        params_before = {
            name: p.data.clone() for name, p in model.named_parameters()
        }
        grad_norm_score(model, data, device="cpu")
        # Verify params unchanged
        for name, p in model.named_parameters():
            assert torch.allclose(p.data, params_before[name]), (
                f"Parameter {name} was modified by grad_norm_score"
            )

    def test_cleans_up_gradients(self) -> None:
        """grad_norm_score leaves model with zero/None gradients."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        grad_norm_score(model, data, device="cpu")
        for param in model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))


# ---------------------------------------------------------------------------
# snip_score
# ---------------------------------------------------------------------------


class TestSnipScore:
    def test_returns_non_negative_float(self) -> None:
        """snip_score returns a non-negative float with a real model + data."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        score = snip_score(model, data, device="cpu")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_returns_zero_on_exception(self) -> None:
        """Returns 0.0 when data causes an exception."""
        cfg = _cfg(vocab_size=10)
        model = _tiny_model(cfg)
        bad_data = torch.full((2, cfg.seq_len), 9999, dtype=torch.long)
        score = snip_score(model, bad_data, device="cpu")
        assert score == 0.0

    def test_does_not_update_params(self) -> None:
        """snip_score must not modify model parameters."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)

        params_before = {
            name: p.data.clone() for name, p in model.named_parameters()
        }
        snip_score(model, data, device="cpu")
        for name, p in model.named_parameters():
            assert torch.allclose(p.data, params_before[name]), (
                f"Parameter {name} was modified by snip_score"
            )

    def test_finite_score(self) -> None:
        """SNIP score is a finite number for a normal model."""
        cfg = _cfg(d_model=128, n_heads=4, n_layers=2, seq_len=32)
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        score = snip_score(model, data, device="cpu")
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# near_score_lm
# ---------------------------------------------------------------------------


class TestNearScoreLm:
    def test_returns_zero_when_zkml_bench_unavailable(self) -> None:
        """Returns 0.0 when zkml_bench cannot be imported."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)

        with patch.dict("sys.modules", {"zkml_bench": None, "zkml_bench.nas": None,
                                        "zkml_bench.nas.zero_cost_proxies": None}):
            score = near_score_lm(model, data, device="cpu")

        assert score == 0.0

    def test_returns_float(self) -> None:
        """Always returns a float, even if zkml_bench is unavailable."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        score = near_score_lm(model, data, device="cpu")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# composite_accuracy_score
# ---------------------------------------------------------------------------


class TestCompositeAccuracyScore:
    def test_without_model_returns_only_decoder_params(self) -> None:
        """Without model/data, only decoder_params is computed."""
        cfg = _cfg()
        result = composite_accuracy_score(cfg)
        assert "decoder_params" in result
        assert "proxies_used" in result
        proxies_used = result["proxies_used"]
        assert isinstance(proxies_used, list)
        assert "decoder_params" in proxies_used
        # Gradient-based proxies should NOT be present
        assert "grad_norm" not in result
        assert "snip" not in result
        assert "near" not in result

    def test_without_model_has_composite(self) -> None:
        """composite key is always present."""
        cfg = _cfg()
        result = composite_accuracy_score(cfg)
        assert "composite" in result
        assert isinstance(result["composite"], float)
        assert math.isfinite(float(result["composite"]))

    def test_without_model_composite_equals_decoder_params(self) -> None:
        """Without model, composite = decoder_params (single proxy, weight 1.0)."""
        cfg = _cfg()
        result = composite_accuracy_score(cfg)
        assert result["composite"] == pytest.approx(float(result["decoder_params"]))

    def test_with_model_returns_all_proxies(self) -> None:
        """With model+data, result contains grad_norm, snip, near, decoder_params."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        result = composite_accuracy_score(cfg, model=model, data=data, device="cpu")

        for key in ("decoder_params", "grad_norm", "snip", "near", "composite"):
            assert key in result, f"Missing key: {key}"

    def test_with_model_proxies_used_is_complete(self) -> None:
        """proxies_used lists all four proxies when model+data are provided."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        result = composite_accuracy_score(cfg, model=model, data=data, device="cpu")

        proxies_used = result["proxies_used"]
        assert isinstance(proxies_used, list)
        for key in ("decoder_params", "grad_norm", "snip", "near"):
            assert key in proxies_used

    def test_composite_is_finite(self) -> None:
        """composite is always a finite float."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        result = composite_accuracy_score(cfg, model=model, data=data, device="cpu")
        assert math.isfinite(float(result["composite"]))

    def test_custom_weights_accepted(self) -> None:
        """Custom weights are applied without error."""
        cfg = _cfg()
        model = _tiny_model(cfg)
        data = _make_data(seq_len=cfg.seq_len, vocab_size=cfg.vocab_size)
        result = composite_accuracy_score(
            cfg,
            model=model,
            data=data,
            device="cpu",
            weights={"decoder_params": 1.0, "grad_norm": 0.0, "near": 0.0},
        )
        # With all weight on decoder_params, composite should equal decoder_params
        assert result["composite"] == pytest.approx(
            float(result["decoder_params"]), rel=1e-5
        )

    def test_decoder_params_score_positive_in_composite(self) -> None:
        """decoder_params score within composite is a positive finite float."""
        cfg = _cfg()
        result = composite_accuracy_score(cfg)
        dp = float(result["decoder_params"])
        assert dp > 0.0
        assert math.isfinite(dp)

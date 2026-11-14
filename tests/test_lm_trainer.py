"""Tests for search.lm_trainer — WikiText-2 language model training."""

from __future__ import annotations

import math

import torch
from torch import nn

from proxy.features import TransformerConfig
from search.lm_trainer import (
    LMTrainer,
    LMTrainingConfig,
    LMTrainingResult,
    WikiText2Dataset,
    load_pretrained_gpt2,
)
from search.transformer_space import ZKTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_CONFIG = TransformerConfig(
    d_model=64,
    n_heads=4,
    n_layers=2,
    seq_len=64,
    ff_mult=2.0,
    activation="gelu",
    use_rotary=False,
    use_layernorm=True,
    vocab_size=50257,
    tie_embeddings=False,
)

_TINY_TRAIN_CFG = LMTrainingConfig(
    max_steps=10,
    eval_steps=5,
    batch_size=2,
    seq_len=64,
    lr=3e-4,
    warmup_steps=2,
    weight_decay=0.1,
    grad_clip=1.0,
    device="cpu",
    seed=42,
)


def _make_tiny_model() -> ZKTransformer:
    """Return a tiny ZKTransformer for fast CPU tests."""
    return ZKTransformer(_TINY_CONFIG)


# ---------------------------------------------------------------------------
# WikiText2Dataset
# ---------------------------------------------------------------------------


class TestWikiText2Dataset:
    """Tests for WikiText2Dataset loading and shape correctness."""

    def test_train_split_loads(self) -> None:
        """Dataset loads without error for the train split."""
        ds = WikiText2Dataset(split="train", seq_len=64)
        assert len(ds) > 0

    def test_validation_split_loads(self) -> None:
        """Dataset loads without error for the validation split."""
        ds = WikiText2Dataset(split="validation", seq_len=64)
        assert len(ds) > 0

    def test_item_shape(self) -> None:
        """Each item has shape (seq_len,)."""
        seq_len = 64
        ds = WikiText2Dataset(split="validation", seq_len=seq_len)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (seq_len,)

    def test_item_dtype_is_long(self) -> None:
        """Items are integer token IDs (dtype=torch.long)."""
        ds = WikiText2Dataset(split="validation", seq_len=64)
        assert ds[0].dtype == torch.long

    def test_tokens_in_vocab_range(self) -> None:
        """All token IDs are in [0, 50257)."""
        ds = WikiText2Dataset(split="validation", seq_len=64)
        item = ds[0]
        assert int(item.min()) >= 0
        assert int(item.max()) < 50257

    def test_multiple_seq_lens(self) -> None:
        """Different seq_len values produce items of the matching shape."""
        for seq_len in (64, 128):
            ds = WikiText2Dataset(split="validation", seq_len=seq_len)
            assert ds[0].shape == (seq_len,)

    def test_cache_reuse(self, tmp_path) -> None:
        """Loading twice with the same cache_dir reuses cached tokens."""
        ds1 = WikiText2Dataset(split="validation", seq_len=64, cache_dir=str(tmp_path))
        ds2 = WikiText2Dataset(split="validation", seq_len=64, cache_dir=str(tmp_path))
        assert len(ds1) == len(ds2)
        torch.testing.assert_close(ds1[0], ds2[0])


# ---------------------------------------------------------------------------
# LMTrainer.evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for LMTrainer.evaluate on an untrained model."""

    def test_returns_finite_values(self) -> None:
        """evaluate returns finite (val_loss, val_perplexity)."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        val_loss, val_ppl = trainer.evaluate(model)
        assert math.isfinite(val_loss), f"val_loss is not finite: {val_loss}"
        assert math.isfinite(val_ppl), f"val_ppl is not finite: {val_ppl}"

    def test_perplexity_above_one(self) -> None:
        """Perplexity of an untrained model is > 1.0."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        _val_loss, val_ppl = trainer.evaluate(model)
        assert val_ppl > 1.0, f"Expected val_ppl > 1.0, got {val_ppl}"

    def test_perplexity_is_exp_of_loss(self) -> None:
        """val_perplexity == exp(val_loss) up to float tolerance."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        val_loss, val_ppl = trainer.evaluate(model)
        expected_ppl = math.exp(min(val_loss, 20.0))
        assert (
            abs(val_ppl - expected_ppl) < 1e-3
        ), f"val_ppl={val_ppl} does not match exp(val_loss)={expected_ppl}"


# ---------------------------------------------------------------------------
# LMTrainer.calibration_run
# ---------------------------------------------------------------------------


class TestCalibrationRun:
    """Tests for LMTrainer.calibration_run."""

    def test_completes_without_error(self) -> None:
        """calibration_run(n_steps=10) finishes for a tiny model."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        result = trainer.calibration_run(model, n_steps=10)
        assert isinstance(result, LMTrainingResult)

    def test_result_has_correct_steps(self) -> None:
        """steps_run matches n_steps."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        result = trainer.calibration_run(model, n_steps=10)
        assert result.steps_run == 10

    def test_perplexity_finite_and_above_one(self) -> None:
        """Result perplexity is finite and > 1.0."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        result = trainer.calibration_run(model, n_steps=10)
        assert math.isfinite(result.val_perplexity)
        assert result.val_perplexity > 1.0


# ---------------------------------------------------------------------------
# Perplexity curve
# ---------------------------------------------------------------------------


class TestPerplexityCurve:
    """Tests for the perplexity_curve output."""

    def test_curve_length_matches_eval_frequency(self) -> None:
        """perplexity_curve has one entry per eval checkpoint (plus final)."""
        cfg = LMTrainingConfig(
            max_steps=10,
            eval_steps=5,
            batch_size=2,
            seq_len=64,
            lr=3e-4,
            warmup_steps=2,
            device="cpu",
            seed=0,
        )
        trainer = LMTrainer(cfg)
        model = _make_tiny_model()
        result = trainer.train(model, max_steps=10)
        # eval at step 5, then step 10 (last step also triggers eval); train()
        # also calls evaluate once more after the loop for final metrics but
        # that extra call does not append to the curve.
        assert len(result.perplexity_curve) >= 2

    def test_curve_grows_with_more_steps(self) -> None:
        """More steps produce a longer or equal perplexity curve."""
        cfg_short = LMTrainingConfig(
            max_steps=5,
            eval_steps=5,
            batch_size=2,
            seq_len=64,
            lr=3e-4,
            warmup_steps=1,
            device="cpu",
            seed=0,
        )
        cfg_long = LMTrainingConfig(
            max_steps=10,
            eval_steps=5,
            batch_size=2,
            seq_len=64,
            lr=3e-4,
            warmup_steps=1,
            device="cpu",
            seed=0,
        )
        trainer_short = LMTrainer(cfg_short)
        trainer_long = LMTrainer(cfg_long)
        res_short = trainer_short.train(_make_tiny_model(), max_steps=5)
        res_long = trainer_long.train(_make_tiny_model(), max_steps=10)
        assert len(res_long.perplexity_curve) >= len(res_short.perplexity_curve)

    def test_curve_entries_are_positive(self) -> None:
        """All (step, ppl) entries have step > 0 and ppl > 1.0."""
        trainer = LMTrainer(_TINY_TRAIN_CFG)
        model = _make_tiny_model()
        result = trainer.train(model, max_steps=10)
        for step, ppl in result.perplexity_curve:
            assert step > 0
            assert ppl > 1.0


# ---------------------------------------------------------------------------
# Loss decreases
# ---------------------------------------------------------------------------


class TestLossDecreases:  # pylint: disable=too-few-public-methods
    """Sanity-check that training reduces loss over 10 steps."""

    def test_train_loss_decreases(self) -> None:
        """Final train loss after 10 steps is lower than initial val loss.

        An untrained model has a high initial validation loss.  After 10
        gradient steps the training loss should be strictly lower.
        """
        cfg = LMTrainingConfig(
            max_steps=10,
            eval_steps=10,
            batch_size=2,
            seq_len=64,
            lr=3e-4,
            warmup_steps=2,
            device="cpu",
            seed=42,
        )
        model = _make_tiny_model()
        trainer = LMTrainer(cfg)

        # Record the initial validation loss of the untrained model.
        initial_val_loss, _ = trainer.evaluate(model)

        # Run training.
        result = trainer.train(model, max_steps=10)

        assert result.final_train_loss < initial_val_loss, (
            f"Expected final_train_loss={result.final_train_loss:.4f} < "
            f"initial_val_loss={initial_val_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# WikiText2Dataset dtype / shape (spec-required tests)
# ---------------------------------------------------------------------------


class TestWikiText2DatasetDtypeShape:
    """Verify that WikiText2Dataset items are (seq_len,) int64 tensors."""

    def test_item_is_int64(self) -> None:
        """Items returned by __getitem__ have dtype torch.int64 (== torch.long)."""
        ds = WikiText2Dataset(split="validation", seq_len=64)
        item = ds[0]
        assert item.dtype == torch.int64, f"Expected int64, got {item.dtype}"

    def test_item_shape_is_seq_len(self) -> None:
        """Item shape is exactly (seq_len,) — a 1-D tensor."""
        seq_len = 64
        ds = WikiText2Dataset(split="validation", seq_len=seq_len)
        assert ds[0].shape == (seq_len,)
        assert ds[0].ndim == 1


# ---------------------------------------------------------------------------
# evaluate() contract
# ---------------------------------------------------------------------------


class TestEvaluateContract:
    """evaluate() returns finite ppl > 1 even for a tiny random-init model."""

    def test_finite_ppl_greater_than_one(self) -> None:
        """Perplexity is finite and strictly > 1 for a random-init tiny model."""
        cfg = LMTrainingConfig(
            max_steps=1,
            eval_steps=1,
            batch_size=2,
            seq_len=64,
            device="cpu",
            seed=0,
        )
        trainer = LMTrainer(cfg)
        model = ZKTransformer(
            TransformerConfig(
                d_model=64,
                n_heads=4,
                n_layers=2,
                seq_len=64,
                ff_mult=2.0,
                activation="gelu",
                use_rotary=False,
                use_layernorm=True,
                vocab_size=50257,
                tie_embeddings=False,
            )
        )
        _loss, ppl = trainer.evaluate(model)
        assert math.isfinite(ppl), f"Expected finite ppl, got {ppl}"
        assert ppl > 1.0, f"Expected ppl > 1.0, got {ppl}"


# ---------------------------------------------------------------------------
# calibration_run perplexity_curve length
# ---------------------------------------------------------------------------


class TestCalibrationRunCurveLength:
    """calibration_run completes and perplexity_curve has the correct length."""

    def test_curve_length(self) -> None:
        """perplexity_curve length matches expected evaluation checkpoints."""
        n_steps = 5
        cfg = LMTrainingConfig(
            max_steps=n_steps,
            eval_steps=5,
            batch_size=2,
            seq_len=64,
            device="cpu",
            seed=0,
        )
        trainer = LMTrainer(cfg)
        model = ZKTransformer(_TINY_CONFIG)
        result = trainer.calibration_run(model, n_steps=n_steps)
        # eval_steps=5, n_steps=5 → one checkpoint at step 5 (which is also
        # the last step); the loop triggers eval once.
        assert len(result.perplexity_curve) >= 1, (
            f"Expected at least one entry in perplexity_curve, "
            f"got {len(result.perplexity_curve)}"
        )

    def test_curve_entries_positive(self) -> None:
        """All curve entries have step > 0 and ppl > 1."""
        cfg = LMTrainingConfig(
            max_steps=5,
            eval_steps=5,
            batch_size=2,
            seq_len=64,
            device="cpu",
            seed=0,
        )
        trainer = LMTrainer(cfg)
        model = ZKTransformer(_TINY_CONFIG)
        result = trainer.calibration_run(model, n_steps=5)
        for step, ppl in result.perplexity_curve:
            assert step > 0, f"step must be > 0, got {step}"
            assert ppl > 1.0, f"ppl must be > 1.0, got {ppl}"


# ---------------------------------------------------------------------------
# build_and_initialize
# ---------------------------------------------------------------------------


class TestBuildAndInitialize:
    """build_and_initialize returns a valid nn.Module regardless of pretrained flag."""

    def test_random_init_returns_model(self) -> None:
        """build_and_initialize returns a ZKTransformer when use_pretrained=False."""
        cfg = LMTrainingConfig(
            use_pretrained=False,
            freeze_embeddings=False,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        assert isinstance(model, nn.Module)
        assert isinstance(model, ZKTransformer)

    def test_pretrained_true_returns_model(self) -> None:
        """build_and_initialize with use_pretrained=True still returns a model.

        When the pretrained weights are unavailable (no internet, no
        transformers package, etc.) the function falls back to random init
        rather than raising.
        """
        cfg = LMTrainingConfig(
            use_pretrained=True,
            freeze_embeddings=False,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        assert isinstance(model, nn.Module)

    def test_model_produces_correct_output_shape(self) -> None:
        """Model returned by build_and_initialize produces logits of correct shape."""
        cfg = LMTrainingConfig(
            use_pretrained=False,
            freeze_embeddings=False,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        model.eval()
        sample = model.get_sample_input(batch_size=2)
        with torch.no_grad():
            logits = model(sample)
        assert logits.shape == (2, _TINY_CONFIG.seq_len, _TINY_CONFIG.vocab_size)


# ---------------------------------------------------------------------------
# freeze_embeddings
# ---------------------------------------------------------------------------


class TestFreezeEmbeddings:
    """freeze_embeddings=True actually freezes the named parameters."""

    def test_token_emb_frozen(self) -> None:
        """token_emb.weight.requires_grad is False when freeze_embeddings=True."""
        cfg = LMTrainingConfig(
            use_pretrained=False,
            freeze_embeddings=True,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        assert not model.token_emb.weight.requires_grad, (  # type: ignore[union-attr]
            "token_emb.weight should be frozen"
        )

    def test_lm_head_frozen(self) -> None:
        """lm_head.weight.requires_grad is False when freeze_embeddings=True."""
        cfg = LMTrainingConfig(
            use_pretrained=False,
            freeze_embeddings=True,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        assert not model.lm_head.weight.requires_grad, (  # type: ignore[union-attr]
            "lm_head.weight should be frozen"
        )

    def test_transformer_blocks_not_frozen(self) -> None:
        """Transformer block parameters remain trainable when freeze_embeddings=True."""
        cfg = LMTrainingConfig(
            use_pretrained=False,
            freeze_embeddings=True,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        # At least one block parameter must have requires_grad=True.
        block_params = list(model.blocks.parameters())  # type: ignore[union-attr]
        assert any(p.requires_grad for p in block_params), (
            "At least one transformer block parameter must remain trainable"
        )

    def test_no_freeze_when_flag_false(self) -> None:
        """Both embedding and lm_head remain trainable when freeze_embeddings=False."""
        cfg = LMTrainingConfig(
            use_pretrained=False,
            freeze_embeddings=False,
            device="cpu",
        )
        trainer = LMTrainer(cfg)
        model = trainer.build_and_initialize(_TINY_CONFIG)
        assert model.token_emb.weight.requires_grad, (  # type: ignore[union-attr]
            "token_emb.weight should be trainable when freeze_embeddings=False"
        )
        assert model.lm_head.weight.requires_grad, (  # type: ignore[union-attr]
            "lm_head.weight should be trainable when freeze_embeddings=False"
        )

#!/usr/bin/env python3
"""Fine-tuned comparison: pretrained GPT-2 weights vs scratch training.

Addresses the reviewer weakness of training from scratch on 2.1M tokens
vs GPT-2's 40B tokens.  All models here start from (or distil from)
pretrained GPT-2 small weights for a fair comparison.

Four conditions compared
------------------------
1. ``gpt2_baseline``      — pretrained GPT-2 small, zero-shot evaluation.
2. ``gpt2_linear_attn``   — GPT-2 architecture (768d/12L) with linear
   attention (no softmax) swapped in; weights copied from pretrained GPT-2;
   fine-tuned for ``FINETUNE_STEPS`` steps.
3. ``zk_11x_distilled``   — 384d/4L ZKTransformer with linear attention;
   distillation-fine-tuned from frozen GPT-2 teacher for ``KD_STEPS`` steps.
4. ``zk_11x_scratch``     — same 384d/4L ZKTransformer trained from scratch
   (our existing result; re-run here for a consistent eval budget).

Key question: does distillation bring ``zk_11x`` closer to GPT-2 quality
than scratch training on the same token budget?
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from proxy.features import TransformerConfig  # noqa: E402
from search.lm_trainer import (  # noqa: E402
    LMTrainer,
    LMTrainingConfig,
    LMTrainingResult,  # used for type annotations in helpers
)
from search.transformer_space import ZKTransformer  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "results" / "finetuned_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE: int = 50257
SEQ_LEN: int = 128
BATCH_SIZE: int = 8

# Number of fine-tuning steps for each condition.
# Set to 500 for a fast validation pass; bump to 3000 for full results.
#
# Note: gpt2_linear_attn is evaluated zero-shot (no fine-tuning) because
# fine-tuning a 124M-param model on CPU for 500 steps takes ~30+ minutes.
# Zero-shot evaluation is also scientifically cleaner: it isolates the
# pure cost of removing softmax from the attention mechanism.
KD_STEPS: int = 3000         # zk_11x_distilled (distillation)
SCRATCH_STEPS: int = 3000    # zk_11x_scratch (from random init)
# Fine-tune config for gpt2_linear_attn (used only when explicitly enabled)
FINETUNE_STEPS: int = 3000   # gpt2_linear_attn fine-tune (optional)

# Distillation mixing coefficient: alpha * KL + (1 - alpha) * CE
DISTILLATION_ALPHA: float = 0.7

# GPT-2 ZK cost reference from operator attribution (seq_len=512, extrapolated)
GPT2_ZK_COST: int = 98_611_619_328
ZK_11X_COST: int = 8_968_283_648  # from earlier ezkl results (seq_len=128 ≈ 8.97 B)

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

# GPT-2 architecture with linear attention (softmax removed)
# d_model/n_heads/n_layers match GPT-2 exactly so full weight copy applies
GPT2_LINEAR_ATTN_CONFIG = TransformerConfig(
    d_model=768,
    n_heads=12,
    n_layers=12,
    seq_len=SEQ_LEN,
    ff_mult=4.0,
    activation="linear",   # triggers linear attention + linear FFN activation
    use_rotary=False,
    use_layernorm=True,
    vocab_size=VOCAB_SIZE,
    tie_embeddings=True,
)

# zk_11x: small ZK-friendly config (384d/4L/linear)
ZK_11X_CONFIG = TransformerConfig(
    d_model=384,
    n_heads=6,
    n_layers=4,
    seq_len=SEQ_LEN,
    ff_mult=4.0,
    activation="linear",
    use_rotary=True,
    use_layernorm=True,
    vocab_size=VOCAB_SIZE,
    tie_embeddings=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_params(model: torch.nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_trainer(
    *,
    max_steps: int,
    distillation_alpha: float = 0.0,
    use_pretrained: bool = False,
    freeze_embeddings: bool = False,
    lr: float = 1e-4,
) -> LMTrainer:
    """Construct an :class:`LMTrainer` with the shared experimental settings."""
    cfg = LMTrainingConfig(
        max_steps=max_steps,
        eval_steps=max(max_steps // 5, 100),
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        lr=lr,
        warmup_steps=max(max_steps // 10, 20),
        weight_decay=0.1,
        grad_clip=1.0,
        device="cpu",
        seed=42,
        max_eval_batches=20,
        use_pretrained=use_pretrained,
        freeze_embeddings=freeze_embeddings,
        distillation_alpha=distillation_alpha,
    )
    return LMTrainer(cfg)


def _load_gpt2_once() -> Any:
    """Load pretrained GPT-2 small from HuggingFace exactly once.

    Returns
    -------
    Any
        A ``GPT2LMHeadModel`` loaded from ``"gpt2"``, frozen and in eval mode.
    """
    from transformers import GPT2LMHeadModel  # pylint: disable=import-outside-toplevel

    print("Loading pretrained GPT-2 small (one-time) … ", end="", flush=True)
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    for param in hf_model.parameters():
        param.requires_grad = False
    print("done", flush=True)
    return hf_model


def _build_zk_from_hf(config: TransformerConfig, hf_model: Any) -> ZKTransformer:
    """Build a :class:`ZKTransformer` and copy weights from a loaded HF model.

    Avoids a second ``from_pretrained("gpt2")`` network call by working
    directly with an already-loaded ``GPT2LMHeadModel``.

    Parameters
    ----------
    config:
        Architectural hyperparameters for the target ZKTransformer.
    hf_model:
        Already-loaded, frozen ``GPT2LMHeadModel``.

    Returns
    -------
    ZKTransformer
        Model with weights copied from *hf_model* (or random init where
        dimensions do not match).
    """
    from search.lm_trainer import _copy_gpt2_weights  # pylint: disable=import-outside-toplevel,protected-access

    model = ZKTransformer(config)
    with torch.no_grad():
        _copy_gpt2_weights(model, hf_model, config)
    return model


def _result_to_dict(result: LMTrainingResult) -> dict[str, Any]:
    """Serialise an :class:`LMTrainingResult` to a JSON-compatible dict."""
    return {
        "val_perplexity": result.val_perplexity,
        "val_loss": result.val_loss,
        "final_train_loss": result.final_train_loss,
        "steps_run": result.steps_run,
        "train_time_s": result.train_time_s,
        "converged": result.converged,
        "best_val_perplexity": result.best_val_perplexity,
        "perplexity_curve": result.perplexity_curve,
    }


def _precompute_teacher_logits(
    n_steps: int,
    batch_size: int,
    seq_len: int,
    hf_teacher: Any,
    n_unique: int = 50,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run GPT-2 small forward pass on WikiText-2 training batches and
    cache the resulting logits.

    Pre-computing teacher logits upfront (once) is far more CPU-efficient
    than running the 124M-param teacher at every student training step.
    The cached logits are kept in memory as a list of tensors of shape
    ``(batch_size, seq_len, vocab_size)``.

    To bound wall-clock time, we pre-compute only ``n_unique`` distinct
    batches and tile them to length ``n_steps``.  GPT-2 inference on CPU
    takes ~5-6 seconds per batch, so ``n_unique=50`` takes ~5 minutes.

    Parameters
    ----------
    n_steps:
        Number of training steps; output list will have exactly this length.
    batch_size:
        Batch size used during student training.
    seq_len:
        Sequence length used during student training.
    n_unique:
        Number of distinct batches to pre-compute.  The list is tiled to
        reach ``n_steps``.  Defaults to 50.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        A pair ``(tokens_list, logits_list)`` each of length ``n_steps``.
        - ``tokens_list[i]``: integer token tensor of shape
          ``(batch_size, seq_len)``.
        - ``logits_list[i]``: teacher logits of shape
          ``(batch_size, seq_len, vocab_size)``, dtype ``float16``.
        Both lists are tiled so tokens and logits always correspond to the
        same input batch.
    """
    from search.lm_trainer import WikiText2Dataset  # pylint: disable=import-outside-toplevel

    n_compute = min(n_steps, n_unique)
    print(
        f"  Pre-computing teacher logits for {n_compute} unique batches "
        f"(tiled to {n_steps} steps) … ",
        end="",
        flush=True,
    )
    t0 = time.time()

    # Use the already-loaded teacher model (avoid re-downloading GPT-2).
    teacher = hf_teacher

    ds = WikiText2Dataset(split="train", seq_len=seq_len)
    loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    torch.manual_seed(42)
    loader_iter = iter(loader)

    unique_tokens: list[torch.Tensor] = []
    unique_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_compute):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            logits = teacher(batch).logits  # (batch, seq_len, vocab_size)
            unique_tokens.append(batch)
            # Store logits as float16 to halve memory footprint
            unique_logits.append(logits.half())

    # Tile unique batches to cover all n_steps (tokens + logits stay in sync)
    tiled_tokens = [unique_tokens[i % n_compute] for i in range(n_steps)]
    tiled_logits = [unique_logits[i % n_compute] for i in range(n_steps)]

    elapsed = time.time() - t0
    mem_mb = sum(t.nelement() * t.element_size() for t in unique_logits) / 1e6
    print(f"done ({elapsed:.0f}s, {mem_mb:.0f} MB logits)  ", flush=True)
    return tiled_tokens, tiled_logits


def _distillation_train_loop(  # pylint: disable=too-many-arguments,too-many-locals
    student: nn.Module,
    tokens_cache: list[torch.Tensor],
    teacher_logits_cache: list[torch.Tensor],
    val_trainer: LMTrainer,
    n_steps: int,
    alpha: float,
    lr: float,
    warmup_steps: int,
) -> LMTrainingResult:
    """Train *student* using pre-cached teacher tokens and logits.

    This is more efficient than :meth:`LMTrainer.train` with
    ``distillation_alpha > 0`` because the heavy teacher model does not run
    at every step — its logits are pre-computed and stored in memory.

    The loss is::

        alpha * KL(student ∥ teacher) + (1 - alpha) * CE(student, tokens)

    Both student and teacher process the **same token batches** (from
    ``tokens_cache``) so the KL divergence is well-defined.

    Parameters
    ----------
    student:
        The model being trained (ZKTransformer or similar).
    tokens_cache:
        Pre-computed input token batches, one tensor per step.
    teacher_logits_cache:
        Pre-computed teacher logits matching ``tokens_cache`` step-for-step.
    val_trainer:
        Used only for :meth:`LMTrainer.evaluate` (no training performed).
    n_steps:
        Number of gradient steps.
    alpha:
        Weight on the KL term (distillation strength).
    lr:
        Peak AdamW learning rate.
    warmup_steps:
        Linear warm-up steps before cosine decay.

    Returns
    -------
    LMTrainingResult
        Training metrics and perplexity curve.
    """
    device = torch.device("cpu")
    student = student.to(device)
    student.train()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(n_steps - warmup_steps, 1),
        eta_min=0.0,
    )

    eval_interval = max(n_steps // 5, 100)
    perplexity_curve: list[tuple[int, float]] = []
    best_val_ppl = float("inf")
    final_train_loss = 0.0
    t0 = time.time()

    for step in range(n_steps):
        # Linear warm-up
        if step < warmup_steps:
            lrate = lr * (step + 1) / max(warmup_steps, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = lrate

        # Fetch pre-cached tokens (student and teacher see the same batch)
        tokens = tokens_cache[step].to(device)

        # Student forward
        student_logits = student(tokens)  # (batch, seq_len, vocab_size)

        # Teacher logits (pre-cached, cast back to float32 for loss computation)
        t_logits = teacher_logits_cache[step].float().to(device)

        # Distillation loss: KL + CE on the same token sequence
        vocab_size = student_logits.size(-1)
        shift_student = student_logits[:, :-1, :].reshape(-1, vocab_size)
        shift_teacher = t_logits[:, :-1, :].reshape(-1, vocab_size)
        shift_targets = tokens[:, 1:].reshape(-1)

        ce_loss = F.cross_entropy(shift_student, shift_targets)
        log_p_student = F.log_softmax(shift_student, dim=-1)
        log_p_teacher = F.log_softmax(shift_teacher, dim=-1)
        kl_loss = F.kl_div(
            log_p_student, log_p_teacher, log_target=True, reduction="batchmean"
        )
        loss = alpha * kl_loss + (1.0 - alpha) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step >= warmup_steps:
            scheduler.step()

        final_train_loss = loss.item()

        # Periodic evaluation
        if (step + 1) % eval_interval == 0 or step == n_steps - 1:
            _eval_loss, val_ppl = val_trainer.evaluate(student)
            perplexity_curve.append((step + 1, val_ppl))
            best_val_ppl = min(best_val_ppl, val_ppl)
            print(
                f"  step={step+1}/{n_steps}  train_loss={final_train_loss:.4f}"
                f"  val_ppl={val_ppl:.2f}  best={best_val_ppl:.2f}",
                flush=True,
            )
            student.train()

    train_time = time.time() - t0
    _final_eval_loss, final_val_ppl = val_trainer.evaluate(student)
    converged = final_val_ppl <= best_val_ppl * 1.05

    return LMTrainingResult(
        final_train_loss=final_train_loss,
        val_perplexity=final_val_ppl,
        val_loss=_final_eval_loss,
        steps_run=n_steps,
        train_time_s=train_time,
        converged=converged,
        best_val_perplexity=best_val_ppl,
        perplexity_curve=perplexity_curve,
    )


# ---------------------------------------------------------------------------
# Condition 1: GPT-2 baseline (zero-shot evaluation, no fine-tuning)
# ---------------------------------------------------------------------------


def run_gpt2_baseline(trainer: LMTrainer, hf_gpt2: Any) -> dict[str, Any]:
    """Evaluate pretrained GPT-2 small zero-shot on WikiText-2.

    No gradient updates are performed; this gives the gold-standard PPL
    for the pretrained model on our eval protocol.

    Parameters
    ----------
    trainer:
        An :class:`LMTrainer` instance used only for its :meth:`evaluate`
        method.
    hf_gpt2:
        Already-loaded, frozen ``GPT2LMHeadModel`` (from :func:`_load_gpt2_once`).

    Returns
    -------
    dict[str, Any]
        Evaluation metrics.
    """
    print("\n[1/4] GPT-2 baseline — zero-shot evaluation")

    class _GPT2Wrapper(torch.nn.Module):
        """Thin wrapper that makes a HuggingFace GPT2LMHeadModel look like
        a ``ZKTransformer`` to :meth:`LMTrainer.evaluate`.

        :meth:`LMTrainer.evaluate` calls ``model(tokens)`` and expects raw
        logits of shape ``(batch, seq_len, vocab_size)``.  The HuggingFace
        model returns a ``CausalLMOutputWithCrossAttentions`` object, so we
        unwrap it here.
        """

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return hf_gpt2(input_ids).logits

    wrapper = _GPT2Wrapper()
    val_loss, val_ppl = trainer.evaluate(wrapper)
    n_params = sum(p.numel() for p in hf_gpt2.parameters())

    print(f"  val_ppl={val_ppl:.2f}  val_loss={val_loss:.4f}  params={n_params:,}")
    return {
        "val_perplexity": val_ppl,
        "val_loss": val_loss,
        "final_train_loss": None,
        "steps_run": 0,
        "train_time_s": 0.0,
        "converged": True,
        "best_val_perplexity": val_ppl,
        "perplexity_curve": [],
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Condition 2: GPT-2 with linear attention, fine-tuned
# ---------------------------------------------------------------------------


def run_gpt2_linear_attn(trainer: LMTrainer, hf_gpt2: Any) -> dict[str, Any]:
    """Evaluate GPT-2 architecture with linear attention, zero-shot.

    Copies all pretrained GPT-2 small weights into a :class:`ZKTransformer`
    with ``activation='linear'`` (full block copy because d_model=768,
    n_heads=12).  The softmax is replaced with a no-op.
    No gradient updates are performed.

    This isolates the pure perplexity cost of removing the softmax:
    a zero-shot result is cleaner than a fine-tuned one because it
    avoids confounding from optimisation dynamics.

    Parameters
    ----------
    trainer:
        Trainer whose :meth:`evaluate` method is used for zero-shot eval.
    hf_gpt2:
        Already-loaded, frozen ``GPT2LMHeadModel`` (from :func:`_load_gpt2_once`).

    Returns
    -------
    dict[str, Any]
        Evaluation metrics (``steps_run=0``, no perplexity curve).
    """
    print("\n[2/4] GPT-2 with linear attention — zero-shot evaluation")
    model = _build_zk_from_hf(GPT2_LINEAR_ATTN_CONFIG, hf_gpt2)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {GPT2_LINEAR_ATTN_CONFIG.d_model}d / "
          f"{GPT2_LINEAR_ATTN_CONFIG.n_layers}L / linear-attn (pretrained init)")
    print(f"  Total params: {n_params:,}", flush=True)

    val_loss, val_ppl = trainer.evaluate(model)
    print(f"  val_ppl={val_ppl:.2f}  val_loss={val_loss:.4f}", flush=True)

    return {
        "val_perplexity": val_ppl,
        "val_loss": val_loss,
        "final_train_loss": None,
        "steps_run": 0,
        "train_time_s": 0.0,
        "converged": True,
        "best_val_perplexity": val_ppl,
        "perplexity_curve": [],
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Condition 3: zk_11x distillation-fine-tuned from GPT-2 teacher
# ---------------------------------------------------------------------------


def run_zk11x_distilled(
    eval_trainer: LMTrainer,
    teacher_cache: tuple[list[torch.Tensor], list[torch.Tensor]],
    hf_gpt2: Any,
) -> dict[str, Any]:
    """Knowledge-distillation fine-tune for the 384d/4L ZK model.

    The student is a :class:`ZKTransformer` (384d/4L/linear) initialised
    with GPT-2 embedding weights (embedding-only copy path because
    d_model=384≠768).  Pre-cached GPT-2 teacher logits drive the KL term,
    avoiding an expensive teacher forward pass at every step.

    Loss = :data:`DISTILLATION_ALPHA` * KL(student ∥ teacher)
           + (1 - :data:`DISTILLATION_ALPHA`) * CE(student, tokens)

    Parameters
    ----------
    eval_trainer:
        Trainer used only for :meth:`LMTrainer.evaluate` (periodic PPL
        evaluation during training and final eval).
    teacher_cache:
        Pre-computed ``(tokens_list, logits_list)`` from
        :func:`_precompute_teacher_logits`.  Both lists are aligned:
        ``tokens_list[i]`` is the input used to produce ``logits_list[i]``.
    hf_gpt2:
        Already-loaded, frozen ``GPT2LMHeadModel`` used for embedding-weight
        copy into the student (avoids a second network download).

    Returns
    -------
    dict[str, Any]
        Training and evaluation metrics.
    """
    print(f"\n[3/4] zk_11x distilled — {KD_STEPS} steps  alpha={DISTILLATION_ALPHA}")
    # _build_zk_from_hf copies embedding weights only (d_model=384≠768).
    model = _build_zk_from_hf(ZK_11X_CONFIG, hf_gpt2)
    n_params = _count_params(model)
    print(f"  Model: {ZK_11X_CONFIG.d_model}d / {ZK_11X_CONFIG.n_layers}L / linear-attn")
    print(f"  Trainable params: {n_params:,}")

    tokens_list, logits_list = teacher_cache
    result = _distillation_train_loop(
        student=model,
        tokens_cache=tokens_list,
        teacher_logits_cache=logits_list,
        val_trainer=eval_trainer,
        n_steps=KD_STEPS,
        alpha=DISTILLATION_ALPHA,
        lr=3e-4,
        warmup_steps=max(KD_STEPS // 10, 20),
    )
    print(f"  val_ppl={result.val_perplexity:.2f}  "
          f"best={result.best_val_perplexity:.2f}  "
          f"time={result.train_time_s:.0f}s")

    out = _result_to_dict(result)
    out["n_params"] = n_params
    return out


# ---------------------------------------------------------------------------
# Condition 4: zk_11x trained from scratch
# ---------------------------------------------------------------------------


def run_zk11x_scratch(trainer: LMTrainer) -> dict[str, Any]:
    """Train the 384d/4L ZK model from random initialisation.

    No pretrained weights, no distillation — pure scratch training on
    WikiText-2 for :data:`SCRATCH_STEPS` steps.  This replicates our
    existing result (PPL≈34.9 at full convergence) on the same step budget
    used by the other conditions.

    Parameters
    ----------
    trainer:
        Trainer configured with ``use_pretrained=False`` and
        ``distillation_alpha=0``.

    Returns
    -------
    dict[str, Any]
        Training and evaluation metrics.
    """
    print(f"\n[4/4] zk_11x scratch — {SCRATCH_STEPS} steps  (random init)")
    model = ZKTransformer(ZK_11X_CONFIG)
    n_params = _count_params(model)
    print(f"  Model: {ZK_11X_CONFIG.d_model}d / {ZK_11X_CONFIG.n_layers}L / linear-attn")
    print(f"  Trainable params: {n_params:,}")

    result = trainer.train(model, max_steps=SCRATCH_STEPS)
    print(f"  val_ppl={result.val_perplexity:.2f}  "
          f"best={result.best_val_perplexity:.2f}  "
          f"time={result.train_time_s:.0f}s")

    out = _result_to_dict(result)
    out["n_params"] = n_params
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate all four conditions and print the comparison table."""
    print("=" * 70)
    print("Fine-tuned Comparison Experiment")
    print(f"seq_len={SEQ_LEN}  batch_size={BATCH_SIZE}")
    print(f"kd_steps={KD_STEPS}  scratch_steps={SCRATCH_STEPS}  "
          f"gpt2_linear_attn=zero-shot")
    print("=" * 70)

    # -- Load GPT-2 weights once (shared across all conditions that need it) -
    hf_gpt2 = _load_gpt2_once()

    # -- Shared eval-only trainer (conditions 1 and 2 need no training) ----
    eval_trainer = _make_trainer(max_steps=1, use_pretrained=False)

    # -- Condition 1: GPT-2 baseline ---------------------------------------
    gpt2_result = run_gpt2_baseline(eval_trainer, hf_gpt2)

    # -- Condition 2: GPT-2 with linear attention (zero-shot) --------------
    # We reuse eval_trainer since no gradient updates are needed.
    gpt2_linear_result = run_gpt2_linear_attn(eval_trainer, hf_gpt2)

    # -- Condition 3: zk_11x with distillation ----------------------------
    # Pre-compute teacher logits once; reuse eval_trainer for periodic eval.
    print("\nPre-computing GPT-2 teacher logits for distillation …")
    teacher_cache: tuple[list[torch.Tensor], list[torch.Tensor]] = (
        _precompute_teacher_logits(
            n_steps=KD_STEPS,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            hf_teacher=hf_gpt2,
            n_unique=200,
        )
    )
    zk11x_distilled_result = run_zk11x_distilled(eval_trainer, teacher_cache, hf_gpt2)
    del teacher_cache  # free memory

    # -- Condition 4: zk_11x from scratch ---------------------------------
    scratch_trainer = _make_trainer(
        max_steps=SCRATCH_STEPS,
        distillation_alpha=0.0,
        use_pretrained=False,
        freeze_embeddings=False,
        lr=3e-4,
    )
    zk11x_scratch_result = run_zk11x_scratch(scratch_trainer)

    # -- Aggregate and save -----------------------------------------------
    results: dict[str, Any] = {
        "metadata": {
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "gpt2_linear_attn_mode": "zero_shot",
            "kd_steps": KD_STEPS,
            "scratch_steps": SCRATCH_STEPS,
            "distillation_alpha": DISTILLATION_ALPHA,
            "gpt2_zk_cost": GPT2_ZK_COST,
            "zk_11x_cost": ZK_11X_COST,
        },
        "conditions": {
            "gpt2_baseline": gpt2_result,
            "gpt2_linear_attn": gpt2_linear_result,
            "zk_11x_distilled": zk11x_distilled_result,
            "zk_11x_scratch": zk11x_scratch_result,
        },
    }

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    # -- Print comparison table -------------------------------------------
    gpt2_ppl = gpt2_result["val_perplexity"]

    print("\n" + "=" * 70)
    print("Comparison Table")
    print(
        f"{'Condition':<25} {'PPL':>8} {'vs GPT-2':>10} "
        f"{'Steps':>7} {'Time(s)':>9} {'ZK Speedup':>12}"
    )
    print("-" * 70)

    rows = [
        ("gpt2_baseline", gpt2_result, GPT2_ZK_COST),
        ("gpt2_linear_attn", gpt2_linear_result, GPT2_ZK_COST),
        ("zk_11x_distilled", zk11x_distilled_result, ZK_11X_COST),
        ("zk_11x_scratch", zk11x_scratch_result, ZK_11X_COST),
    ]
    for name, res, zk_cost in rows:
        ppl = res["val_perplexity"]
        steps = res["steps_run"]
        t = res["train_time_s"]
        speedup = GPT2_ZK_COST / zk_cost
        vs_gpt2 = ppl / gpt2_ppl if gpt2_ppl > 0 else float("nan")
        print(
            f"{name:<25} {ppl:>8.2f} {vs_gpt2:>10.3f} "
            f"{steps:>7} {t:>9.1f} {speedup:>11.1f}x"
        )

    print("=" * 70)
    print("\nKey finding:")
    zk11x_dist_ppl = zk11x_distilled_result["val_perplexity"]
    zk11x_scr_ppl = zk11x_scratch_result["val_perplexity"]
    improvement = zk11x_scr_ppl - zk11x_dist_ppl
    speedup_zk = GPT2_ZK_COST / ZK_11X_COST
    print(f"  zk_11x distilled PPL:  {zk11x_dist_ppl:.2f}")
    print(f"  zk_11x scratch PPL:    {zk11x_scr_ppl:.2f}")
    print(f"  Distillation gain:     {improvement:+.2f} PPL points")
    print(f"  ZK speedup (11x):      {speedup_zk:.1f}x cheaper to prove")


if __name__ == "__main__":
    main()

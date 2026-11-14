"""Language model trainer for ZK-NAS.

Trains GPT-2-class ``ZKTransformer`` models on WikiText-2 and evaluates
perplexity. Supports two modes:

1. **Short calibration runs** (500 steps) — validates zero-cost proxy
   correlations by checking Spearman(proxy_score, calibration_ppl).
2. **Full training** — trains to convergence for Pareto frontier results.

All training runs on CPU; no mixed precision or distributed training.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from proxy.features import TransformerConfig

import tiktoken
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.path.expanduser("~/.cache/zkml-nas"))
_WIKITEXT_DATASET = "Salesforce/wikitext"
_WIKITEXT_CONFIG = "wikitext-2-raw-v1"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LMTrainingConfig:  # pylint: disable=too-many-instance-attributes
    """Hyperparameters for language model training on WikiText-2."""

    max_steps: int = 10000
    """Total number of gradient update steps for a full training run."""

    eval_steps: int = 500
    """Evaluate on the validation set every this many steps."""

    batch_size: int = 4
    """Number of sequences per gradient update."""

    seq_len: int = 128
    """Token sequence length; must match ``TransformerConfig.seq_len``."""

    lr: float = 3e-4
    """Peak learning rate for AdamW."""

    warmup_steps: int = 200
    """Number of linear warm-up steps before cosine decay."""

    weight_decay: float = 0.1
    """AdamW weight decay coefficient."""

    grad_clip: float = 1.0
    """Maximum gradient norm for gradient clipping (0.0 = disabled)."""

    device: str = "cpu"
    """Torch device string, e.g. ``"cpu"`` or ``"cuda"``."""

    seed: int = 42
    """Random seed for reproducibility."""

    dataset: str = "wikitext-2-raw-v1"
    """HuggingFace dataset configuration name."""

    max_eval_batches: int = 20
    """Maximum number of batches used during validation (0 = use all).

    Capping evaluation is essential for fast CPU tests and calibration runs
    where an exact perplexity is not required.  Set to 0 for full evaluation.
    """

    use_pretrained: bool = True
    """When True, :meth:`LMTrainer.build_and_initialize` loads GPT-2 small
    weights via :func:`load_pretrained_gpt2` before returning the model.
    Falls back to random init if the pretrained weights are unavailable.
    """

    freeze_embeddings: bool = True
    """When True, :meth:`LMTrainer.build_and_initialize` freezes the
    ``token_emb`` and ``lm_head`` parameters so gradients are not computed
    for the large vocabulary layers during training.
    """

    distillation_alpha: float = 0.0
    """Knowledge-distillation mixing coefficient.

    When > 0, :meth:`LMTrainer.train` computes

    ``alpha * KL(student || teacher) + (1 - alpha) * CE``

    where the teacher is GPT-2 small loaded once and frozen.  Set to 0.0
    (default) to use plain cross-entropy with no distillation.
    """


@dataclass
class LMTrainingResult:  # pylint: disable=too-many-instance-attributes
    """Results produced by a completed training run."""

    final_train_loss: float
    """Cross-entropy loss on the last training batch."""

    val_perplexity: float
    """Perplexity on the WikiText-2 validation set at the end of training."""

    val_loss: float
    """Cross-entropy loss on the WikiText-2 validation set at the end of training."""

    steps_run: int
    """Number of gradient update steps actually performed."""

    train_time_s: float
    """Wall-clock training time in seconds."""

    converged: bool
    """True if ``val_perplexity`` is within 5 % of ``best_val_perplexity``."""

    best_val_perplexity: float
    """Lowest validation perplexity observed during training."""

    perplexity_curve: list[tuple[int, float]] = field(default_factory=list)
    """List of ``(step, perplexity)`` pairs recorded at every evaluation."""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WikiText2Dataset(Dataset):
    """WikiText-2 tokenized as fixed-length non-overlapping blocks.

    Tokens are produced by the GPT-2 BPE tokenizer (tiktoken ``"gpt2"``).
    Tokenized data is cached to ``~/.cache/zkml-nas/`` so subsequent loads
    are instant.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"validation"``, or ``"test"``.
    seq_len:
        Block length in tokens; each :meth:`__getitem__` call returns exactly
        this many tokens.
    tokenizer_name:
        tiktoken encoding name, defaults to ``"gpt2"``.
    cache_dir:
        Directory for cached token tensors. Defaults to
        ``~/.cache/zkml-nas/``.
    """

    def __init__(
        self,
        split: str,
        seq_len: int = 128,
        tokenizer_name: str = "gpt2",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = split
        self.seq_len = seq_len

        cache_root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"wikitext2_{split}_{seq_len}.pt"

        if cache_path.exists():
            logger.debug("Loading cached tokens from %s", cache_path)
            self._tokens: torch.Tensor = torch.load(
                cache_path, weights_only=True
            )
        else:
            logger.info(
                "Tokenizing WikiText-2 %s split; caching to %s", split, cache_path
            )
            self._tokens = self._tokenize(split, tokenizer_name)
            torch.save(self._tokens, cache_path)

        # Trim to a multiple of seq_len and reshape into blocks.
        n_blocks = len(self._tokens) // seq_len
        self._blocks: torch.Tensor = self._tokens[: n_blocks * seq_len].reshape(
            n_blocks, seq_len
        )

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(split: str, tokenizer_name: str) -> torch.Tensor:
        """Load WikiText-2 from HuggingFace, concatenate and tokenize.

        Returns
        -------
        torch.Tensor
            1-D long tensor of all token IDs.
        """
        import datasets as hf_datasets  # pylint: disable=import-outside-toplevel

        ds = hf_datasets.load_dataset(
            _WIKITEXT_DATASET,
            _WIKITEXT_CONFIG,
            split=split,
            trust_remote_code=False,
        )
        enc = tiktoken.get_encoding(tokenizer_name)

        all_ids: list[int] = []
        for example in ds:
            text: str = example["text"]  # type: ignore[index]
            if text.strip():
                all_ids.extend(enc.encode(text))

        return torch.tensor(all_ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of non-overlapping token blocks."""
        return len(self._blocks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return token block of shape ``(seq_len,)``."""
        return self._blocks[idx]


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

#: ZK-friendly activations that use polynomial / linear functions.
_ZK_FRIENDLY_ACTIVATIONS: frozenset[str] = frozenset({"poly2", "poly3", "linear"})

#: GPT-2 small hidden dimension used for full weight-copy path.
_GPT2_D_MODEL: int = 768
#: GPT-2 small number of attention heads.
_GPT2_N_HEADS: int = 12
#: GPT-2 small number of transformer layers.
_GPT2_N_LAYERS: int = 12


def load_pretrained_gpt2(config: TransformerConfig) -> nn.Module:
    """Load HuggingFace GPT-2 small weights into a :class:`~search.transformer_space.ZKTransformer`.

    Three adaptation paths are supported:

    1. **Full copy** — ``config.n_layers <= 12`` and ``config.d_model == 768``
       and ``config.n_heads == 12``: all transformer block weights are copied;
       layers beyond ``config.n_layers`` are simply dropped.
    2. **Embedding-only copy** — ``config.d_model != 768``: embedding and LM-head
       weights are copied (sliced to fit ``config.d_model``); transformer blocks
       are reinitialized randomly.  Embeddings carry most of the vocabulary
       signal, so this still helps meaningfully.
    3. **Random init fallback** — if ``transformers`` is not installed or the
       download fails, a randomly initialised :class:`~search.transformer_space.ZKTransformer`
       is returned with a logged warning.

    For ZK-friendly activations (``poly2``, ``poly3``, ``linear``), the
    standard GPT-2 weights are used as the initialisation point.  The
    ``ZKTransformer`` forward pass already replaces GELU/softmax with the
    configured polynomial equivalents, so no weight patching is needed.

    Parameters
    ----------
    config:
        Architectural hyperparameters for the target model.

    Returns
    -------
    nn.Module
        :class:`~search.transformer_space.ZKTransformer` with weights
        initialised from GPT-2 small (or randomly if unavailable).
    """
    # Import here to avoid a hard dependency at module level.
    from search.transformer_space import ZKTransformer  # pylint: disable=import-outside-toplevel

    model = ZKTransformer(config)

    try:
        from transformers import AutoModelForCausalLM  # pylint: disable=import-outside-toplevel

        logger.info("Loading pretrained GPT-2 small weights from HuggingFace…")
        pretrained = AutoModelForCausalLM.from_pretrained("gpt2")
        pretrained.eval()

        with torch.no_grad():
            _copy_gpt2_weights(model, pretrained, config)

        logger.info("Pretrained GPT-2 weights loaded successfully.")

    except ImportError:
        logger.warning(
            "transformers not available; using random init for ZKTransformer."
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Failed to load pretrained GPT-2 weights (%s); falling back to random init.",
            exc,
        )

    return model


def _copy_gpt2_weights(
    model: nn.Module,
    pretrained: nn.Module,
    config: TransformerConfig,
) -> None:
    """Copy GPT-2 small weights into *model* in-place.

    Handles two cases:
    - Full copy when ``config.d_model == 768`` (GPT-2 hidden dim).
    - Embedding-only copy otherwise (sliced to ``config.d_model``).

    Parameters
    ----------
    model:
        Target :class:`~search.transformer_space.ZKTransformer`.
    pretrained:
        Source ``AutoModelForCausalLM`` loaded from ``"gpt2"``.
    config:
        Architectural hyperparameters of *model*.
    """
    from search.transformer_space import ZKTransformer  # pylint: disable=import-outside-toplevel

    if not isinstance(model, ZKTransformer):
        raise TypeError(f"Expected ZKTransformer, got {type(model)}")

    # Narrow the type so attribute access is type-safe after the isinstance check.
    zk_model: ZKTransformer = model

    src_emb = pretrained.transformer.wte.weight.data  # (50257, 768)
    dst_vocab = min(config.vocab_size, src_emb.shape[0])
    dst_dim = min(config.d_model, src_emb.shape[1])

    # Always copy embedding weights (truncated/sliced to fit).
    zk_model.token_emb.weight.data[:dst_vocab, :dst_dim].copy_(
        src_emb[:dst_vocab, :dst_dim]
    )

    # Copy positional embeddings when the model uses them (non-rotary).
    if zk_model.pos_emb is not None and hasattr(pretrained.transformer, "wpe"):
        src_pos = pretrained.transformer.wpe.weight.data  # (1024, 768)
        dst_seq = min(config.seq_len, src_pos.shape[0])
        dst_dim_pos = min(config.d_model, src_pos.shape[1])
        zk_model.pos_emb.weight.data[:dst_seq, :dst_dim_pos].copy_(
            src_pos[:dst_seq, :dst_dim_pos]
        )

    # Copy LM head weights (same slice as embedding).
    if not config.tie_embeddings:
        zk_model.lm_head.weight.data[:dst_vocab, :dst_dim].copy_(
            src_emb[:dst_vocab, :dst_dim]
        )

    # Full transformer-block copy only when dimensions match.
    full_copy = (
        config.d_model == _GPT2_D_MODEL
        and config.n_heads == _GPT2_N_HEADS
        and config.n_layers <= _GPT2_N_LAYERS
    )
    if not full_copy:
        logger.debug(
            "d_model=%d != 768 or n_heads=%d != 12; skipping transformer block copy.",
            config.d_model,
            config.n_heads,
        )
        return

    src_blocks = pretrained.transformer.h  # list of GPT-2 transformer blocks
    for dst_idx, src_block in enumerate(src_blocks[: config.n_layers]):
        dst_block = zk_model.blocks[dst_idx]

        # Attention QKV projection (GPT-2 stores as c_attn: (d, 3d))
        src_attn = src_block.attn
        if hasattr(src_attn, "c_attn"):
            dst_block.attn.qkv.weight.data.copy_(
                src_attn.c_attn.weight.data.T  # GPT-2 uses Conv1D (transposed)
            )
            dst_block.attn.qkv.bias.data.copy_(src_attn.c_attn.bias.data)
        # Attention output projection
        if hasattr(src_attn, "c_proj"):
            dst_block.attn.out_proj.weight.data.copy_(
                src_attn.c_proj.weight.data.T
            )
            dst_block.attn.out_proj.bias.data.copy_(src_attn.c_proj.bias.data)

        # FFN (GPT-2 stores as c_fc / c_proj inside mlp)
        src_mlp = src_block.mlp
        if hasattr(src_mlp, "c_fc"):
            dst_block.ffn.fc1.weight.data.copy_(src_mlp.c_fc.weight.data.T)
            dst_block.ffn.fc1.bias.data.copy_(src_mlp.c_fc.bias.data)
        if hasattr(src_mlp, "c_proj"):
            dst_block.ffn.fc2.weight.data.copy_(src_mlp.c_proj.weight.data.T)
            dst_block.ffn.fc2.bias.data.copy_(src_mlp.c_proj.bias.data)

        # Layer normalisation (only when config uses LayerNorm, not RMSNorm)
        if config.use_layernorm and hasattr(src_block, "ln_1"):
            dst_block.norm1.weight.data.copy_(src_block.ln_1.weight.data)
            dst_block.norm1.bias.data.copy_(src_block.ln_1.bias.data)
            dst_block.norm2.weight.data.copy_(src_block.ln_2.weight.data)
            dst_block.norm2.bias.data.copy_(src_block.ln_2.bias.data)

    # Final layer norm
    if config.use_layernorm and hasattr(pretrained.transformer, "ln_f"):
        zk_model.final_norm.weight.data.copy_(pretrained.transformer.ln_f.weight.data)
        zk_model.final_norm.bias.data.copy_(pretrained.transformer.ln_f.bias.data)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class LMTrainer:
    """Train a :class:`~search.transformer_space.ZKTransformer` on WikiText-2.

    Parameters
    ----------
    training_config:
        Hyperparameters governing the training loop.
    """

    def __init__(self, training_config: LMTrainingConfig) -> None:
        self.cfg = training_config
        torch.manual_seed(self.cfg.seed)
        self._device = torch.device(self.cfg.device)

        # Cap PyTorch inter-op threads to avoid OpenMP thread-pool overhead on
        # large-core machines.  Small matrices (d_model ≤ 1024) are slower with
        # many threads due to synchronisation costs, so we clamp to at most 4.
        _n_threads = min(torch.get_num_threads(), 4)
        torch.set_num_threads(_n_threads)

        # Lazily constructed datasets / loaders.
        self._train_loader: Optional[DataLoader] = None  # type: ignore[type-arg]
        self._val_dataset: Optional[WikiText2Dataset] = None
        self.best_state: Optional[dict] = None  # best-validation weights (see train)
        self.best_step: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_train_loader(self) -> DataLoader[torch.Tensor]:
        """Return (or build) the training DataLoader."""
        if self._train_loader is None:
            ds = WikiText2Dataset(
                split="train",
                seq_len=self.cfg.seq_len,
            )
            self._train_loader = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
        return self._train_loader

    def _get_val_dataset(self) -> WikiText2Dataset:
        """Return (or build) the validation dataset."""
        if self._val_dataset is None:
            self._val_dataset = WikiText2Dataset(
                split="validation",
                seq_len=self.cfg.seq_len,
            )
        return self._val_dataset

    @staticmethod
    def _compute_lr(
        step: int,
        warmup_steps: int,
        max_steps: int,
        lr: float,
    ) -> float:
        """Compute learning rate with linear warm-up then cosine decay."""
        if step < warmup_steps:
            return lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _compute_loss(
        logits: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss for next-token prediction.

        Parameters
        ----------
        logits:
            Shape ``(batch, seq_len, vocab_size)``.
        tokens:
            Integer token tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Scalar cross-entropy loss.
        """
        vocab_size = logits.size(-1)
        # Predict token t+1 from token t for positions 0..seq_len-2.
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_targets = tokens[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_targets)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(  # pylint: disable=too-many-locals
        self,
        model: nn.Module,
        max_steps: Optional[int] = None,
    ) -> LMTrainingResult:
        """Train *model* on WikiText-2 with AdamW and cosine LR schedule.

        Parameters
        ----------
        model:
            Any ``nn.Module`` whose ``forward(input_ids)`` returns logits of
            shape ``(batch, seq_len, vocab_size)``.
        max_steps:
            If provided, overrides ``training_config.max_steps``.

        Returns
        -------
        LMTrainingResult
            Training metrics and perplexity curve.
        """
        n_steps = max_steps if max_steps is not None else self.cfg.max_steps
        model = model.to(self._device)
        model.train()

        # Load a frozen teacher once if distillation is requested.
        teacher: Optional[nn.Module] = None
        if self.cfg.distillation_alpha > 0.0:
            teacher = self._load_teacher()
            if teacher is not None:
                teacher = teacher.to(self._device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # CosineAnnealingLR decays over the full training period.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(n_steps - self.cfg.warmup_steps, 1),
            eta_min=0.0,
        )

        train_loader = self._get_train_loader()
        loader_iter = iter(train_loader)

        perplexity_curve: list[tuple[int, float]] = []
        self.best_state, self.best_step = None, 0
        best_val_ppl = float("inf")
        final_train_loss = 0.0
        t0 = time.time()

        for step in range(n_steps):
            # Manually set LR during warmup; after warmup, the scheduler takes over.
            if step < self.cfg.warmup_steps:
                lr_now = self._compute_lr(
                    step, self.cfg.warmup_steps, n_steps, self.cfg.lr
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

            # Fetch next batch (cycle the loader if exhausted).
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            tokens = batch.to(self._device)
            logits = model(tokens)

            if teacher is not None and self.cfg.distillation_alpha > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(tokens).logits
                loss = self._distillation_loss(
                    logits, teacher_logits, tokens, self.cfg.distillation_alpha
                )
            else:
                loss = self._compute_loss(logits, tokens)

            optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            # Advance cosine schedule only after warmup.
            if step >= self.cfg.warmup_steps:
                scheduler.step()

            final_train_loss = loss.item()

            # Periodic evaluation.
            if (step + 1) % self.cfg.eval_steps == 0 or step == n_steps - 1:
                _eval_loss, val_ppl = self.evaluate(model)
                perplexity_curve.append((step + 1, val_ppl))
                if val_ppl < best_val_ppl:
                    # Keep the best-validation weights for checkpoint selection.
                    self.best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    self.best_step = step + 1
                best_val_ppl = min(best_val_ppl, val_ppl)
                logger.info(
                    "step=%d  train_loss=%.4f  val_ppl=%.2f",
                    step + 1,
                    final_train_loss,
                    val_ppl,
                )
                print(  # visible even without logging config (for Mako)
                    f"  step={step+1}/{n_steps}  train_loss={final_train_loss:.4f}"
                    f"  val_ppl={val_ppl:.2f}  best={best_val_ppl:.2f}",
                    flush=True,
                )
                model.train()

        train_time = time.time() - t0
        final_val_loss, final_val_ppl = self.evaluate(model)
        model.train()

        converged = final_val_ppl <= best_val_ppl * 1.05

        return LMTrainingResult(
            final_train_loss=final_train_loss,
            val_perplexity=final_val_ppl,
            val_loss=final_val_loss,
            steps_run=n_steps,
            train_time_s=train_time,
            converged=converged,
            best_val_perplexity=best_val_ppl,
            perplexity_curve=perplexity_curve,
        )

    def evaluate(self, model: nn.Module) -> tuple[float, float]:
        """Evaluate *model* on the WikiText-2 validation set.

        Parameters
        ----------
        model:
            ``nn.Module`` with ``forward(input_ids) -> logits``.

        Returns
        -------
        tuple[float, float]
            ``(val_loss, val_perplexity)``.
        """
        val_ds = self._get_val_dataset()
        val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        model = model.to(self._device)
        model.eval()
        total_loss = 0.0
        n_batches = 0
        max_batches = self.cfg.max_eval_batches  # 0 means unlimited

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch.to(self._device)
                logits = model(tokens)
                loss = self._compute_loss(logits, tokens)
                total_loss += loss.item()
                n_batches += 1
                if max_batches > 0 and n_batches >= max_batches:
                    break

        if n_batches == 0:
            return float("inf"), float("inf")

        avg_loss = total_loss / n_batches
        perplexity = math.exp(min(avg_loss, 20.0))  # clip to avoid overflow
        return avg_loss, perplexity

    # ------------------------------------------------------------------
    # Pretrained initialisation
    # ------------------------------------------------------------------

    def build_and_initialize(self, config: TransformerConfig) -> nn.Module:
        """Create a :class:`~search.transformer_space.ZKTransformer` and
        optionally initialise it with GPT-2 small pretrained weights.

        Behaviour is controlled by :attr:`LMTrainingConfig.use_pretrained`
        and :attr:`LMTrainingConfig.freeze_embeddings`:

        - ``use_pretrained=True`` calls :func:`load_pretrained_gpt2` which
          copies weights from ``"gpt2"`` where dimensions allow and falls back
          to random init silently.
        - ``freeze_embeddings=True`` sets ``requires_grad=False`` on
          ``token_emb.weight`` and ``lm_head.weight`` so the large vocabulary
          layers are excluded from gradient computation during training.

        Parameters
        ----------
        config:
            Architectural hyperparameters for the model.

        Returns
        -------
        nn.Module
            Initialised :class:`~search.transformer_space.ZKTransformer`.
        """
        from search.transformer_space import ZKTransformer  # pylint: disable=import-outside-toplevel

        zk_model: ZKTransformer
        if self.cfg.use_pretrained:
            pretrained_or_random = load_pretrained_gpt2(config)
            # load_pretrained_gpt2 always returns a ZKTransformer.
            assert isinstance(pretrained_or_random, ZKTransformer)
            zk_model = pretrained_or_random
        else:
            zk_model = ZKTransformer(config)

        if self.cfg.freeze_embeddings:
            zk_model.token_emb.weight.requires_grad = False
            zk_model.lm_head.weight.requires_grad = False
            logger.debug("Froze token_emb and lm_head weights.")

        return zk_model

    # ------------------------------------------------------------------
    # Distillation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_teacher() -> Optional[nn.Module]:
        """Load frozen GPT-2 small as a knowledge-distillation teacher.

        Returns
        -------
        nn.Module or None
            Frozen GPT-2 causal LM, or ``None`` if unavailable.
        """
        try:
            from transformers import AutoModelForCausalLM  # pylint: disable=import-outside-toplevel

            teacher = AutoModelForCausalLM.from_pretrained("gpt2")
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            return teacher
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Could not load GPT-2 teacher for distillation (%s); "
                "falling back to CE-only loss.",
                exc,
            )
            return None

    @staticmethod
    def _distillation_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        tokens: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Compute ``alpha * KL(student || teacher) + (1 - alpha) * CE``.

        Parameters
        ----------
        student_logits:
            Shape ``(batch, seq_len, vocab_size)``.
        teacher_logits:
            Shape ``(batch, seq_len, vocab_size)`` from the frozen teacher.
        tokens:
            Integer token tensor of shape ``(batch, seq_len)``.
        alpha:
            Weight on the KL term; ``1 - alpha`` weights the CE term.

        Returns
        -------
        torch.Tensor
            Scalar combined loss.
        """
        vocab_size = student_logits.size(-1)
        shift_student = student_logits[:, :-1, :].reshape(-1, vocab_size)
        shift_teacher = teacher_logits[:, :-1, :].reshape(-1, vocab_size)
        shift_targets = tokens[:, 1:].reshape(-1)

        ce_loss = F.cross_entropy(shift_student, shift_targets)

        # KL divergence: KL(P_student || P_teacher) = sum P_s * (log P_s - log P_t)
        log_p_student = F.log_softmax(shift_student, dim=-1)
        log_p_teacher = F.log_softmax(shift_teacher, dim=-1)
        kl_loss = F.kl_div(log_p_student, log_p_teacher, log_target=True, reduction="batchmean")

        return alpha * kl_loss + (1.0 - alpha) * ce_loss

    def calibration_run(
        self,
        model: nn.Module,
        n_steps: int = 500,
    ) -> LMTrainingResult:
        """Short training run for zero-cost proxy calibration.

        Trains *model* for *n_steps* gradient updates and returns perplexity.
        Used to check whether zero-cost proxies correlate with short-run PPL
        via ``Spearman(proxy_score, proxy_calibration_ppl)``.

        Parameters
        ----------
        model:
            Model to train.
        n_steps:
            Number of gradient steps; defaults to 500.

        Returns
        -------
        LMTrainingResult
            Training result with ``val_perplexity`` as the calibration metric.
        """
        return self.train(model, max_steps=n_steps)

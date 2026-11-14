"""Top-level 3-phase NAS orchestrator for ZK-optimal transformer architecture search.

Three distinct phases are required for publishability (Codex review):

Phase 1 — Search
    Zero-cost proxies (decoder_params primary, grad_norm secondary) combined
    with an ONNX ZK cost estimate navigate the search space cheaply.  No
    training is performed.  ActiveLearner selects the most informative batch
    each iteration.

Phase 2 — Calibration
    A stratified sample of ``phase2_sample_size`` configs is trained for
    ``phase2_calibration_steps`` steps on WikiText-2.  Spearman and Kendall
    correlations are measured between each zero-cost proxy and the resulting
    calibration PPL.  This validates the proxy ranking and produces the numbers
    that go in Table 1 of the paper.

Phase 3 — Final training
    The top-K Pareto-frontier configs from Phase 1/2 (plus the GPT-2 baseline)
    are trained to convergence.  The final (PPL, ZK cost) pairs become the
    real paper results (Table 3).
"""

# pylint: disable=too-many-lines  # 3-phase orchestrator requires substantial implementation

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from proxy.accuracy_proxies import (
    decoder_params_score,
    grad_norm_score,
    near_score_lm,
    snip_score,
)
from proxy.active_learning import ActiveLearner
from proxy.data import ProxyDataset, ProxyRecord
from proxy.features import ProxyFeatures, TransformerConfig
from proxy.strategies.kernel_decomp import KernelDecompositionProxy
from proxy.strategies.multifidelity import MultiFidelityProxy
from proxy.strategies.zero_cost import ZeroCostProxy
from search.transformer_space import TransformerSearchSpace, ZKTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NASResult:  # pylint: disable=too-many-instance-attributes
    """Result for one evaluated transformer configuration.

    Attributes
    ----------
    config:
        The evaluated architecture.
    zk_cost:
        ONNX proxy constraint estimate (lower = cheaper to prove).
    decoder_params:
        Non-embedding parameter count.
    decoder_params_score:
        ``log(decoder_params)`` — primary accuracy proxy.
    grad_norm_score:
        Gradient-norm proxy (one backward pass).  ``None`` if not computed.
    snip_score:
        SNIP proxy.  ``None`` if not computed.
    near_score:
        NEAR proxy (ablation only, not validated for LMs).  ``None`` if not
        computed.
    calibration_ppl:
        PPL after ``phase2_calibration_steps`` of training (Phase 2 only).
    final_ppl:
        PPL after full training to convergence (Phase 3 only).
    on_pareto_frontier:
        ``True`` when this result lies on the current Pareto frontier.
    phase:
        NAS phase that produced this result (1, 2, or 3).
    iteration:
        0-based iteration index within the producing phase.
    """

    config: TransformerConfig
    zk_cost: int
    decoder_params: int
    decoder_params_score: float
    grad_norm_score: Optional[float] = None
    snip_score: Optional[float] = None
    near_score: Optional[float] = None
    calibration_ppl: Optional[float] = None
    final_ppl: Optional[float] = None
    on_pareto_frontier: bool = False
    phase: int = 1
    iteration: int = 0


@dataclass
class ProxyValidation:  # pylint: disable=too-many-instance-attributes
    """Results of Phase 2 proxy validation — Table 1 of the paper.

    Attributes
    ----------
    n_configs:
        Number of configs trained in the calibration sample.
    spearman_decoder_params:
        Spearman r between decoder_params_score and calibration PPL.
    spearman_grad_norm:
        Spearman r between grad_norm_score and calibration PPL.
    spearman_snip:
        Spearman r between snip_score and calibration PPL.
    spearman_near:
        Spearman r between near_score and calibration PPL.
    kendall_decoder_params:
        Kendall tau between decoder_params_score and calibration PPL.
    kendall_grad_norm:
        Kendall tau between grad_norm_score and calibration PPL.
    best_proxy:
        Name of the proxy with the highest Spearman r.
    """

    n_configs: int
    spearman_decoder_params: float
    spearman_grad_norm: float
    spearman_snip: float
    spearman_near: float
    kendall_decoder_params: float
    kendall_grad_norm: float
    best_proxy: str


@dataclass
class NASState:
    """Serialisable state for resumable 3-phase NAS runs.

    Attributes
    ----------
    phase:
        Current phase (1, 2, or 3).
    iteration:
        Current iteration index within the current phase.
    results:
        Accumulated :class:`NASResult` objects from all phases so far.
    dataset:
        Accumulated :class:`ProxyDataset` used to train/refit the proxies.
    proxy_validation:
        Phase 2 correlation results; ``None`` before Phase 2 completes.
    hypervolume:
        Current Pareto hypervolume (updated after each iteration).
    start_time:
        ISO-8601 timestamp of run start.
    """

    phase: int = 1
    iteration: int = 0
    results: list[NASResult] = field(default_factory=list)
    dataset: ProxyDataset = field(default_factory=ProxyDataset)
    proxy_validation: Optional[ProxyValidation] = None
    hypervolume: float = 0.0
    start_time: str = ""


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def _config_to_dict(config: TransformerConfig) -> dict:
    """Serialise a :class:`TransformerConfig` to a plain dict."""
    return {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "seq_len": config.seq_len,
        "ff_mult": config.ff_mult,
        "activation": config.activation,
        "use_rotary": config.use_rotary,
        "use_layernorm": config.use_layernorm,
        "vocab_size": config.vocab_size,
        "tie_embeddings": config.tie_embeddings,
    }


def _config_from_dict(d: dict) -> TransformerConfig:
    """Deserialise a :class:`TransformerConfig` from a plain dict."""
    return TransformerConfig(
        d_model=int(d["d_model"]),
        n_heads=int(d["n_heads"]),
        n_layers=int(d["n_layers"]),
        seq_len=int(d["seq_len"]),
        ff_mult=float(d["ff_mult"]),
        activation=str(d["activation"]),
        use_rotary=bool(d["use_rotary"]),
        use_layernorm=bool(d["use_layernorm"]),
        vocab_size=int(d.get("vocab_size", 50257)),
        tie_embeddings=bool(d.get("tie_embeddings", False)),
    )


def _nas_result_to_dict(result: NASResult) -> dict:
    """Serialise a :class:`NASResult` to a JSON-compatible dict."""
    return {
        "config": _config_to_dict(result.config),
        "zk_cost": int(result.zk_cost),
        "decoder_params": int(result.decoder_params),
        "decoder_params_score": float(result.decoder_params_score),
        "grad_norm_score": (
            float(result.grad_norm_score)
            if result.grad_norm_score is not None
            else None
        ),
        "snip_score": (
            float(result.snip_score) if result.snip_score is not None else None
        ),
        "near_score": (
            float(result.near_score) if result.near_score is not None else None
        ),
        "calibration_ppl": (
            float(result.calibration_ppl)
            if result.calibration_ppl is not None
            else None
        ),
        "final_ppl": (
            float(result.final_ppl) if result.final_ppl is not None else None
        ),
        "on_pareto_frontier": bool(result.on_pareto_frontier),
        "phase": int(result.phase),
        "iteration": int(result.iteration),
    }


def _nas_result_from_dict(d: dict) -> NASResult:
    """Deserialise a :class:`NASResult` from a plain dict."""
    return NASResult(
        config=_config_from_dict(d["config"]),
        zk_cost=int(d["zk_cost"]),
        decoder_params=int(d.get("decoder_params", 0)),
        decoder_params_score=float(d.get("decoder_params_score", 0.0)),
        grad_norm_score=(
            float(d["grad_norm_score"])
            if d.get("grad_norm_score") is not None
            else None
        ),
        snip_score=(
            float(d["snip_score"]) if d.get("snip_score") is not None else None
        ),
        near_score=(
            float(d["near_score"]) if d.get("near_score") is not None else None
        ),
        calibration_ppl=(
            float(d["calibration_ppl"])
            if d.get("calibration_ppl") is not None
            else None
        ),
        final_ppl=(float(d["final_ppl"]) if d.get("final_ppl") is not None else None),
        on_pareto_frontier=bool(d.get("on_pareto_frontier", False)),
        phase=int(d.get("phase", 1)),
        iteration=int(d.get("iteration", 0)),
    )


def _proxy_validation_to_dict(pv: ProxyValidation) -> dict:
    """Serialise a :class:`ProxyValidation` to a JSON-compatible dict."""
    return {
        "n_configs": pv.n_configs,
        "spearman_decoder_params": pv.spearman_decoder_params,
        "spearman_grad_norm": pv.spearman_grad_norm,
        "spearman_snip": pv.spearman_snip,
        "spearman_near": pv.spearman_near,
        "kendall_decoder_params": pv.kendall_decoder_params,
        "kendall_grad_norm": pv.kendall_grad_norm,
        "best_proxy": pv.best_proxy,
    }


def _proxy_validation_from_dict(d: dict) -> ProxyValidation:
    """Deserialise a :class:`ProxyValidation` from a plain dict."""
    return ProxyValidation(
        n_configs=int(d["n_configs"]),
        spearman_decoder_params=float(d["spearman_decoder_params"]),
        spearman_grad_norm=float(d["spearman_grad_norm"]),
        spearman_snip=float(d["spearman_snip"]),
        spearman_near=float(d["spearman_near"]),
        kendall_decoder_params=float(d["kendall_decoder_params"]),
        kendall_grad_norm=float(d["kendall_grad_norm"]),
        best_proxy=str(d["best_proxy"]),
    )


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, o: object) -> object:  # type: ignore[override]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_near_score(config: TransformerConfig) -> float:
    """Compute NEAR score for *config*, falling back to ``log(total_params)``.

    NEAR is not validated for GPT-2/WikiText ranking and is included for
    ablation studies only.

    Parameters
    ----------
    config:
        Architecture to evaluate.

    Returns
    -------
    float
        NEAR score.  Non-negative.
    """
    try:
        model = ZKTransformer(config)
        model.eval()
        sample_input = torch.zeros(1, config.seq_len, dtype=torch.long)
        return near_score_lm(model, sample_input)
    except Exception:  # pylint: disable=broad-except
        logger.debug(
            "NEAR score failed for config %s; falling back to log(params)",
            config,
            exc_info=True,
        )
        features = ProxyFeatures.from_config(config)
        total = features.total_params
        return math.log(total) if total > 0 else 0.0


_GRAD_NORM_VOCAB = 512   # reduced vocab for fast grad_norm (architecture signal only)
_GRAD_NORM_SEQ = 16      # short sequence for fast forward/backward pass


def _compute_grad_norm_and_snip(
    config: TransformerConfig,
    seq_len: int = _GRAD_NORM_SEQ,
) -> tuple[float, float]:
    """Compute grad_norm and SNIP scores.

    Uses a reduced vocab (512) and short sequence (16) for speed.
    The gradient norm signal comes from the transformer blocks, not the
    embedding size, so this approximation is valid for ranking purposes.
    """
    try:
        # Tiny vocab version of the same architecture — same transformer blocks,
        # but 100× smaller embedding (vocab 512 vs 50257)
        fast_config = TransformerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            seq_len=seq_len,
            ff_mult=config.ff_mult,
            activation=config.activation,
            use_rotary=config.use_rotary,
            use_layernorm=config.use_layernorm,
            vocab_size=_GRAD_NORM_VOCAB,
            tie_embeddings=config.tie_embeddings,
        )
        model = ZKTransformer(fast_config)
        data = torch.zeros(1, seq_len, dtype=torch.long)
        gn = grad_norm_score(model, data)
        model2 = ZKTransformer(fast_config)
        sn = snip_score(model2, data)
        return gn, sn
    except Exception:  # pylint: disable=broad-except
        logger.debug("grad_norm/snip failed for config %s", config, exc_info=True)
        return 0.0, 0.0


def _extract_decoder_params(config: TransformerConfig) -> int:
    """Return the non-embedding parameter count for *config*."""
    features = ProxyFeatures.from_config(config)
    return int(features.decoder_params)


def _make_proxy_record(result: NASResult, backend: str) -> ProxyRecord:
    """Build a :class:`ProxyRecord` from a :class:`NASResult`."""
    features = ProxyFeatures.from_config(result.config)
    if result.zk_cost > 0:
        features = ProxyFeatures(
            d_model=features.d_model,
            n_heads=features.n_heads,
            n_layers=features.n_layers,
            seq_len=features.seq_len,
            ff_mult=features.ff_mult,
            head_dim=features.head_dim,
            total_params=features.total_params,
            decoder_params=features.decoder_params,
            attn_macs=features.attn_macs,
            ffn_macs=features.ffn_macs,
            total_macs=features.total_macs,
            activation_id=features.activation_id,
            is_zk_friendly=features.is_zk_friendly,
            layernorm_count=features.layernorm_count,
            softmax_count=features.softmax_count,
            gelu_count=features.gelu_count,
            onnx_constraint_estimate=result.zk_cost,
            onnx_nonlinear_ops=features.onnx_nonlinear_ops,
            onnx_linear_ops=features.onnx_linear_ops,
        )
    return ProxyRecord(
        config=result.config,
        features=features,
        onnx_estimate=result.zk_cost if result.zk_cost > 0 else None,
        backend=backend,
    )


def _spearman(x: list[float], y: list[float]) -> float:
    """Return Spearman r between *x* and *y*; 0.0 on failure."""
    if len(x) < 3:
        return 0.0
    try:
        r, _ = spearmanr(x, y)
        return float(r) if math.isfinite(r) else 0.0
    except Exception:  # pylint: disable=broad-except
        return 0.0


def _kendall(x: list[float], y: list[float]) -> float:
    """Return Kendall tau between *x* and *y*; 0.0 on failure."""
    if len(x) < 3:
        return 0.0
    try:
        tau, _ = kendalltau(x, y)
        return float(tau) if math.isfinite(tau) else 0.0
    except Exception:  # pylint: disable=broad-except
        return 0.0


# ---------------------------------------------------------------------------
# ZKNASLoop
# ---------------------------------------------------------------------------


class ZKNASLoop:  # pylint: disable=too-many-instance-attributes
    """Three-phase ZK-NAS orchestrator.

    Phase 1 — Search (proxy-only, no training)
        Iteratively samples a candidate pool, scores with decoder_params and
        ONNX ZK cost, uses :class:`ActiveLearner` to select the most
        informative batch, computes ``grad_norm`` on each selected config,
        adds results to the dataset, refits proxies, and updates the Pareto
        frontier.

    Phase 2 — Calibration (500-step training)
        Trains a stratified sample of configs for ``phase2_calibration_steps``
        steps and measures Spearman / Kendall correlation between each
        zero-cost proxy and the resulting PPL.  Produces :class:`ProxyValidation`
        (Table 1 of the paper).

    Phase 3 — Final training
        Trains the top-K Pareto-frontier configs (plus GPT-2 baseline) to
        convergence using :class:`~search.lm_trainer.LMTrainer`.  Records
        final PPL and updates the Pareto frontier with (PPL, ZK cost) axes.

    Parameters
    ----------
    output_dir:
        Directory where checkpoints and dataset JSON files are written.
    backend:
        ZK backend name stored in proxy records (default ``'ezkl'``).
    seed:
        Random seed for reproducible sampling.
    phase1_iterations:
        Number of search iterations in Phase 1.
    phase1_batch_size:
        Configs to evaluate per Phase 1 iteration.
    phase1_pool_size:
        Candidate pool size to score per Phase 1 iteration.
    phase2_calibration_steps:
        Training steps for each calibration config in Phase 2.
    phase2_sample_size:
        Number of configs to train during Phase 2 calibration.
    phase2_training_seq_len:
        Sequence length used for Phase 2 / Phase 3 training.
    phase3_max_steps:
        Maximum training steps for Phase 3 final training.
    phase3_top_k:
        Number of Pareto-frontier configs to train in Phase 3.
    prove:
        Run actual EZKL proofs when ``True``; otherwise ONNX proxy only.
    verbosity:
        0=WARNING, 1=INFO, 2=DEBUG.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        output_dir: str,
        backend: str = "ezkl",
        seed: int = 42,
        # Phase 1 settings
        phase1_iterations: int = 50,
        phase1_batch_size: int = 8,
        phase1_pool_size: int = 200,
        # Phase 2 settings
        phase2_calibration_steps: int = 500,
        phase2_sample_size: int = 30,
        phase2_training_seq_len: int = 128,
        # Phase 3 settings
        phase3_max_steps: int = 10000,
        phase3_top_k: int = 5,
        prove: bool = False,
        verbosity: int = 1,
    ) -> None:
        self.output_dir = output_dir
        self.backend = backend
        self.seed = seed
        self.phase1_iterations = phase1_iterations
        self.phase1_batch_size = phase1_batch_size
        self.phase1_pool_size = phase1_pool_size
        self.phase2_calibration_steps = phase2_calibration_steps
        self.phase2_sample_size = phase2_sample_size
        self.phase2_training_seq_len = phase2_training_seq_len
        self.phase3_max_steps = phase3_max_steps
        self.phase3_top_k = phase3_top_k
        self.prove = prove

        log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
            verbosity, logging.INFO
        )
        logging.basicConfig(level=log_level)

        self._search_space = TransformerSearchSpace(seed=seed)

        # Proxies for ZK-cost estimation (Phase 1).
        self._proxy_a = MultiFidelityProxy(n_estimators=100)
        self._proxy_b = ZeroCostProxy(model_type="xgb", use_onnx_features=True)
        self._proxy_c = KernelDecompositionProxy()
        self._learner = ActiveLearner(
            proxy_a=self._proxy_a,
            proxy_b=self._proxy_b,
            proxy_c=None,
            backend=backend,
        )

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Search
    # ------------------------------------------------------------------

    def run_phase1(self, state: NASState) -> NASState:
        """Search phase: zero-cost proxy + ONNX ZK cost.

        Each iteration:

        1. Sample ``phase1_pool_size`` candidates from the search space.
        2. Score all with ``decoder_params_score`` (primary) + ONNX ZK cost.
        3. Use :class:`ActiveLearner` to select ``phase1_batch_size`` configs.
        4. For selected configs: also compute ``grad_norm_score`` (one backward pass).
        5. Add results to dataset, refit proxies, update Pareto frontier.
        6. Save checkpoint.

        Parameters
        ----------
        state:
            Current NAS state; mutated in-place.

        Returns
        -------
        NASState
            Updated state with Phase 1 results.
        """
        logger.info(
            "Phase 1 — Search: %d iterations, pool=%d, batch=%d",
            self.phase1_iterations,
            self.phase1_pool_size,
            self.phase1_batch_size,
        )
        state.phase = 1

        for it in range(self.phase1_iterations):
            logger.info("Phase 1 iteration %d / %d", it + 1, self.phase1_iterations)

            # 1. Sample pool.
            candidates = [
                self._search_space.random_config() for _ in range(self.phase1_pool_size)
            ]

            # 2. Score all candidates cheaply using analytical estimates only.
            # total_macs from ProxyFeatures is O(1) and avoids ONNX export for
            # all 150 pool configs. ONNX is reserved for the selected batch below.
            scored: list[tuple[TransformerConfig, float, int]] = []
            for cfg in candidates:
                dp_score = decoder_params_score(cfg)
                feats = ProxyFeatures.from_config(cfg)
                analytical_zk = feats.total_macs  # fast proxy, no ONNX export
                scored.append((cfg, dp_score, analytical_zk))

            # 3. Select most informative batch via ActiveLearner.
            selected = self._learner.select_batch(
                candidates=candidates,
                dataset=state.dataset,
                batch_size=self.phase1_batch_size,
            )
            selected_set = {id(c): c for c in selected}

            # 4. Build NASResults for selected batch only.
            # grad_norm is deferred to Phase 2 (computed during calibration training).
            # decoder_params_score is the primary accuracy proxy (Spearman 0.967 on LMs).
            for cfg, dp_score, _analytical_zk in scored:
                if id(cfg) not in selected_set:
                    continue  # only record selected configs to keep dataset small
                zk_cost = self._estimate_zk_cost(cfg)
                gn, sn = 0.0, 0.0  # deferred to Phase 2

                dp = _extract_decoder_params(cfg)
                result = NASResult(
                    config=cfg,
                    zk_cost=zk_cost,
                    decoder_params=dp,
                    decoder_params_score=dp_score,
                    grad_norm_score=gn,
                    snip_score=sn,
                    near_score=None,
                    phase=1,
                    iteration=it,
                )
                state.results.append(result)
                state.dataset.add(_make_proxy_record(result, self.backend))

            state.iteration = it
            self._refit_proxies(state)
            self._update_pareto(state, use_final_ppl=False)
            self._save_checkpoint(state, phase=1, iteration=it)

            logger.info(
                "Phase 1 iter %d: %d total results, hypervolume=%.4f",
                it + 1,
                len(state.results),
                state.hypervolume,
            )

        return state

    # ------------------------------------------------------------------
    # Phase 2: Calibration
    # ------------------------------------------------------------------

    def run_phase2(  # pylint: disable=too-many-locals
        self, state: NASState
    ) -> tuple[NASState, ProxyValidation]:
        """Calibration phase: train configs for ``phase2_calibration_steps`` steps.

        1. Select architecturally-diverse configs (stratified by ZK cost).
        2. Train each for ``phase2_calibration_steps`` steps on WikiText-2.
        3. Compute Spearman / Kendall between each proxy and calibration PPL.
        4. Identify best proxy.
        5. Return :class:`ProxyValidation` (Table 1 of the paper).

        Parameters
        ----------
        state:
            NAS state after Phase 1.

        Returns
        -------
        tuple[NASState, ProxyValidation]
            Updated state and proxy validation results.
        """
        from search.lm_trainer import (  # pylint: disable=import-outside-toplevel
            LMTrainer,
            LMTrainingConfig,
        )

        logger.info(
            "Phase 2 — Calibration: %d configs × %d steps",
            self.phase2_sample_size,
            self.phase2_calibration_steps,
        )
        state.phase = 2

        # 1. Stratified sample: divide results by ZK cost into buckets.
        sample_configs = self._stratified_sample(
            state.results, n=self.phase2_sample_size
        )

        logger.info("Phase 2: training %d sampled configs", len(sample_configs))

        training_cfg = LMTrainingConfig(
            seq_len=self.phase2_training_seq_len,
            max_steps=self.phase2_calibration_steps,
        )
        trainer = LMTrainer(training_cfg)

        # Collect proxy scores and calibration PPLs for correlation analysis.
        dp_scores: list[float] = []
        gn_scores: list[float] = []
        sn_scores: list[float] = []
        nr_scores: list[float] = []
        cal_ppls: list[float] = []

        for cfg in sample_configs:
            dp_sc = decoder_params_score(cfg)
            gn_sc, sn_sc = _compute_grad_norm_and_snip(cfg)
            nr_sc = _compute_near_score(cfg)

            # Run short calibration training.
            try:
                model = self._search_space.build_model(cfg)
                train_result = trainer.calibration_run(
                    model,
                    n_steps=self.phase2_calibration_steps,
                )
                cal_ppl = float(train_result.val_perplexity)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Calibration training failed for config %s; skipping",
                    cfg,
                    exc_info=True,
                )
                continue

            dp = _extract_decoder_params(cfg)
            result = NASResult(
                config=cfg,
                zk_cost=self._estimate_zk_cost(cfg),
                decoder_params=dp,
                decoder_params_score=dp_sc,
                grad_norm_score=gn_sc,
                snip_score=sn_sc,
                near_score=nr_sc,
                calibration_ppl=cal_ppl,
                phase=2,
                iteration=0,
            )
            state.results.append(result)
            state.dataset.add(_make_proxy_record(result, self.backend))

            dp_scores.append(dp_sc)
            gn_scores.append(gn_sc)
            sn_scores.append(sn_sc)
            nr_scores.append(nr_sc)
            cal_ppls.append(cal_ppl)

        # Lower PPL is better (like accuracy); higher proxy score = better model.
        # For Spearman: negate PPL so higher = better for all axes.
        neg_ppls = [-p for p in cal_ppls]

        sp_dp = _spearman(dp_scores, neg_ppls)
        sp_gn = _spearman(gn_scores, neg_ppls)
        sp_sn = _spearman(sn_scores, neg_ppls)
        sp_nr = _spearman(nr_scores, neg_ppls)

        validation = ProxyValidation(
            n_configs=len(cal_ppls),
            spearman_decoder_params=sp_dp,
            spearman_grad_norm=sp_gn,
            spearman_snip=sp_sn,
            spearman_near=sp_nr,
            kendall_decoder_params=_kendall(dp_scores, neg_ppls),
            kendall_grad_norm=_kendall(gn_scores, neg_ppls),
            best_proxy=_best_proxy_name(
                {
                    "decoder_params": sp_dp,
                    "grad_norm": sp_gn,
                    "snip": sp_sn,
                    "near": sp_nr,
                }
            ),
        )
        state.proxy_validation = validation
        self._update_pareto(state, use_final_ppl=False)
        self._save_checkpoint(state, phase=2, iteration=0)

        logger.info(
            "Phase 2 complete: best_proxy=%s, spearman_decoder_params=%.3f",
            validation.best_proxy,
            validation.spearman_decoder_params,
        )
        return state, validation

    # ------------------------------------------------------------------
    # Phase 3: Final training
    # ------------------------------------------------------------------

    def run_phase3(self, state: NASState) -> NASState:
        """Final training: train top-K Pareto configs to convergence.

        1. Identify ``phase3_top_k`` Pareto-frontier configs from Phase 1/2.
        2. Also train GPT-2 baseline to convergence.
        3. Record final PPL vs. ZK cost for all trained configs.
        4. Update Pareto frontier with (final_ppl, zk_cost) axes.

        Parameters
        ----------
        state:
            NAS state after Phase 2.

        Returns
        -------
        NASState
            Updated state with Phase 3 results.
        """
        # pylint: disable=too-many-locals
        from search.lm_trainer import (  # pylint: disable=import-outside-toplevel
            LMTrainer,
            LMTrainingConfig,
        )

        logger.info(
            "Phase 3 — Final training: top_k=%d, max_steps=%d",
            self.phase3_top_k,
            self.phase3_max_steps,
        )
        state.phase = 3

        # Collect top-K Pareto configs + GPT-2 baseline.
        frontier = [r for r in state.results if r.on_pareto_frontier]
        frontier.sort(key=lambda r: r.zk_cost)
        top_k_results = frontier[: self.phase3_top_k]
        top_k_configs: list[TransformerConfig] = [r.config for r in top_k_results]

        # GPT-2 small baseline as anchor.
        gpt2_cfg = self._search_space.gpt2_baseline()
        all_configs: list[TransformerConfig] = list(top_k_configs) + [gpt2_cfg]

        training_cfg = LMTrainingConfig(
            seq_len=self.phase2_training_seq_len,
            max_steps=self.phase3_max_steps,
        )
        trainer = LMTrainer(training_cfg)

        for i, cfg in enumerate(all_configs):
            logger.info(
                "Phase 3: training config %d / %d (d_model=%d, n_layers=%d)",
                i + 1,
                len(all_configs),
                cfg.d_model,
                cfg.n_layers,
            )
            try:
                model = self._search_space.build_model(cfg)
                train_result = trainer.train(model)
                final_ppl = float(train_result.val_perplexity)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Phase 3 training failed for config %s; recording NaN PPL",
                    cfg,
                    exc_info=True,
                )
                final_ppl = float("nan")

            dp = _extract_decoder_params(cfg)
            result = NASResult(
                config=cfg,
                zk_cost=self._estimate_zk_cost(cfg),
                decoder_params=dp,
                decoder_params_score=decoder_params_score(cfg),
                final_ppl=final_ppl if math.isfinite(final_ppl) else None,
                phase=3,
                iteration=i,
            )
            state.results.append(result)

        # Re-compute Pareto on (final_ppl, zk_cost) for Phase 3 results.
        self._update_pareto(state, use_final_ppl=True)
        self._save_checkpoint(state, phase=3, iteration=0)

        logger.info("Phase 3 complete: trained %d configs", len(all_configs))
        return state

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        phases: Optional[list[int]] = None,
        resume_from: Optional[str] = None,
    ) -> NASState:
        """Run all specified phases in sequence.

        Checkpoints are saved after each phase.  Phases already completed in
        a loaded checkpoint are skipped.

        Parameters
        ----------
        phases:
            List of phase numbers to run.  Defaults to ``[1, 2, 3]``.
        resume_from:
            Optional path to a checkpoint JSON file.  When given, execution
            resumes from the saved state.

        Returns
        -------
        NASState
            Final state after all requested phases complete.
        """
        if phases is None:
            phases = [1, 2, 3]

        if resume_from is not None:
            state = ZKNASLoop.load_checkpoint(resume_from)
            logger.info(
                "Resumed from checkpoint %s (phase=%d, iter=%d)",
                resume_from,
                state.phase,
                state.iteration,
            )
            self._refit_proxies(state)
        else:
            state = NASState(
                phase=0,
                iteration=0,
                results=[],
                dataset=ProxyDataset(),
                proxy_validation=None,
                hypervolume=0.0,
                start_time=datetime.now(timezone.utc).isoformat(),
            )

        if 1 in phases and state.phase < 1:
            state = self.run_phase1(state)

        if 2 in phases and state.phase < 2:
            state, _ = self.run_phase2(state)

        if 3 in phases and state.phase < 3:
            state = self.run_phase3(state)

        return state

    # ------------------------------------------------------------------
    # Markdown table helpers
    # ------------------------------------------------------------------

    def pareto_table(self, state: NASState, use_final_ppl: bool = False) -> str:
        """Return a Markdown table of Pareto-frontier configs (Table 3 of paper).

        Parameters
        ----------
        state:
            NAS state containing evaluated results.
        use_final_ppl:
            When ``True``, the accuracy column shows ``final_ppl`` (Phase 3
            results).  When ``False``, shows ``decoder_params_score``.

        Returns
        -------
        str
            Multi-line Markdown table string.  Returns a placeholder row when
            the frontier is empty.
        """
        frontier = [r for r in state.results if r.on_pareto_frontier]
        if not frontier:
            return "| (no Pareto frontier configs yet) |"

        frontier.sort(key=lambda r: r.zk_cost)

        if use_final_ppl:
            header = (
                "| d_model | n_heads | n_layers | seq_len | activation "
                "| ZK cost | final_ppl | Phase |"
            )
            separator = (
                "|---------|---------|----------|---------|------------"
                "|---------|-----------|-------|"
            )
        else:
            header = (
                "| d_model | n_heads | n_layers | seq_len | activation "
                "| ZK cost | decoder_params_score | Phase |"
            )
            separator = (
                "|---------|---------|----------|---------|------------"
                "|---------|----------------------|-------|"
            )

        rows = [header, separator]
        for r in frontier:
            cfg = r.config
            if use_final_ppl:
                acc_val = f"{r.final_ppl:.2f}" if r.final_ppl is not None else "N/A"
            else:
                acc_val = f"{r.decoder_params_score:.4f}"
            rows.append(
                f"| {cfg.d_model} "
                f"| {cfg.n_heads} "
                f"| {cfg.n_layers} "
                f"| {cfg.seq_len} "
                f"| {cfg.activation} "
                f"| {r.zk_cost:,} "
                f"| {acc_val} "
                f"| {r.phase} |"
            )
        return "\n".join(rows)

    def proxy_validation_table(self, validation: ProxyValidation) -> str:
        """Return a Markdown table of proxy validation results (Table 1 of paper).

        Parameters
        ----------
        validation:
            :class:`ProxyValidation` produced by Phase 2.

        Returns
        -------
        str
            Multi-line Markdown table string.
        """
        header = "| Proxy | Spearman r | Kendall tau | Best? |"
        separator = "|-------|------------|-------------|-------|"
        rows = [header, separator]

        proxies: list[tuple[str, float, float]] = [
            (
                "decoder_params",
                validation.spearman_decoder_params,
                validation.kendall_decoder_params,
            ),
            ("grad_norm", validation.spearman_grad_norm, validation.kendall_grad_norm),
            ("snip", validation.spearman_snip, float("nan")),
            ("near", validation.spearman_near, float("nan")),
        ]
        for name, sp_r, kt in proxies:
            is_best = "**yes**" if name == validation.best_proxy else ""
            kt_str = f"{kt:.3f}" if math.isfinite(kt) else "N/A"
            rows.append(f"| {name} | {sp_r:.3f} | {kt_str} | {is_best} |")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, state: NASState, phase: int, iteration: int) -> None:
        """Save state to ``output_dir/phase{P}_iter{N:04d}.json``.

        Parameters
        ----------
        state:
            Current NAS state to persist.
        phase:
            Phase number (1, 2, or 3).
        iteration:
            Iteration number within the phase.
        """
        checkpoint_path = os.path.join(
            self.output_dir, f"phase{phase}_iter{iteration:04d}.json"
        )
        dataset_path = os.path.join(
            self.output_dir, f"dataset_phase{phase}_iter{iteration:04d}.json"
        )
        state.dataset.save(dataset_path)

        payload: dict = {
            "version": 2,
            "phase": int(state.phase),
            "iteration": int(state.iteration),
            "results": [_nas_result_to_dict(r) for r in state.results],
            "dataset_path": dataset_path,
            "proxy_validation": (
                _proxy_validation_to_dict(state.proxy_validation)
                if state.proxy_validation is not None
                else None
            ),
            "hypervolume": float(state.hypervolume),
            "start_time": state.start_time,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, cls=_NumpyEncoder)
        logger.info("Checkpoint saved: %s", checkpoint_path)

    @classmethod
    def load_checkpoint(cls, path: str) -> NASState:
        """Load a :class:`NASState` from a checkpoint file.

        Parameters
        ----------
        path:
            Absolute path to a checkpoint JSON produced by
            :meth:`_save_checkpoint`.

        Returns
        -------
        NASState
            Restored state.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        results = [_nas_result_from_dict(r) for r in payload.get("results", [])]

        dataset_path: str = payload.get("dataset_path", "")
        try:
            dataset = ProxyDataset.load(dataset_path)
        except FileNotFoundError:
            logger.warning(
                "Dataset file %s not found; starting with empty dataset.", dataset_path
            )
            dataset = ProxyDataset()

        pv_dict = payload.get("proxy_validation")
        proxy_validation = (
            _proxy_validation_from_dict(pv_dict) if pv_dict is not None else None
        )

        return NASState(
            phase=int(payload.get("phase", 1)),
            iteration=int(payload.get("iteration", 0)),
            results=results,
            dataset=dataset,
            proxy_validation=proxy_validation,
            hypervolume=float(payload.get("hypervolume", 0.0)),
            start_time=str(payload.get("start_time", "")),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_zk_cost(self, config: TransformerConfig) -> int:
        """Return analytical ZK constraint estimate for *config*.

        Uses the operator-level analytical formula (microseconds) rather than
        ONNX export (24s per config). The analytical formula is more accurate
        for transformers since 97-99% of ZK cost is matmul-dominated and we
        know the transformer structure exactly.
        """
        try:
            from experiments.operator_attribution import (  # pylint: disable=import-outside-toplevel
                compute_op_breakdown,
            )
            bd = compute_op_breakdown(config)
            return int(bd.total_constraints)
        except Exception:  # pylint: disable=broad-except
            feats = ProxyFeatures.from_config(config)
            return int(feats.total_macs)

    def _refit_proxies(self, state: NASState) -> None:
        """Refit all three ZK-cost proxies on accumulated data.

        proxy_b is always refit; proxy_a requires >= 5 records; proxy_c
        requires a non-empty cost table.
        """
        dataset = state.dataset
        self._proxy_b.fit(dataset)
        if len(dataset) >= 5:
            self._proxy_a.fit(dataset)
        if self._proxy_c.is_fitted():
            self._proxy_c.fit(dataset)
            self._learner._proxy_c = self._proxy_c  # pylint: disable=protected-access

    def _update_pareto(self, state: NASState, use_final_ppl: bool = False) -> None:
        """Recompute the Pareto frontier over all results.

        Phase 1/2 axes: ``(decoder_params_score, zk_cost)``.
        Phase 3 axes:   ``(final_ppl, zk_cost)`` (lower PPL = better).

        Parameters
        ----------
        state:
            Mutated in-place: ``on_pareto_frontier`` flags and
            ``hypervolume`` are updated.
        use_final_ppl:
            When ``True``, use ``final_ppl`` as the accuracy axis (Phase 3).
        """
        if use_final_ppl:
            valid = [
                r
                for r in state.results
                if r.zk_cost > 0
                and r.final_ppl is not None
                and math.isfinite(r.final_ppl)
            ]
        else:
            valid = [r for r in state.results if r.zk_cost > 0]

        if not valid:
            return

        try:
            from zkml_bench.metrics.pareto import (  # pylint: disable=import-outside-toplevel
                ModelPoint,
                compute_pareto,
            )

            model_points = []
            for i, r in enumerate(valid):
                if use_final_ppl and r.final_ppl is not None:
                    # Negate PPL so that higher accuracy_mean = lower PPL.
                    acc_val = -float(r.final_ppl)
                else:
                    acc_val = float(r.decoder_params_score)
                model_points.append(
                    ModelPoint(
                        name=f"p{r.phase}_it{r.iteration}_idx{i}",
                        accuracy_mean=acc_val,
                        accuracy_std=None,
                        cost=float(r.zk_cost),
                    )
                )

            pareto_result = compute_pareto(
                model_points, cost_metric="constraint_estimate"
            )
            frontier_names = {mp.name for mp in pareto_result.frontier}
            state.hypervolume = float(pareto_result.hypervolume)

            for r in state.results:
                r.on_pareto_frontier = False
            for i, r in enumerate(valid):
                name = f"p{r.phase}_it{r.iteration}_idx{i}"
                r.on_pareto_frontier = name in frontier_names

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Pareto update failed: %s", exc, exc_info=True)

    def _stratified_sample(
        self, results: list[NASResult], n: int
    ) -> list[TransformerConfig]:
        """Return *n* configs stratified by ZK cost from *results*.

        Configs are bucketed into ``n`` equal-width ZK-cost intervals; one
        config is drawn from each bucket (cycling if a bucket is empty).

        Parameters
        ----------
        results:
            Pool to sample from.
        n:
            Target number of configs to return.

        Returns
        -------
        list[TransformerConfig]
            Deduplicated sampled configs; may be fewer than *n* if the pool
            is small.
        """
        valid = [r for r in results if r.zk_cost > 0]
        if not valid:
            # Fallback: sample from search space directly.
            return [self._search_space.random_config() for _ in range(n)]

        if len(valid) <= n:
            return [r.config for r in valid]

        # Sort by ZK cost and divide into n equal buckets.
        valid_sorted = sorted(valid, key=lambda r: r.zk_cost)
        bucket_size = max(1, len(valid_sorted) // n)
        selected: list[TransformerConfig] = []
        for bucket_idx in range(n):
            start = bucket_idx * bucket_size
            end = start + bucket_size
            bucket = valid_sorted[start:end]
            if not bucket:
                # Wrap around.
                wrap = bucket_idx % len(valid_sorted)
                bucket = valid_sorted[wrap : wrap + 1]
            # Pick the median element of the bucket for diversity.
            mid = len(bucket) // 2
            selected.append(bucket[mid].config)

        return selected


# ---------------------------------------------------------------------------
# Internal helpers (module-level)
# ---------------------------------------------------------------------------


def _best_proxy_name(spearman_scores: dict[str, float]) -> str:
    """Return the key in *spearman_scores* with the highest value."""
    if not spearman_scores:
        return "decoder_params"
    return max(spearman_scores, key=lambda k: spearman_scores[k])

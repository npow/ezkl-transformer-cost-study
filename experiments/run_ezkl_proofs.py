#!/usr/bin/env python3
"""EZKL proof measurement for the 6 ZK-NAS Pareto configs.

This script measures actual EZKL circuit costs for the 6 Pareto-frontier
transformer configurations at seq_len=64 (feasible for proof generation;
seq_len=512 would require many hours per config).

Measured metrics (in order of increasing time):
  1. ONNX proxy: constraint_estimate, num_rows_snapped (seconds per config)
  2. EZKL calibration: logrows, num_rows from gen_settings + calibrate_settings
     (3-15 minutes per config)
  3. Full EZKL proof: setup_time_s, prove_time_s, proof_size_bytes
     (30-180 minutes per config; skipped by default)

Results are saved to experiments/results/ezkl_proofs/.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, "/root/code/zkml-bench/src")

import torch
import onnx
import ezkl

from proxy.features import TransformerConfig
from search.transformer_space import ZKTransformer
from zkml_bench.metrics.zk_proxy import compute_proxy
from experiments.operator_attribution import compute_op_breakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results" / "ezkl_proofs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pareto configs (seq_len=64 for EZKL feasibility)
# ---------------------------------------------------------------------------

SEQ_LEN = 64
VOCAB_SIZE = 50257
# GPT-2 baseline analytical ZK cost at seq_len=512 (from operator_attribution)
GPT2_COST_512 = 98_611_619_328

CONFIGS: dict[str, TransformerConfig] = {
    "gpt2_baseline": TransformerConfig(
        768, 12, 12, SEQ_LEN, 4.0, "gelu", False, True, VOCAB_SIZE, True
    ),
    "zk_2x": TransformerConfig(
        640, 8, 10, SEQ_LEN, 4.0, "gelu", True, True, VOCAB_SIZE, True
    ),
    "zk_3x": TransformerConfig(
        512, 8, 8, SEQ_LEN, 4.0, "relu", True, True, VOCAB_SIZE, True
    ),
    "zk_5x": TransformerConfig(
        384, 6, 10, SEQ_LEN, 4.0, "poly2", True, True, VOCAB_SIZE, True
    ),
    "zk_8x": TransformerConfig(
        384, 6, 6, SEQ_LEN, 4.0, "linear", True, True, VOCAB_SIZE, True
    ),
    "zk_11x": TransformerConfig(
        384, 6, 4, SEQ_LEN, 4.0, "linear", True, True, VOCAB_SIZE, True
    ),
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EZKLProofResult:
    """Measured metrics for one Pareto config."""

    name: str
    seq_len: int

    # Architecture
    d_model: int
    n_heads: int
    n_layers: int
    activation: str

    # ONNX proxy metrics (fast, always measured)
    proxy_constraint_estimate: int
    proxy_num_rows_snapped: int
    proxy_time_s: float
    proxy_method: str

    # Analytical cost from operator_attribution (at seq_len=64)
    analytical_cost_64: int
    # Analytical cost at seq_len=512 (for comparison with paper table)
    analytical_cost_512: int

    # EZKL calibration metrics (gen_settings + calibrate_settings)
    ezkl_logrows: Optional[int]
    ezkl_num_rows: Optional[int]
    calibration_time_s: Optional[float]
    calibration_status: str  # "ok" | "failed" | "skipped"

    # Full EZKL proof metrics (optional; requires hours)
    setup_time_s: Optional[float]
    prove_time_s: Optional[float]
    verify_time_s: Optional[float]
    proof_size_bytes: Optional[int]
    peak_memory_mb: Optional[float]
    proof_status: str  # "ok" | "failed" | "skipped"

    # Derived
    speedup_vs_gpt2_proxy: Optional[float]
    speedup_vs_gpt2_ezkl: Optional[float]


# ---------------------------------------------------------------------------
# ONNX export helper
# ---------------------------------------------------------------------------


def export_onnx(model: torch.nn.Module, sample_input: torch.Tensor, path: str) -> None:
    """Export model to ONNX at *path*, suppressing noisy torch.onnx logs."""
    buf = io.BytesIO()
    with warnings.catch_warnings(), torch.no_grad():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            model,
            sample_input,
            buf,
            opset_version=18,
            input_names=["input_ids"],
            output_names=["logits"],
        )
    buf.seek(0)
    model_proto = onnx.load_from_string(buf.read())
    onnx.save(model_proto, path)


# ---------------------------------------------------------------------------
# EZKL calibration-only mode (get circuit metadata without full proof)
# ---------------------------------------------------------------------------


def run_ezkl_calibration(
    onnx_path: str,
    seq_len: int,
    tmpdir: str,
    n_cal_samples: int = 4,
) -> tuple[Optional[int], Optional[int], float, str]:
    """Run gen_settings + calibrate_settings to get circuit metadata.

    Parameters
    ----------
    onnx_path:
        Path to the exported ONNX model.
    seq_len:
        Sequence length (used to generate calibration data).
    tmpdir:
        Temporary directory for EZKL intermediate files.
    n_cal_samples:
        Number of calibration samples (4 is sufficient for metadata).

    Returns
    -------
    tuple of (logrows, num_rows, elapsed_s, status)
        logrows and num_rows are None if calibration fails.
        status is "ok" or "failed".
    """
    settings_path = os.path.join(tmpdir, "settings.json")
    cal_json = os.path.join(tmpdir, "cal_data.json")

    # Write calibration data: list of flat int token lists
    cal_data = {"input_data": [[0] * seq_len for _ in range(n_cal_samples)]}
    with open(cal_json, "w") as f:
        json.dump(cal_data, f)

    t0 = time.perf_counter()

    # Step 1: gen-settings
    try:
        ok = ezkl.gen_settings(model=onnx_path, output=settings_path)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("gen_settings raised: %s", exc)
        return None, None, elapsed, "failed"

    if not ok or not os.path.exists(settings_path):
        elapsed = time.perf_counter() - t0
        logger.error("gen_settings returned %s; settings file missing", ok)
        return None, None, elapsed, "failed"

    # Step 2: calibrate-settings
    try:
        ok = ezkl.calibrate_settings(
            data=cal_json,
            model=onnx_path,
            settings=settings_path,
            target="resources",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("calibrate_settings raised: %s", exc)
        return None, None, elapsed, "failed"

    elapsed = time.perf_counter() - t0

    if not ok or not os.path.exists(settings_path):
        logger.error("calibrate_settings returned %s", ok)
        return None, None, elapsed, "failed"

    # Parse logrows from settings
    with open(settings_path) as f:
        settings = json.load(f)
    logrows = settings.get("run_args", {}).get("logrows")
    if logrows is None:
        logger.error("logrows missing from settings.json")
        return None, None, elapsed, "failed"

    num_rows = 2**logrows
    return logrows, num_rows, elapsed, "ok"


# ---------------------------------------------------------------------------
# Full EZKL proof (optional)
# ---------------------------------------------------------------------------


def run_ezkl_full_proof(
    onnx_path: str,
    seq_len: int,
    tmpdir: str,
    cache_dir: str,
) -> dict[str, object]:
    """Run the full EZKL prove pipeline and return timing/size metrics.

    Parameters
    ----------
    onnx_path:
        Path to ONNX model.
    seq_len:
        Sequence length for witness generation.
    tmpdir:
        Directory for ephemeral intermediate files.
    cache_dir:
        Directory for cached SRS and key files.

    Returns
    -------
    dict with keys: setup_time_s, prove_time_s, verify_time_s,
    proof_size_bytes, peak_memory_mb, status.
    """
    import threading
    import psutil

    settings_path = os.path.join(tmpdir, "settings.json")
    cal_json = os.path.join(tmpdir, "cal_data.json")
    circuit_path = os.path.join(tmpdir, "circuit.ezkl")
    input_json = os.path.join(tmpdir, "input.json")
    witness_json = os.path.join(tmpdir, "witness.json")
    proof_json = os.path.join(tmpdir, "proof.json")
    srs_path = os.path.join(cache_dir, "kzg.srs")
    vk_path = os.path.join(tmpdir, "vk.key")
    pk_path = os.path.join(tmpdir, "pk.key")

    os.makedirs(cache_dir, exist_ok=True)

    # Write input and calibration data
    token_seq = [0] * seq_len
    with open(input_json, "w") as f:
        json.dump({"input_data": [token_seq]}, f)
    with open(cal_json, "w") as f:
        json.dump({"input_data": [token_seq] * 4}, f)

    # Memory polling
    peak_mb = [0.0]
    stop_mem = threading.Event()

    def _poll() -> None:
        proc = psutil.Process()
        while not stop_mem.is_set():
            try:
                mem = proc.memory_info().rss / (1024 * 1024)
                if mem > peak_mb[0]:
                    peak_mb[0] = mem
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)

    def _fail(msg: str) -> dict[str, object]:
        stop_mem.set()
        return {
            "setup_time_s": None,
            "prove_time_s": None,
            "verify_time_s": None,
            "proof_size_bytes": None,
            "peak_memory_mb": None,
            "status": f"failed:{msg}",
        }

    # gen-settings + calibrate
    try:
        if not ezkl.gen_settings(model=onnx_path, output=settings_path):
            return _fail("gen_settings")
        if not ezkl.calibrate_settings(
            data=cal_json,
            model=onnx_path,
            settings=settings_path,
            target="resources",
        ):
            return _fail("calibrate_settings")
    except Exception as exc:
        return _fail(f"calibrate:{exc}")

    # compile-circuit
    try:
        if not ezkl.compile_circuit(
            model=onnx_path,
            compiled_circuit=circuit_path,
            settings_path=settings_path,
        ):
            return _fail("compile_circuit")
    except Exception as exc:
        return _fail(f"compile:{exc}")

    # get-srs (cached)
    if not os.path.exists(srs_path):
        try:
            if not ezkl.get_srs(settings_path=settings_path, srs_path=srs_path):
                return _fail("get_srs")
        except Exception as exc:
            return _fail(f"get_srs:{exc}")

    # Start memory polling
    mem_thread = threading.Thread(target=_poll, daemon=True)
    mem_thread.start()

    # setup
    t_setup = time.perf_counter()
    try:
        ok = ezkl.setup(
            model=circuit_path,
            vk_path=vk_path,
            pk_path=pk_path,
            srs_path=srs_path,
        )
    except Exception as exc:
        return _fail(f"setup:{exc}")
    setup_time = time.perf_counter() - t_setup

    if not ok:
        return _fail("setup_returned_false")

    # gen-witness
    try:
        if not ezkl.gen_witness(
            data=input_json,
            model=circuit_path,
            output=witness_json,
        ):
            return _fail("gen_witness")
    except Exception as exc:
        return _fail(f"gen_witness:{exc}")

    # prove
    t_prove = time.perf_counter()
    try:
        ok = ezkl.prove(
            witness=witness_json,
            model=circuit_path,
            pk_path=pk_path,
            proof_path=proof_json,
            proof_type="single",
            srs_path=srs_path,
        )
    except Exception as exc:
        return _fail(f"prove:{exc}")
    prove_time = time.perf_counter() - t_prove

    if not ok:
        return _fail("prove_returned_false")

    # verify
    t_verify = time.perf_counter()
    try:
        verified = ezkl.verify(
            proof_path=proof_json,
            settings_path=settings_path,
            vk_path=vk_path,
            srs_path=srs_path,
        )
    except Exception as exc:
        return _fail(f"verify:{exc}")
    verify_time = time.perf_counter() - t_verify

    stop_mem.set()

    proof_size = (
        os.path.getsize(proof_json) if os.path.exists(proof_json) else None
    )

    return {
        "setup_time_s": setup_time,
        "prove_time_s": prove_time,
        "verify_time_s": verify_time,
        "proof_size_bytes": proof_size,
        "peak_memory_mb": peak_mb[0],
        "status": "ok" if verified else "failed:verify",
    }


# ---------------------------------------------------------------------------
# Per-config measurement
# ---------------------------------------------------------------------------


def measure_config(
    name: str,
    cfg: TransformerConfig,
    run_full_proof: bool = False,
    proof_cache_dir: Optional[str] = None,
) -> EZKLProofResult:
    """Measure all available metrics for one config.

    Parameters
    ----------
    name:
        Config name (used for logging and result key).
    cfg:
        Transformer architectural config (must have seq_len=64).
    run_full_proof:
        If True, attempt the full prove pipeline after calibration.
    proof_cache_dir:
        Directory for caching SRS files across runs.
    """
    logger.info("=== Measuring %s ===", name)

    model = ZKTransformer(cfg).eval()
    sample_input = torch.zeros(1, cfg.seq_len, dtype=torch.long)

    # ------------------------------------------------------------------
    # 1. ONNX proxy (always runs; takes seconds)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    proxy_result = compute_proxy(model, sample_input)
    proxy_time = time.perf_counter() - t0
    logger.info(
        "%s proxy: constraint_estimate=%d, snapped=%d, method=%s, time=%.2fs",
        name,
        proxy_result.constraint_estimate,
        proxy_result.constraint_estimate_snapped,
        proxy_result.proxy_method,
        proxy_time,
    )

    # ------------------------------------------------------------------
    # 2. Analytical cost (at seq_len=64 and seq_len=512)
    # ------------------------------------------------------------------
    bd_64 = compute_op_breakdown(cfg)
    analytical_cost_64 = bd_64.total_constraints

    cfg_512 = TransformerConfig(
        cfg.d_model, cfg.n_heads, cfg.n_layers, 512,
        cfg.ff_mult, cfg.activation, cfg.use_rotary,
        cfg.use_layernorm, cfg.vocab_size, cfg.tie_embeddings,
    )
    bd_512 = compute_op_breakdown(cfg_512)
    analytical_cost_512 = bd_512.total_constraints

    # ------------------------------------------------------------------
    # 3. Export ONNX, then run EZKL calibration
    # ------------------------------------------------------------------
    ezkl_logrows: Optional[int] = None
    ezkl_num_rows: Optional[int] = None
    calibration_time_s: Optional[float] = None
    calibration_status = "skipped"

    setup_time_s: Optional[float] = None
    prove_time_s: Optional[float] = None
    verify_time_s: Optional[float] = None
    proof_size_bytes: Optional[int] = None
    peak_memory_mb: Optional[float] = None
    proof_status = "skipped"

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")

        logger.info("%s: exporting ONNX...", name)
        try:
            export_onnx(model, sample_input, onnx_path)
            onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info("%s: ONNX exported (%.1f MB)", name, onnx_size_mb)
        except Exception as exc:
            logger.error("%s: ONNX export failed: %s", name, exc)
            calibration_status = "failed:onnx_export"
        else:
            logger.info("%s: running EZKL calibration...", name)
            ezkl_logrows, ezkl_num_rows, calibration_time_s, calibration_status = (
                run_ezkl_calibration(onnx_path, cfg.seq_len, tmpdir)
            )
            logger.info(
                "%s: calibration %s — logrows=%s, num_rows=%s, time=%.1fs",
                name,
                calibration_status,
                ezkl_logrows,
                ezkl_num_rows,
                calibration_time_s or 0.0,
            )

            if run_full_proof and calibration_status == "ok":
                logger.info("%s: running full EZKL proof...", name)
                cache_dir = (
                    proof_cache_dir or str(RESULTS_DIR / "srs_cache" / name)
                )
                proof_metrics = run_ezkl_full_proof(
                    onnx_path,
                    cfg.seq_len,
                    tmpdir,
                    cache_dir,
                )
                proof_status = proof_metrics["status"]
                setup_time_s = proof_metrics["setup_time_s"]
                prove_time_s = proof_metrics["prove_time_s"]
                verify_time_s = proof_metrics["verify_time_s"]
                proof_size_bytes = proof_metrics["proof_size_bytes"]
                peak_memory_mb = proof_metrics["peak_memory_mb"]
                logger.info(
                    "%s: proof %s — setup=%.1fs, prove=%.1fs",
                    name,
                    proof_status,
                    setup_time_s or 0.0,
                    prove_time_s or 0.0,
                )

    # ------------------------------------------------------------------
    # Derived: speedup vs GPT-2 (filled in by print_summary_table)
    # ------------------------------------------------------------------
    speedup_vs_gpt2_proxy: Optional[float] = None
    speedup_vs_gpt2_ezkl: Optional[float] = None

    return EZKLProofResult(
        name=name,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        activation=cfg.activation,
        proxy_constraint_estimate=proxy_result.constraint_estimate,
        proxy_num_rows_snapped=proxy_result.constraint_estimate_snapped,
        proxy_time_s=proxy_time,
        proxy_method=proxy_result.proxy_method,
        analytical_cost_64=analytical_cost_64,
        analytical_cost_512=analytical_cost_512,
        ezkl_logrows=ezkl_logrows,
        ezkl_num_rows=ezkl_num_rows,
        calibration_time_s=calibration_time_s,
        calibration_status=calibration_status,
        setup_time_s=setup_time_s,
        prove_time_s=prove_time_s,
        verify_time_s=verify_time_s,
        proof_size_bytes=proof_size_bytes,
        peak_memory_mb=peak_memory_mb,
        proof_status=proof_status,
        speedup_vs_gpt2_proxy=speedup_vs_gpt2_proxy,
        speedup_vs_gpt2_ezkl=speedup_vs_gpt2_ezkl,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(results: list[EZKLProofResult]) -> None:
    """Print the paper-ready summary table to stdout."""
    gpt2 = next(r for r in results if r.name == "gpt2_baseline")

    # Fill speedup fields
    for r in results:
        r.speedup_vs_gpt2_proxy = (
            gpt2.proxy_constraint_estimate / r.proxy_constraint_estimate
        )
        if gpt2.ezkl_num_rows and r.ezkl_num_rows:
            r.speedup_vs_gpt2_ezkl = gpt2.ezkl_num_rows / r.ezkl_num_rows

    header = (
        f"{'Config':<16} {'seq':<5} {'analytical_512':>16} "
        f"{'proxy_estimate':>16} {'ezkl_rows':>12} "
        f"{'logrows':>8} {'cal_time':>10} {'speedup_proxy':>14} {'speedup_ezkl':>13}"
    )
    sep = "-" * len(header)
    print()
    print("=== ZK-NAS Pareto Configs: EZKL Circuit Metadata ===")
    print(f"seq_len for EZKL = {SEQ_LEN} (feasibility; full proofs at this seq_len)")
    print()
    print(header)
    print(sep)
    for r in results:
        ezkl_rows_str = f"{r.ezkl_num_rows:,}" if r.ezkl_num_rows else "N/A"
        logrows_str = str(r.ezkl_logrows) if r.ezkl_logrows else "N/A"
        cal_str = f"{r.calibration_time_s:.0f}s" if r.calibration_time_s else "N/A"
        speedup_p = (
            f"{r.speedup_vs_gpt2_proxy:.1f}x" if r.speedup_vs_gpt2_proxy else "N/A"
        )
        speedup_e = (
            f"{r.speedup_vs_gpt2_ezkl:.1f}x" if r.speedup_vs_gpt2_ezkl else "N/A"
        )
        print(
            f"{r.name:<16} {r.seq_len:<5} {r.analytical_cost_512:>16,} "
            f"{r.proxy_constraint_estimate:>16,} {ezkl_rows_str:>12} "
            f"{logrows_str:>8} {cal_str:>10} {speedup_p:>14} {speedup_e:>13}"
        )
    print(sep)

    # Full proof table (if any completed)
    proven = [r for r in results if r.proof_status == "ok"]
    if proven:
        print()
        print("=== Full EZKL Proof Timings ===")
        hdr2 = (
            f"{'Config':<16} {'setup_s':>10} {'prove_s':>10} "
            f"{'verify_s':>10} {'proof_kb':>10} {'mem_mb':>10}"
        )
        print(hdr2)
        print("-" * len(hdr2))
        for r in proven:
            proof_kb = (
                f"{r.proof_size_bytes / 1024:.1f}" if r.proof_size_bytes else "N/A"
            )
            mem_str = f"{r.peak_memory_mb:.0f}" if r.peak_memory_mb else "N/A"
            print(
                f"{r.name:<16} {r.setup_time_s or 0:>10.1f} "
                f"{r.prove_time_s or 0:>10.1f} {r.verify_time_s or 0:>10.1f} "
                f"{proof_kb:>10} {mem_str:>10}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(run_full_proof: bool = False) -> None:
    """Run measurements for all 6 Pareto configs.

    Parameters
    ----------
    run_full_proof:
        Set True to attempt the full prove pipeline after calibration.
        Each config may take 30-180 minutes; disabled by default.
    """
    logger.info(
        "Starting EZKL measurements: seq_len=%d, full_proof=%s",
        SEQ_LEN,
        run_full_proof,
    )

    results: list[EZKLProofResult] = []
    for name, cfg in CONFIGS.items():
        result = measure_config(name, cfg, run_full_proof=run_full_proof)
        results.append(result)

        # Save incremental JSON after each config
        out_path = RESULTS_DIR / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("Saved %s", out_path)

    # Fill speedup fields and print summary
    print_summary_table(results)

    # Save combined JSON
    all_path = RESULTS_DIR / "all_results.json"
    with open(all_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info("Saved combined results to %s", all_path)

    # Save markdown table
    _save_markdown(results, RESULTS_DIR / "summary_table.md")
    logger.info("Done. Results in %s", RESULTS_DIR)


def _save_markdown(results: list[EZKLProofResult], path: Path) -> None:
    """Write a GitHub-flavoured markdown table for the paper."""
    lines: list[str] = [
        "# EZKL Circuit Metadata — ZK-NAS Pareto Configs",
        "",
        f"seq_len = {SEQ_LEN} (feasibility; full proofs at this seq_len)",
        "",
        "| Config | seq_len | analytical_cost_512 | proxy_estimate | ezkl_logrows"
        " | ezkl_num_rows | cal_time_s | speedup_proxy | speedup_ezkl |",
        "|--------|---------|---------------------|----------------|-------------|"
        "---------------|------------|---------------|--------------|",
    ]
    for r in results:
        speedup_p = (
            f"{r.speedup_vs_gpt2_proxy:.1f}x" if r.speedup_vs_gpt2_proxy else "N/A"
        )
        speedup_e = (
            f"{r.speedup_vs_gpt2_ezkl:.1f}x" if r.speedup_vs_gpt2_ezkl else "N/A"
        )
        ezkl_rows = f"{r.ezkl_num_rows:,}" if r.ezkl_num_rows else "N/A"
        logrows_str = str(r.ezkl_logrows) if r.ezkl_logrows else "N/A"
        cal_str = f"{r.calibration_time_s:.0f}" if r.calibration_time_s else "N/A"
        lines.append(
            f"| {r.name} | {r.seq_len} | {r.analytical_cost_512:,}"
            f" | {r.proxy_constraint_estimate:,} | {logrows_str}"
            f" | {ezkl_rows} | {cal_str} | {speedup_p} | {speedup_e} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-proof",
        action="store_true",
        help="Run the full EZKL prove pipeline (30-180 min per config).",
    )
    args = parser.parse_args()
    main(run_full_proof=args.full_proof)

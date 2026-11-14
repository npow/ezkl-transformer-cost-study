# What Costs Rows and Proving Time in Small EZKL Transformer Circuits, and What It Costs in Perplexity

Paper: `paper/main.tex` (publish.fun package in `publishfun/`, built by `paper/make_publishfun.py`).

## What this project shows (EZKL 23.0.5, small causal transformers)

1. **Cost model vs. measurement (108 compiled configs + 16 out-of-range).** An analytical constraint model ranks
   compiled rows well (Spearman 0.93) but no better than the product d*L*s (0.96), is uncalibrated
   (measured/modelled 0.12-0.53) and extrapolates worse to held-out widths/depths/sequence lengths
   (24-38% error) than a power law (6-18%) or a calibrated additive model (8-11%). At width 256-512
   (out of range) its fitted prediction errs by 122% against 10% for the product d*L*s.
2. **Proofs (24 configs verified; 20 run twice).** Rows predict mean proving time across shapes (Spearman 0.89)
   but proving time is a step function of the power-of-two domain (a boundary crossing doubled it).
   Run-to-run wall-clock differs by 0.82-1.91x on a shared machine, so proving times do not separate
   ReLU, poly2 and GELU. Keys 8.5-40.5 GB, peak RAM 21-103 GB for these toy models.
3. **What costs rows.** At short sequences the FFN nonlinearity dominates (width 128: ReLU +7%, poly2
   +20%, GELU +42%); softmax grows with sequence length (width 64: 4% at s=8 to 24% at s=64).
4. **Mask-fill artifact.** The causal-mask fill value moved EZKL's logrows from 19 to 26 at constant
   rows (`experiments/run_maskfill_sweep.py`).
5. **Quality (WikiText-2, 3 seeds, validation-selected checkpoints).** ReLU, poly2 and GELU are within
   1% (test PPL 166-167 at 384d/6L; ReLU nominally lowest, not significant at n=3); larger models help; nothing trained
   from scratch approaches pretrained GPT-2 (43.5 test).
6. **Correctness note.** The attention originally had no causal mask (leaked next-token targets). It is
   fixed and guarded by `tests/test_transformer_space.py::test_model_is_causal*`.

Limits: tiny circuits (width <= 128, <= 4 layers, seq <= 32; decomposition to 64), proofs only to domain
2^21, single prover/version, WikiText-2 only, quantized accuracy not measured. See the paper's
limitations section.

## Layout

```
proxy/        analytical cost model inputs (op_costs.py vendored) and accuracy proxies
search/       transformer space (causal), WikiText-2 trainer (best-checkpoint tracking), NAS loop (not evaluated)
experiments/  run_ezkl_validation.py   108-config grid, softmax x FFN decomposition   -> artifacts/ezkl_validation*/
              run_ezkl_proving.py      24 real proofs (setup/prove/verify, memory)      -> artifacts/ezkl_proving*/
              run_maskfill_sweep.py    logrows vs causal-mask fill                      -> artifacts/ezkl_validation/
              run_causal_lm_study.py   leak-free LM study, checkpoint selection          -> artifacts/causal_study/
              analyze_validation.py    aggregates everything                             -> artifacts/summary.json
paper/        main.tex, make_publishfun.py
artifacts/    raw per-run JSON records, model_components.json, summary.json
```

## Reproducing

```bash
uv sync --extra dev
# Quality study (GPU recommended)
for g in reference ablation shape gpt2_rotary distill decomp; do
  uv run python experiments/run_causal_lm_study.py --group $g; done

# EZKL experiments need `pip install ezkl==23.0.5` on top of the project dependencies
python experiments/run_ezkl_validation.py                                   # calibrated-scale grid + decomposition
python experiments/run_ezkl_validation.py --scale 11 --out ezkl_validation_scale11 --what grid
python experiments/run_ezkl_validation.py --scale 11 --out ezkl_validation_scale11 --what decomp_T
python experiments/run_ezkl_validation.py --scale 11 --out ezkl_validation_large --what large   # width 256/512
python experiments/run_maskfill_sweep.py
python experiments/run_ezkl_proving.py     # needs ~100 GB RAM and ~50 GB free disk per proof at k=21
python experiments/run_ezkl_proving.py --small-only --out ezkl_proving_run2    # repeat run (domain <= 2^20)

python experiments/analyze_validation.py
uv run --with pypandoc_binary python paper/make_publishfun.py               # publish.fun package
```

Tests: `uv run --extra dev pytest -q`. A few tests depend on the environment (ONNX/`ml_dtypes`
versions, a pandas/numpy build check) or on the optional third-party package `zkml_bench` (not needed
for any result in the paper) and fail or are skipped independently of this project's code.

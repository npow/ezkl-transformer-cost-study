# What Costs Rows in Small EZKL Transformer Circuits, and What It Costs in Quantized Quality

Paper: `paper/main.tex`. The `publishfun/` directory contains the current manuscript package.

## Findings

This study measures compiled EZKL 23.0.5 circuit size, proof-domain behavior, and language-model quality for small causal transformers.

1. **Author-defined operation-count stand-in.** The paper makes one explicit per-operator formalization of a broad cost heuristic. It is not a cost model proposed by EZKL or by prior work. Across 108 compiled configurations, it ranks row counts (Spearman 0.93) no better than the dimension product (0.96), is uncalibrated, and transfers worse than shape baselines on held-out axes. On 16 larger configurations, its fitted prediction errs by 122% versus 10% for the dimension product.
2. **Circuit layout.** The revision reports EZKL's `num_inner_cols` and dynamic column size and evaluates `num_rows * num_inner_cols`. Row counts alone are not used to infer how MACs are packed.
3. **What costs rows.** At short sequences the FFN nonlinearity dominates the row overhead for GELU and poly2 (width 128: ReLU +7%, poly2 +20%, GELU +42%); softmax overhead grows with sequence length (width 64: 4% at `s=8` to 24% at `s=64`).
4. **Mask fill.** The causal-mask fill value moved EZKL's `logrows` from 19 to 26 at constant rows (`experiments/run_maskfill_sweep.py`).
5. **Quality.** The full-vocabulary WikiText-2 study uses GPT-2 BPE and non-overlapping 256-token blocks. Those scores are not directly comparable with the 29.41 WikiText-2 result reported by Radford et al. under a different context/evaluation protocol. A separate 512-symbol model is trained, compiled, and evaluated on identical targets: test PPL is 17.356 full precision and 17.376 after EZKL fixed-point execution (960 targets). The auditable weights, token map, ONNX, compiled model, settings, hashes, and per-split metrics are in `artifacts/revision_matched/`.

The archived proof timings were collected on a shared machine. The paper does not make activation-level proving-time claims. Limits include the small compiled shapes and one matched quantized-quality configuration; see the paper's limitations section.

## Layout

```
proxy/        author-defined operation-count stand-in inputs and quality proxies
search/       transformer configurations and WikiText-2 trainer
experiments/  run_ezkl_validation.py             compiled grid and decomposition
              analyze_advice_columns.py           rows, inner columns and row-column summary
              run_matched_quantized_quality.py    trained/compiled matched model
              run_maskfill_sweep.py               proving-domain versus causal-mask fill
              run_causal_lm_study.py               full-vocabulary quality study
              analyze_validation.py                grid statistics
paper/        manuscript and generated PDF
publishfun/   submission body, metadata and keywords
artifacts/    raw per-run JSON records and revision measurements
```

## Reproducing

```bash
uv sync --extra ezkl
uv run python experiments/run_ezkl_validation.py --what grid --out revision_columns
uv run python experiments/analyze_advice_columns.py
uv run python experiments/run_matched_quantized_quality.py
uv run python experiments/run_maskfill_sweep.py
uv run python experiments/run_causal_lm_study.py --group ablation
```

The revised code and artifacts are published in release tag `v2`. The old proof records are retained as supporting artifacts. Reproducing them requires substantial memory and disk space; their proving times are not used for claims in the revised paper.

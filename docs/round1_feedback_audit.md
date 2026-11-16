# Round 1 feedback audit

Audited against the authenticated publish.fun Round 1 reports on 2026-11-16. This is an author-side completeness audit, not editorial approval.

| Editor requirement | Evidence in revision | Assessment |
|---|---|---|
| Differentiate overlapping prior work | Related Work cites Zenodo 21266805, ePrint 2026/1390, nanoGPT layout account, and practitioner argument | Addressed by citations and incremental framing |
| Author-defined cost model | Abstract, introduction, methods call it an author-defined stand-in | Addressed; no universal cost-model refutation claimed |
| Columns / MAC-packing interpretation | 108 records in revision_columns; every num_inner_cols=2; row-column product correlations reported; causal packing claim withdrawn | Addressed for requested setting; not a total physical-column inventory |
| GPT-2 protocol discrepancy | Raw dataset, GPT-2 BPE, non-overlapping 256-token protocol stated; direct comparison removed | Editor's explanation alternative used; standard rerun remains unperformed |
| Same trained/compiled weights and quantized quality | revision_matched and revision_quantization: 3 activations × scales 11–13, 960 targets in each split | Addressed for small 512-symbol proxy only |
| Reliable timings or remove inference | Timing comparisons and inferences removed | Editor's removal alternative used; idle-host timing question remains unanswered |

## Individual reviewer questions

LayerNorm removal/replacement and predictor orthogonalization were not run. Existing fit sensitivity demonstrates non-identification; the manuscript withdraws the mechanistic coefficient interpretation. This answers the concern about the inference, but supplies no new causal operator-cost experiment.

Per-variant learning-rate tuning was not run. Activation results are descriptive under a shared protocol, with n=3 and sign-test p=0.125. The discussion/conclusion must not claim quality equivalence or a tuned activation ranking.

The attention comparison changes both softmax and masking implementation. The result now states that confound locally, and the unmeasured longer-sequence proving-cost extrapolation is removed. An isolated softmax ablation remains absent.

Quantized quality is measured for GELU, ReLU and poly2 at scales 11–13, with fixed weights per activation and frozen held-out samples. It does not establish full-vocabulary quantized parity. Recompiled circuit bytes have different hashes; evaluation and released hashes are retained separately. A saved GELU batch and two ReLU batches reproduced their NLLs with released circuits.

## Validation and remaining risk

The artifact audit checked all nine released circuit hashes and both 960-target denominators per record, and confirmed 108 column-grid records. The PDF builds and the Markdown package is regenerated. The response letter now answers each reviewer's numbered question explicitly.

The editor may still request additional experiments despite the permitted alternatives. The unresolved causal LayerNorm ablation, mask confound, lack of tuning and small quantized sample remain scientific limitations. Resubmission is not acceptance.

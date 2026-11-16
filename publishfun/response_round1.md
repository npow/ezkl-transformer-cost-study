# Response to the editor and reviewers (Round 1)

Dear Editor and Reviewers,

Thank you for the detailed reports. We substantially revised the manuscript and added new EZKL measurements. The changes are summarized below.

1. **Prior work and contribution.** We added a related-work section citing the EZKL 23.0.5 ReLU-MLP measurements, the ReLU-count cost study, the EZKL nanoGPT column-layout explanation, the Trust the Process study, and practitioner guidance. The revised text distinguishes those results from our controlled transformer grid and does not claim a universal cost law.

2. **Cost heuristic.** We now call this an author-defined operation-count stand-in. The paper gives its formula and per-operator weights and states that it is not an EZKL or prior-work model.

3. **Rows and columns.** We report EZKL `num_inner_cols` and `total_dynamic_col_size` for the full compiled grid, calculate `num_rows × num_inner_cols`, and compare its rank correlations with the operation-count stand-in and dimension-product baseline. The unsupported interpretation that a zero fitted matrix-product coefficient means MACs are packed into one row has been removed.

4. **GPT-2 perplexity protocol.** We explain that the former 44.4/43.5 zero-shot scores used GPT-2 BPE and non-overlapping 256-token blocks on the raw WikiText-2 data. We contrast that protocol with the 29.41 result cited by Radford et al. (attributed there to Gong et al.), explain why context-window and tokenization differences prevent a direct comparison, and remove the pretrained row from the quality table.

5. **Matched compiled and trained models.** We trained a width-128, four-head, two-layer model (sequence length 16, 512 output symbols) for ReLU, GELU and polynomial-square activations, each from a separately trained 500-step AdamW run. We retained the 511 most frequent GPT-2 BPE tokens in WikiText-2 training and mapped every other token to UNK. For each activation, the same trained weights were evaluated in full precision and through compiled EZKL fixed-point witnesses at scales 11, 12 and 13 on identical held-out contexts and targets. Each validation and test split has 64 blocks and 960 targets. Across the nine activation-scale pairs, the test PPL change from full precision ranges from -0.07% to +0.11%; we see no monotonic scale trend and do not infer general losslessness. The ReLU scale-11 result is 16.636/16.635 validation PPL and 17.356/17.376 test PPL (full precision/quantized); its compiled circuit has 2,319,073 rows and two inner columns. We report sample-index and token-id hashes, model/ONNX/circuit hashes, fixed-point settings, and per-split NLLs in `artifacts/revision_quantization/*_metrics.json`, with the original ReLU scale-11 bundle in `artifacts/revision_matched/`. These are small-model feasibility measurements, not a general quality-cost frontier.

6. **Proving-time claims.** We removed the activation-level proving-time section and all variant timing comparisons. The available proof records came from a shared machine; they are retained as supporting artifacts but are not used for timing claims.

7. **Other reviewer questions and scope.** We do not claim that the additive NNLS fit identifies matrix-product or LayerNorm costs: their predictors are highly collinear (Spearman 0.97), and omitting the matrix-product term leaves the fit's $R^2$ unchanged. We did not run a LayerNorm-removal ablation. We also did not run per-variant learning-rate sweeps; the three-seed activation ordering remains explicitly suggestive (sign-test $p=0.125$), and the text notes that shared learning rates may under-tune the linear variants. The attention decomposition's mask-implementation difference is stated as a limitation. We did not re-evaluate GPT-2 under the cited 29.41 protocol: the manuscript now describes the raw WikiText-2/GPT-2-BPE/non-overlapping-256-token protocol, explains the context/tokenization mismatch, and removes the zero-shot row and direct comparison. We make no GPT-2-scale circuit claim.

8. **Quantization scale and activation sensitivity.** In response to the request for quantized GELU, ReLU and polynomial results at scales 11--13, we added matched evaluations at all nine activation-scale pairs. Every pair uses the same activation-specific trained weights and 960 test targets as its full-precision reference. Test PPL changes were: ReLU +0.112%, +0.046%, and -0.046%; GELU +0.002%, +0.099%, and -0.007%; poly2 -0.065%, +0.050%, and -0.012% at scales 11, 12, and 13, respectively. The expanded results and sample/artifact audits are in `artifacts/revision_quantization/`; records preserve evaluation-time and released-circuit hashes because recompiling the same settings changes the byte hash. A released GELU scale-11 circuit exactly reproduced a saved batch of eight NLL values. The paper states that these are small-model sensitivity measurements and do not establish quality for larger or full-vocabulary models.

The revised manuscript, scripts, and raw records are available in the linked repository at release `v3`.

Sincerely,

Nissan Pow

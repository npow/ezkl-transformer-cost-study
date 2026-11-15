# Response to the editor and reviewers (Round 1)

Dear Editor and Reviewers,

Thank you for the detailed reports. We substantially revised the manuscript and added new EZKL measurements. The changes are summarized below.

1. **Prior work and contribution.** We added a related-work section citing the EZKL 23.0.5 ReLU-MLP measurements, the ReLU-count cost study, the EZKL nanoGPT column-layout explanation, the Trust the Process study, and practitioner guidance. The revised text distinguishes those results from our controlled transformer grid and does not claim a universal cost law.

2. **Cost heuristic.** We now call this an author-defined operation-count stand-in. The paper gives its formula and per-operator weights and states that it is not an EZKL or prior-work model.

3. **Rows and columns.** We report EZKL `num_inner_cols` and `total_dynamic_col_size` for the full compiled grid, calculate `num_rows × num_inner_cols`, and compare its rank correlations with the operation-count stand-in and dimension-product baseline. The unsupported interpretation that a zero fitted matrix-product coefficient means MACs are packed into one row has been removed.

4. **GPT-2 perplexity protocol.** We explain that the former 44.4/43.5 zero-shot scores used GPT-2 BPE and non-overlapping 256-token blocks on the raw WikiText-2 data. We contrast that protocol with the 29.41 result cited by Radford et al. (attributed there to Gong et al.), explain why context-window and tokenization differences prevent a direct comparison, and remove the pretrained row from the quality table.

5. **Matched compiled and trained model.** We trained a width-128, four-head, two-layer ReLU transformer (sequence length 16, 512 output symbols) from scratch for 500 AdamW steps. We retained the 511 most frequent GPT-2 BPE tokens in WikiText-2 training and mapped every other token to UNK. We compiled that trained state once with EZKL and evaluated full-precision and fixed-point witness logits on identical held-out contexts and targets. Validation PPL is 16.636 full precision and 16.635 quantized; test PPL is 17.356 and 17.376, respectively. Each split has 64 blocks and 960 targets. We report sample-index and token-id hashes, artifact hashes, compile settings (2,319,073 rows, two inner columns, fixed-point scale 11), and per-split NLL in `artifacts/revision_matched/metrics.json`. The test difference is 0.11% on this small sample; this is one feasibility point, not a general quality-cost frontier.

6. **Proving-time claims.** We removed the activation-level proving-time section and all variant timing comparisons. The available proof records came from a shared machine; they are retained as supporting artifacts but are not used for timing claims.

The revised manuscript, scripts, and raw records are available in the linked repository at release `v2`.

Sincerely,

Nissan Pow

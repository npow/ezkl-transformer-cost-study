# 1. Introduction

Verifiable inference—proving that a model ran correctly on private inputs—is costly for transformer-scale models (Sun et al. 2024; Qu et al. 2025; Wang 2026). Prior work and engineering accounts discuss the relative cost of nonlinearities, matrix products, and lookup ranges (Ganescu and Passerat-Palmbach 2024; Jo 2026; Zkonduit 2023; Pow 2024; Gupta 2026; Abdulrahman 2026). We measure those effects in a controlled EZKL 23.0.5 transformer grid. To make an operation-counting intuition testable, we define one explicit per-operator predictor; it is an author-built stand-in for a broad heuristic, not a published cost model. Our contributions are an EZKL-specific row-and-column characterization against simple shape baselines, a causal-mask-fill measurement of proving-domain size, and a small matched measurement of compiled cost and quantized quality.

1.  **A controlled EZKL circuit-shape study**: 108 configurations, simple shape baselines, a cluster bootstrap, held-out-axis and out-of-range prediction, plus advice-column counts (Section 4).

2.  **Implementation-level measurements**: attention/FFN row decomposition and mask-fill effects on proving domains (Sections 4–5).

3.  **A trained-and-compiled small model**: full-precision and EZKL-quantized quality are measured on the same architecture and weights (Sections 6–7).

#### What this paper does not show.

We do not compare proving times because the available runs were collected on a shared machine. We do not study other proof systems or circuits at GPT-2 scale (Section 9).

# 2. Related Work

#### ZK-ML proof systems.

Sun et al. (2024) introduce tlookup and zkAttn arguments for transformer nonlinearities; Qu et al. (2025) reduce the cost of non-linear layers with constraint fusion; Wang (2026) prove each layer separately and chain the layer proofs with commitments; Maheri et al. (2025) shrink circuits by sparsification and by narrowing activation ranges, which shrinks the lookup tables; Kang et al. (2022) proved inference of an ImageNet-scale model with ZK-SNARKs. These works fix the architecture and improve the prover or the model. We hold the prover (EZKL (Zkonduit 2024)) fixed and measure how architecture choices change compiled circuit size.

#### Measurements and cost guidance for EZKL.

Ganescu and Passerat-Palmbach (2024) study parameter/constraint scaling and proof runtime for a nanoGPT implementation. Jo (Jo 2026) measures EZKL 23.0.5 row scaling for ReLU MLP depth and analyzes lookup cost; a Zenodo artifact reports ReLU-count correlations with EZKL/Halo2 proof cost (Abdulrahman 2026). The EZKL nanoGPT account explains that dot-product constraints wrap into additional advice columns when the row domain fills (Zkonduit 2023). Practitioner accounts argue that matrix products are relatively cheap and nonlinearities or lookups dominate in some proof systems (Pow 2024; Gupta 2026). Pow’s earlier EZKL study also compares accuracy with proving cost (Pow 2024). These works establish the relevant hypotheses and implementation context. We add a controlled transformer activation/attention grid, report both rows and inner columns, test the row–column product against a shape-only baseline, and measure how a fixed-point compiled model’s logits and perplexity compare with its full-precision counterpart. Our results are limited to EZKL 23.0.5 and the tested small shapes; they do not establish a universal zkML cost law.

#### Hardware-aware NAS and cost models.

Cai et al. (2019), Tan et al. (2019) and Wu et al. (2019) established latency-aware search with hardware cost proxies, validated against measured latency; Javaheripi et al. (2022) observe that decoder parameter count correlates with perplexity and use it as a training-free proxy for language models. Bian et al. (2020) search for architectures under homomorphic-encryption and garbled-circuit secure inference. Our analogous measurement is compiled circuit size (Section 4); proving-time comparisons are outside this paper’s claims.

#### Zero-cost proxies.

SNIP (Lee et al. 2019) and NEAR (Husistein et al. 2025) are training-free proxies for network quality; Serianni and Kalita (2023) study training-free metrics for RNNs and transformers. We do not evaluate proxies.

# 3. Setup

#### Models.

Decoder-only transformers with pre-LayerNorm, tied embeddings, causal attention, and an FFN activation from {GELU, ReLU, poly2 ($`x^2`$), identity}. Attention is either softmax or unnormalized (“linear”) causal attention. The variant called “linear” throughout removes *both* the softmax and the FFN nonlinearity (identity FFN); Sections 4 and 6 also report the two factors separately. Positions are learned absolute embeddings unless rotary is stated; the EZKL grid uses learned absolute positions.

#### Author-defined operation-count stand-in.

The model (`experiments/operator_attribution.py`; per-operator costs vendored in `proxy/op_costs.py`) sums, per operator category, an operation count times a per-operation constraint cost. Matrix products cost 1 constraint per multiply-accumulate, counted as 2 MACs per product: per layer, QKV and output projections $`8sd^2`$, attention scores and values $`4s^2d`$, and FFN up and down projections $`16\,sd^2`$ at an expansion of 4. Softmax costs 32 per score element, the FFN activation costs 2 (ReLU), 24 (GELU), 2 (poly2) or 0 (identity) per hidden element, LayerNorm 24 per element, and the token embedding (plus position embedding when rotary is off) 2 per table entry. It has no term for the output (LM-head) projection. Here $`s`$ is sequence length and $`d`$ the width. We call its output “modelled constraints.” This explicit formalization is our own stand-in for a broad operation-counting intuition; it is not a cost model proposed by EZKL or by prior work. Results comparing it with EZKL test this formalization only.

#### Measurement.

Each model is exported to ONNX (opset 18) and passed to EZKL 23.0.5 (`gen_settings`, then `calibrate_settings` with `target=resources`). Calibration uses four fixed random token sequences and chooses the fixed-point scale itself (`param_scale` is 13 in 86 records, 12 in 17 and 11 in 5; the linear variant is always 13), which confounds row counts with the scale, so we replicate the grid with the scale pinned (Section 4, “Robustness to the fixed-point scale”). We record `num_rows`, the EZKL `num_inner_cols` setting and dynamic column size. The revision grid reports these directly (Section 4); row count alone does not describe the advice-column layout. The grid measures compiled circuit size. Weights are random initialization. The causal mask uses a finite fill value of $`-100`$ (Section 5). `total_assignments` equals $`2\times`$ `num_rows` (exactly in 72 of 108 records, within $`10^{-4}`$ in the rest), so we report rows only.

# 4. Measured EZKL Circuit Size

The grid is the full product of width $`\in\{64,96,128\}`$, layers $`\in\{1,2,4\}`$, sequence length $`\in\{8,16,32\}`$ and four activation variants (108 configurations; vocabulary 512; 4 heads). All 108 compiled. The grid has only 27 distinct shapes, and the four variants at a shape are near-duplicates, so all intervals below are cluster bootstraps over shapes.

#### The operation-count stand-in does not beat a shape baseline.

The Spearman correlation between modelled constraints and measured rows is $`0.927`$ (95% CI $`[0.83,0.97]`$). The product $`d_{\mathrm{model}}n_{\mathrm{layers}} s`$ achieves $`0.964`$ ($`[0.91,0.98]`$) and $`n_{\mathrm{layers}} s`$ alone $`0.935`$; the paired difference (stand-in minus product) has 95% CI $`[-0.091,+0.005]`$. Within each variant the stand-in scores $`0.92`$–$`0.94`$ and the product $`0.97`$–$`0.98`$. On this grid, a monotone shape proxy ranks configurations at least as well as our stand-in.

#### The stand-in is uncalibrated and extrapolates poorly.

The ratio of measured rows to modelled constraints ranges from $`0.12`$ to $`0.53`$ (median $`0.22`$), so rows are *smaller* than the modelled count and no single constant converts one to the other. A log-log fit on the model output alone has slope $`0.77`$ and $`R^2{=}0.85`$; a fit on the three dimensions gives $`\mathrm{rows}\propto d_{\mathrm{model}}^{0.71}n_{\mathrm{layers}}^{0.79}s^{1.10}`$ with $`R^2{=}0.97`$. The model also misorders $`27`$ of $`162`$ within-shape variant pairs.

To test transfer, Table 1 holds out one value of one axis (each of three widths, three depths or three sequence lengths in turn; the smallest and largest values are extrapolations, the middle one an interpolation), fits on the remaining configurations, and predicts the held-out rows (mean absolute percentage error, averaged over held-out values). Besides the model and the product $`d_{\mathrm{model}}n_{\mathrm{layers}} s`$, it includes a three-exponent power law and a *calibrated additive model*: non-negative least squares of rows on the model’s own operator categories (matrix products, softmax, activation, LayerNorm, embedding), optionally with an output-projection term the model lacks. The operation-count stand-in with a fitted log-log line is worst on every axis; the additive fit is best on width, within $`1`$ point of the best on depth and $`2`$ points behind the power law on sequence length.

| Predictor | held-out $`d_{\mathrm{model}}`$ | held-out $`n_{\mathrm{layers}}`$ | held-out $`s`$ |
|:---|:--:|:--:|:--:|
| Operation-count stand-in (log-log fit) | 38.1 | 24.0 | 37.0 |
| Product $`d_{\mathrm{model}}n_{\mathrm{layers}} s`$ (log-log fit) | 14.7 | 17.2 | 21.4 |
| Power law $`d_{\mathrm{model}}^{x}n_{\mathrm{layers}}^{y}s^{z}`$ | 17.6 | 7.2 | 6.0 |
| Additive NNLS, model’s categories | 11.2 | 7.5 | 8.0 |
| Additive NNLS $`+`$ LM-head term | 11.8 | 7.6 | 8.1 |

**Table 1.** Leave-one-axis-out error of measured rows (mean absolute percentage error, 108 configurations; lower is better). Log-log fits include per-variant offsets; the additive fits have 5 or 6 non-negative weights.

The fitted additive weights (rows per modelled constraint) are $`0`$ for the matrix product and embedding terms, $`2.7`$ for softmax, $`1.4`$ for the activation and $`8.0`$ for LayerNorm; they exceed one because the matrix-product term, which dominates the modelled total, receives no weight. Under a cluster bootstrap over shapes ($`1000`$ resamples) the matrix-product weight is exactly zero in $`92\%`$ of fits (95% interval $`[0,0.011]`$). This does not show that matrix products are free: the matrix-product and LayerNorm terms are strongly collinear (Spearman $`0.97`$), the fit without the matrix-product term explains the same variance as the full fit ($`R^2{=}0.989`$ in both), and the matrix-product term alone gives $`R^2{=}0.79`$. The data cannot separate the two terms. We therefore make no mechanistic inference from the zero matrix-product coefficient; row counts alone do not establish how EZKL lays out multiply-accumulates across its advice columns. The next comparison reports both dimensions.

#### Rows and advice columns.

Across all 108 calibrated grid records, EZKL reports `num_inner_cols`$`=2`$. The recorded dynamic-column size is determined by width: 32,768 at width 64, 49,152 at width 96, and 65,536 at width 128 (36 configurations each). Thus $`\verb|num_rows|\times
\verb|num_inner_cols|`$ is exactly twice the row count and preserves every rank and Spearman correlation. Its correlation with the operation-count stand-in is $`0.927`$ and with $`d_{\mathrm{model}}n_{\mathrm{layers}} s`$ is $`0.964`$, identical to the row-only correlations. This observed constant column count does not reveal how individual multiply-accumulates are assigned within rows or columns.

| Width | Configurations | Dynamic-column size | Rows $`\times`$ inner columns |
|------:|:--------------:|:-------------------:|:-----------------------------:|
|    64 |       36       |       32,768        |      695,258–12,869,102       |
|    96 |       36       |       49,152        |      794,412–15,971,904       |
|   128 |       36       |       65,536        |     1,096,154–21,646,966      |

**Table 2.** Advice-column measurements in the 108-circuit revision grid. All records at a given width have the listed dynamic-column size and two inner columns. The product is reported as a range of row counts doubled.

#### What costs rows: the FFN nonlinearity at short sequences.

Table 3 crosses attention (softmax or linear) with the FFN activation at $`n_{\mathrm{layers}}{=}2`$, $`s{=}16`$, and reports rows relative to linear attention with an identity FFN.

| FFN activation | Linear attention | Softmax attention |
|:---------------|:----------------:|:-----------------:|
| identity       |       1.00       | 1.02 (1.02–1.08)  |
| ReLU           | 1.07 (1.06–1.07) | 1.09 (1.09–1.14)  |
| poly2          | 1.20 (1.17–1.22) | 1.23 (1.23–1.25)  |
| GELU           | 1.42 (1.36–1.44) | 1.44 (1.44–1.48)  |

**Table 3.** Measured EZKL rows relative to linear attention with identity FFN, same shape ($`n_{\mathrm{layers}}{=}2`$, $`s{=}16`$). Entries are the value at width 128 with the range over widths 64–128 in parentheses.

Softmax adds $`2`$–$`8\%`$ (falling with width: $`1.079`$, $`1.031`$, $`1.022`$ at widths $`64`$, $`96`$, $`128`$); the FFN activation adds $`7\%`$ (ReLU), $`20\%`$ (poly2) or $`42\%`$ (GELU). Over the full grid, relative to the identity-FFN linear variant, GELU uses $`1.45\times`$ the rows (by width $`1.43`$, $`1.46`$, $`1.43`$), poly2 $`1.25\times`$ ($`1.26`$, $`1.24`$, $`1.22`$) and ReLU $`1.10\times`$ ($`1.14`$, $`1.10`$, $`1.09`$); GELU’s overhead is flat in width, while poly2’s and ReLU’s shrink slightly (by 3 and 5 points between widths 64 and 128). The operation-count stand-in puts the linear variant only $`3`$–$`8.5\%`$ below GELU at these shapes, against $`31\%`$ measured.

#### Softmax grows with sequence length.

Table 4 repeats the decomposition at sequence lengths $`8`$–$`64`$ for widths 64 and 128 (two layers, scale pinned at 11). The softmax overhead grows quickly with sequence length and falls with width: at width 64 it rises from $`4\%`$ ($`s{=}8`$) to $`24\%`$ ($`s{=}64`$), where it equals the GELU overhead ($`23\%`$ over softmax attention with identity FFN); at width 128 it rises from $`1.3\%`$ to $`6.3\%`$. The FFN nonlinearity therefore dominates only at short sequences and wide models; at width 64 and $`s{=}64`$, softmax is as costly as GELU, and because softmax is quadratic in $`s`$ it should dominate at the longer sequences used in practice, which we did not measure. Across the whole decomposition, ReLU uses $`16`$–$`25\%`$ fewer rows than GELU (and $`16`$–$`29\%`$ across the 27 grid shapes, median $`23\%`$).

| Width | FFN activation | $`s{=}8`$ | $`s{=}16`$ | $`s{=}32`$ | $`s{=}64`$ |
|:------|:---------------|:---------:|:----------:|:----------:|:----------:|
| 64    | identity       |   1.04    |    1.08    |    1.14    |    1.24    |
| 64    | ReLU           |   1.11    |    1.14    |    1.20    |    1.29    |
| 64    | GELU           |   1.42    |    1.44    |    1.47    |    1.52    |
| 128   | identity       |   1.01    |    1.02    |    1.04    |    1.06    |
| 128   | ReLU           |   1.09    |    1.09    |    1.10    |    1.12    |
| 128   | GELU           |   1.45    |    1.44    |    1.43    |    1.41    |

**Table 4.** Measured rows relative to linear attention with identity FFN at the same shape, versus sequence length ($`n_{\mathrm{layers}}{=}2`$, scale pinned at 11). All entries use softmax attention.

#### Robustness to the fixed-point scale.

Calibration chose the scale in the main grid (11–13). Re-running all 108 configurations with the scale pinned at 11 gives identical `logrows` in every configuration and row counts that are identical in 72 of 108 and differ by a few rows (below $`3\times10^{-5}`$ relative) in the rest, so every statistic above is unchanged at the reported precision. (Weights are random and unseeded, which accounts for the few-row differences between repeated compilations.)

#### Out-of-range check.

To test extrapolation beyond the fitted range, we compiled 16 further configurations (widths 256 and 512, 1–2 layers, sequence lengths 16 and 64; scale pinned at 11; up to $`2.2\times10^{7}`$ rows at domain $`2^{25}`$) and predicted their rows with every predictor fitted on the 108-configuration grid (Table 5). The operation-count stand-in’s fitted line overpredicts by a factor of about $`2`$ (error $`122\%`$), and its raw measured/modelled ratio falls to $`0.03`$–$`0.10`$ (median $`0.057`$), below the $`0.12`$–$`0.53`$ seen on the grid: the gap widens with size. The product $`d_{\mathrm{model}}n_{\mathrm{layers}} s`$ and the power law predict within about $`10\%`$; the additive fit, the best predictor in the leave-one-out test, is worse out of range ($`16\%`$). Activation overheads persist at the larger widths: relative to the linear variant, GELU uses $`1.45`$–$`1.54\times`$ the rows, poly2 $`1.23`$–$`1.31\times`$ and ReLU $`1.08`$–$`1.17\times`$ (against $`1.43`$–$`1.46`$, $`1.22`$–$`1.26`$ and $`1.09`$–$`1.14`$ at widths 64–128).

| Predictor | Error (%) | Median pred./measured | Spearman |
|:---|:--:|:--:|:--:|
| Operation-count stand-in (log-log fit) | 122.2 | 2.13 | 0.88 |
| Product $`d_{\mathrm{model}}n_{\mathrm{layers}} s`$ (log-log fit) | 10.0 | 1.00 | 0.99 |
| Power law $`d_{\mathrm{model}}^{x}n_{\mathrm{layers}}^{y}s^{z}`$ | 10.7 | 0.90 | 0.99 |
| Additive NNLS | 16.1 | 1.15 | 0.96 |

**Table 5.** Prediction of rows for 16 out-of-range configurations (width 256–512, sequence length 16–64) by predictors fitted on the 108-configuration grid (width 64–128).

#### Scope.

The grid has at most four layers, width 128 and sequence length 32, and the out-of-range check reaches width 512 and sequence length 64 with 1–2 layers. GPT-2-scale models (width 768, 12 layers, sequence length 512 or more) lie outside our measured range; none of the fitted laws should be extrapolated beyond the tested shapes.

# 5. Proving-Domain Size Depends on the Mask Fill Value

EZKL’s `logrows` sets the proving domain and hence key size and memory. A causal softmax needs a masked-score fill value; $`-\infty`$ cannot be quantized (`gen_settings` fails), so a large negative constant is used. Table 6 shows that this choice alone moves `logrows` while the row count is unchanged: EZKL sizes its softmax lookup range from the most negative input, so a fill of $`-10^4`$ gives a table spanning about $`1.6\times10^8`$ entries.

| Shape    | Variant           | $`-10^4`$ | $`-10^3`$ | $`-100`$ | $`-30`$ | $`-10`$ |
|:---------|:------------------|:---------:|:---------:|:--------:|:-------:|:-------:|
| 64/1/8   | GELU, ReLU, poly2 |    26     |    22     |    19    |   19    |   19    |
| 64/1/8   | linear            |    19     |    19     |    19    |   19    |   19    |
| 128/2/16 | GELU, poly2       |    26     |    22     |    22    |   22    |   22    |
| 128/2/16 | ReLU              |    26     |    22     |    21    |   21    |   21    |
| 128/2/16 | linear            |    21     |    21     |    21    |   21    |   21    |

**Table 6.** `logrows` versus causal-mask fill value at two shapes. Row counts are constant across fills (to within $`2\times10^{-5}`$, from unseeded random weights): $`458{,}765`$ (GELU), $`375{,}821`$ (ReLU), $`407{,}565`$ (poly2) and $`347{,}629`$ (linear) at width 64, 1 layer, $`s{=}8`$; $`2.72`$M, $`2.05`$M, $`2.31`$M and $`1.88`$M at width 128, 2 layers, $`s{=}16`$. The linear variant has no softmax, so no fill value applies and its single measurement is repeated. The calibration-chosen scale varied between 11 and 13 across cells; the domain sizes are set by the lookup range and are unaffected.

With a fill of $`-100`$ ($`e^{-100}\approx4\times10^{-44}`$, negligible in fp32), `logrows` ranges $`19`$–$`24`$ for all variants and tracks the row count. Domain-size comparisons between circuits are therefore only meaningful at a fixed, modest fill; we report all grid results at $`-100`$. Row counts were constant across fill values (to within $`2\times10^{-5}`$). This is consistent with the observation of Maheri et al. (2025) that narrowing activation ranges shrinks the lookup tables.

# 6. Quality Study

#### Protocol.

WikiText-2 (Merity et al. 2017) (raw), GPT-2 (Radford et al. 2019) BPE, non-overlapping blocks of 256 tokens (2.39M training tokens). Every model is randomly initialized and trained with AdamW (peak LR $`3\times10^{-4}`$, cosine schedule, 200 warm-up steps, weight decay $`0.1`$, gradient-norm clipping at $`1.0`$, no dropout; the same settings for every model, with no per-variant tuning), batch 16, 5,000 steps ($`\approx8.6`$ epochs), three seeds shared across configurations. Validation perplexity is evaluated every 250 steps; we select the checkpoint with the lowest validation loss (the trainer’s per-batch mean, which differs from the token-weighted value reported below by about $`0.5`$ perplexity) and report validation and test perplexity at that checkpoint (token-weighted, full splits). Validation perplexity at the selected checkpoint is therefore optimistically biased, and test perplexity is the unbiased figure. $`\pm`$ is the sample standard deviation over seeds ($`n{=}3`$). The scratch-trained models use rotary positions except where marked. We do not compare their perplexities with GPT-2 results reported under a different context window or tokenization protocol.

| Model | Params | Val PPL | Test PPL | Final val | Best step |
|:---|---:|---:|---:|---:|---:|
| GPT-2-small architecture, learned positions | 124M | $`179.2\pm2.2`$ | $`187.0\pm2.6`$ | 275.3 | 2250 |
| GPT-2-small architecture, rotary | 124M | $`150.9\pm0.1`$ | $`158.4\pm0.9`$ | 243.4 | 2250 |
| 640d/10L GELU | 81M | $`147.6\pm1.7`$ | $`156.0\pm1.9`$ | 198.5 | 2750 |
| 512d/8L ReLU | 51M | $`151.7\pm0.7`$ | $`157.8\pm1.1`$ | 204.0 | 2250 |
| 384d/10L poly2 | 37M | $`155.0\pm1.2`$ | $`161.0\pm1.2`$ | 223.4 | 2417 |
| 384d/6L GELU | 30M | $`161.2\pm1.0`$ | $`167.0\pm1.4`$ | 200.0 | 2250 |
| 384d/6L ReLU | 30M | $`159.7\pm1.7`$ | $`165.7\pm1.7`$ | 199.6 | 2250 |
| 384d/6L poly2 | 30M | $`160.4\pm1.7`$ | $`166.6\pm1.8`$ | 229.0 | 2250 |
| 384d/6L linear | 30M | $`237.3\pm1.5`$ | $`244.0\pm1.8`$ | 240.7 | 3500 |
| 384d/4L linear | 26M | $`247.0\pm3.1`$ | $`253.6\pm2.8`$ | 251.8 | 3500 |
| 384d/4L linear, distilled from GPT-2 | 26M | $`185.2\pm2.0`$ | $`187.7\pm1.9`$ | 185.3 | 4917 |

**Table 7.** Perplexity at the validation-selected checkpoint (mean $`\pm`$ s.d., 3 seeds). “Final val” is the last-step validation perplexity. The table contains scratch-trained models only; GPT-2 zero-shot values are omitted because Radford et al. report WikiText-2 test PPL $`29.41`$ and attribute that table entry to Gong et al.; context-window and tokenization choices affect perplexity (Radford et al. 2019; Hugging Face 2026). Distillation uses a frozen GPT-2 teacher ($`\alpha{=}0.7`$) and is not a from-scratch result.

#### Findings.

(i) At matched shape (384d/6L) GELU, ReLU and poly2 are within about $`1\%`$ of each other ($`161.2`$, $`159.7`$, $`160.4`$ validation; $`167.0`$, $`165.7`$, $`166.6`$ test). Paired by seed, ReLU is the lowest in all three seeds (below GELU by $`1.5\pm0.7`$ and below poly2 by $`0.7\pm0.5`$ validation perplexity, mean $`\pm`$ s.d.); with three seeds this has a one-sided sign-test $`p{=}0.125`$, so the ordering is suggestive and small, and we do not claim equality or a ranking. (ii) Capacity helps: perplexity falls from $`161`$ (30M) to $`148`$ (81M), and rotary beats learned positions by $`28`$ perplexity points at GPT-2 size, where the learned-position model over-fits fastest. (iii) The “linear” variant (no softmax, identity FFN) is $`47\%`$ worse than GELU at matched shape (it may also be under-tuned, since the learning rate is shared). (iv) Distillation lowers the 26M linear model’s perplexity by $`25\%`$ ($`247.0\to185.2`$; that run was still improving at step 5,000). (v) The pretrained GPT-2 zero-shot model scored $`44.4`$ on validation and $`43.5`$ on test under our GPT-2-BPE, non-overlapping 256-token protocol. Radford et al. report $`29.41`$ for WikiText-2 and attribute that table entry to Gong et al.; our protocol instead uses the raw dataset, GPT-2 BPE and non-overlapping 256-token blocks. Context length and tokenization affect perplexity, and non-overlapping chunks lose context (Radford et al. 2019; Hugging Face 2026). We therefore omit the zero-shot row from the table and make no direct comparison with scratch-trained models. (vi) All softmax models over-fit: the best checkpoint is at steps $`2{,}250`$–$`2{,}750`$ and the final-step perplexity is $`18`$–$`61\%`$ higher, so conclusions drawn from final checkpoints would be wrong (Table 7, “Final val”).

#### Attention and FFN activation separately.

Table 8 crosses the two factors at 384d/6L with the same protocol (the identity-FFN/linear-attention cell is the “linear” variant above).

| FFN activation | Linear attention | Softmax attention |
|:---------------|:----------------:|:-----------------:|
| identity       | $`237.3\pm1.5`$  |  $`192.4\pm0.6`$  |
| ReLU           | $`211.8\pm2.5`$  |  $`159.7\pm1.7`$  |
| GELU           | $`214.7\pm1.5`$  |  $`161.2\pm1.0`$  |

**Table 8.** Validation perplexity at the selected checkpoint, 384d/6L (mean $`\pm`$ s.d., 3 seeds). Test perplexities have the same ordering ($`244.0`$, $`198.1`$, $`219.3`$, $`165.7`$, $`221.7`$, $`167.0`$ in the order of the cells below, row by row, linear then softmax).

Both factors matter. Replacing softmax by linear attention raises perplexity by $`23\%`$ with an identity FFN and by $`33\%`$ with ReLU or GELU; removing the FFN nonlinearity raises it by $`19`$–$`20\%`$ with softmax attention and by $`11`$–$`12\%`$ with linear attention. The $`47\%`$ gap of the “linear” variant to GELU is therefore a combination of both effects, not attributable to either alone.

# 7. Reading Cost and Quality Together

The original quality and circuit studies used different architectures, so their measurements cannot establish a cost–quality frontier. We replace that cross-shape comparison with one small model trained from scratch, compiled with EZKL, and evaluated in full precision and after EZKL fixed-point quantization using the same weights (Table 9). This single configuration is a feasibility measurement; it does not establish a general frontier.

The matched model has width 128, four heads, two ReLU transformer layers, sequence length 16, learned positions, LayerNorm, tied embeddings and a 512-symbol output vocabulary. Its vocabulary is a proxy: the 511 most frequent GPT-2 BPE tokens in the training split receive dedicated ids, and all remaining tokens map to one UNK id. The model is initialized from seed 51,051 and trained for 500 AdamW steps (batch 16, learning rate $`3\times10^{-4}`$, weight decay $`0.1`$, gradient clipping at 1.0). For each held-out split, 64 evenly spaced, non-overlapping blocks contribute the 15 next-token targets at positions 2–16, for 960 targets. The EZKL 23.0.5 compilation has 2,319,073 rows, two inner columns, `total_dynamic_col_size`=65,536, `logrows`=22, and `param_scale`=11. We compile this trained state once and evaluate each sample through its EZKL fixed-point witness; both metrics therefore use identical weights, token ids, contexts and target ids. The sample-index and token-id hashes, token map, state dict, ONNX model, compiled circuit and settings are released with the per-split losses. This deliberately small-vocabulary point isolates quantization effects and is not directly comparable to the full 50,257-token models above. Full-precision/EZKL perplexity is 16.636/16.635 on validation and 17.356/17.376 on test. The test difference is $`0.11\%`$ on this sample; it is not evidence that quantization is lossless in general.

| Split      | Target tokens | Full precision PPL | EZKL PPL |
|:-----------|--------------:|-------------------:|---------:|
| Validation |           960 |             16.636 |   16.635 |
| Test       |           960 |             17.356 |   17.376 |

**Table 9.** Full-precision and EZKL fixed-point perplexity for the same trained model and held-out token targets. Each split contains 64 non-overlapping 16-token blocks; positions 2–16 contribute 960 target tokens.

# 8. Implementation Notes and Sensitivity

#### Causal masking.

Attention is causal: a lower-triangular mask is applied to the scores, and a regression test asserts, for every activation, with rotary on and off and for every attention/FFN combination used here, that changing a future token leaves earlier-position logits unchanged to $`10^{-6}`$. The mask matters for correctness: without it, next-token targets leak and perplexities are not comparable with those of causal language models (we found and fixed this in our own code before running any experiment reported here).

#### Checkpoint selection.

Validation-selected checkpoints and final-step perplexities differ by $`18`$–$`61\%`$ (Table 7), enough to change two conclusions: final-step perplexity would make poly2 appear $`14\%`$ worse than GELU and make capacity appear not to matter. We report the selected checkpoint and keep final-step numbers.

#### Mask fill value.

Section 5 gives the sensitivity of `logrows` to the mask fill; domain sizes are comparable only at a fixed, modest fill.

# 9. Discussion and Limitations

#### Takeaways.

For small causal transformers compiled with EZKL 23.0.5: (1) the author-defined operation-count stand-in is not better than the product of the dimensions for ranking, is uncalibrated, and extrapolates worse than fitted alternatives (at width 256–512 its fitted prediction errs by $`122\%`$ against $`10\%`$ for the dimension product); a calibrated additive model is best on held-out grid values but not out of range (it puts no weight on the matrix-product term, an exploratory result given collinear features); (2) at short sequences the FFN nonlinearity drives the row overhead for GELU and poly2 (ReLU’s is comparable to softmax’s at width 64), but softmax grows with sequence length and at width 64 matches GELU’s overhead by $`s{=}64`$; (3) the matched tiny model provides one feasibility point for full-precision and EZKL-quantized perplexity; proving-time comparisons are excluded because the available runs were made on a shared machine; (4) ReLU, poly2 and GELU are within $`1\%`$ in perplexity (ReLU nominally lowest), and ReLU has the fewest rows among them; (5) domain-size comparisons are only meaningful at a fixed, modest mask fill.

#### Limitations.

(1) *Scale:* the compiled grid has width $`\le128`$, depth $`\le4`$, sequence length $`\le32`$ (decomposition to $`64`$) and vocabulary 512, and the out-of-range check reaches width 512 with 1–2 layers and sequence length $`\le64`$; proofs are limited to width $`\le128`$, 1–2 layers, sequence length $`\le16`$ and domain $`\le2^{21}`$. We do not measure GPT-2-scale circuits, which is outside the tested range, and the fitted laws should not be extrapolated beyond the range tested. (2) *Proving measurements are noisy:* two runs for the 20 configurations with domain at most $`2^{20}`$ and one for the four at $`2^{21}`$, on a machine shared with other jobs (load average $`20`$–$`40`$; run-to-run time ratios $`0.82`$–$`1.91\times`$), with a locally generated, insecure reference string; we could not obtain an idle machine. (3) *Single prover and version:* EZKL 23.0.5, random weights and fixed random calibration tokens; the fixed-point scale was chosen by calibration (replicated with the scale pinned); other backends are not measured. (4) *Joint quality and cost:* only one tiny vocabulary-proxy model is compiled and evaluated in full precision and after EZKL quantization; this is a feasibility point, not a quality–cost frontier, and it is not directly comparable to the 50,257-token quality study. (5) *Quality data:* WikiText-2 is small, models over-fit within about 2,500 steps, a single learning rate and no dropout or regularization sweep are used, and $`n{=}3`$ seeds, so differences of about $`1\%`$ (such as among GELU, ReLU and poly2) are consistent in sign but small, and activation rankings could change under a tuned learning rate or regularization; the “linear” variants may be under-tuned. (6) *Quantized quality:* the matched measurement covers one tiny model and a small held-out sample; it does not establish quantized quality for larger architectures or the full-vocabulary study. (7) *Implementation confound:* linear attention multiplies the scores by the causal mask while softmax attention uses a masked fill followed by softmax, so the measured softmax overhead also includes that implementation difference. (8) *No search procedure* is evaluated; the repository contains search code about which this paper makes no claim.

#### Reproducibility.

Code, the 108 EZKL records at calibrated and pinned scale, the decomposition records (sequence lengths 8–64), the 16 out-of-range records, the mask-fill sweep, supporting proof records (not used for timing claims), and all training-run records (with learning curves) are in <https://github.com/npow/ezkl-transformer-cost-study> (release tag `v2`; scripts `run_ezkl_validation.py`, `run_maskfill_sweep.py`, `run_causal_lm_study.py`, `analyze_validation.py`, `analyze_advice_columns.py`, `run_matched_quantized_quality.py`; raw data under `artifacts/`; the stand-in’s operator table is vendored in `proxy/op_costs.py`). All reported numbers are produced by those scripts.

#### Conclusion.

Within the range we could compile, our author-defined operation-count stand-in adds little over the product of the dimensions and fitted alternatives beat it in held-out and out-of-range row prediction; the FFN nonlinearity is the main activation-dependent row cost at short sequences while softmax grows with sequence length; proving-time comparisons are excluded because available timings were collected on a shared machine; and the full-precision study found ReLU had the fewest rows among nonlinearities without a material perplexity penalty in that protocol. For the one matched small-vocabulary model, EZKL’s fixed-point test perplexity differs by $`0.11\%`$ from full precision on 960 targets. Neither result establishes quantized quality at GPT-2 scale, which remains to be measured with full-size circuits.

# References

- Abdulrahman. ZK Cost-Aware Model Training: An Empirical Analysis of ReLU Activation Count as a Predictor of Zero-Knowledge Proof Cost. Zenodo, 2026. doi:10.5281/zenodo.21266805.

- Ganescu, B.-M. and Passerat-Palmbach, J. Trust the Process: Zero-Knowledge Machine Learning to Enhance Trust in Generative AI Interactions. arXiv:2402.06414, 2024.

- Gupta, G. C. In zkML the matrix multiplies are basically free—it is the ReLU and softmax that cost you. Proof of Tech, 2026. <https://proofoftech.org/blog/zkml-nonlinearities-are-the-tax/>.

- Jo, M. J. A separation principle for lookup-based zkML: Activation-function structure cannot reduce per-lookup proving cost. Cryptology ePrint Archive, Paper 2026/1390, 2026. <https://eprint.iacr.org/2026/1390>.

- Pow, N. zkML: Tradeoffs in accuracy vs. proving cost. NP Labs, 2024. <https://np.engineering/posts/zkml-tradeoffs/>.

- Zkonduit. Honey I SNARKED the GPT: How we got nanoGPT into Halo2–KZG circuits using EZKL. EZKL Blog, 2023. <https://blog.ezkl.xyz/post/nanogpt/>.

- Bian, S., Jiang, W., Lu, Q., Shi, Y., and Sato, T. NASS: Optimizing secure inference via neural architecture search. In *European Conference on Artificial Intelligence (ECAI)*, 2020. arXiv:2001.11854.

- Cai, H., Zhu, L., and Han, S. ProxylessNAS: Direct neural architecture search on target task and hardware. In *International Conference on Learning Representations (ICLR)*, 2019.

- Zkonduit. EZKL: a library and command-line tool for zero-knowledge machine learning inference (version 23.0.5). <https://github.com/zkonduit/ezkl>.

- Hugging Face. Perplexity of fixed-length models. Transformers documentation, accessed 2026. <https://huggingface.co/docs/transformers/perplexity>.

- Husistein, R. T., Reiher, M., and Eckhoff, M. NEAR: A training-free pre-estimator of machine learning model performance. In *International Conference on Learning Representations (ICLR)*, 2025. arXiv:2408.08776.

- Javaheripi, M., de Rosa, G. H., Mukherjee, S., Shah, S., Religa, T. L., Mendes, C. C. T., Bubeck, S., Koushanfar, F., and Dey, D. LiteTransformerSearch: Training-free neural architecture search for efficient language models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv:2203.02094.

- Kang, D., Hashimoto, T., Stoica, I., and Sun, Y. Scaling up trustless DNN inference with zero-knowledge proofs. arXiv preprint arXiv:2210.08674, 2022.

- Lee, N., Ajanthan, T., and Torr, P. H. S. SNIP: Single-shot network pruning based on connection sensitivity. In *International Conference on Learning Representations (ICLR)*, 2019.

- Maheri, M. M., Haddadi, H., and Davidson, A. TeleSparse: Practical privacy-preserving verification of deep neural networks. *Proceedings on Privacy Enhancing Technologies (PoPETs)*, 2025(4):861–880, 2025.

- Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. In *International Conference on Learning Representations (ICLR)*, 2017.

- Qu, W., Sun, Y., et al. zkGPT: An efficient non-interactive zero-knowledge proof framework for LLM inference. In *USENIX Security Symposium*, pages 2045–2063, 2025.

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. OpenAI Blog, 2019.

- Serianni, A. and Kalita, J. Training-free neural architecture search for RNNs and transformers. In *Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 2522–2540, 2023.

- Sun, H., Li, J., and Zhang, H. zkLLM: Zero knowledge proofs for large language models. In *ACM Conference on Computer and Communications Security (CCS)*, 2024. arXiv:2404.16109.

- Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., and Le, Q. V. MnasNet: Platform-aware neural architecture search for mobile. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

- Wang, Z. G. NANOZK: Layerwise zero-knowledge proofs for verifiable large language model inference. arXiv preprint arXiv:2603.18046, 2026.

- Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., Tian, Y., Vajda, P., Jia, Y., and Keutzer, K. FBNet: Hardware-aware efficient ConvNet design via differentiable neural architecture search. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019.

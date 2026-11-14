# ZK-NAS Method Draft

## 3. Method

### 3.1 Problem Formulation

Given a ZK proving system P (e.g., EZKL/Halo2+plookup), we seek transformer architectures A* that solve:

```
A* = argmax_{A ∈ S} {quality(A)} 
     s.t. ZK_cost(A, P) ≤ budget
```

where S is the transformer search space (Section 3.2), quality(A) is language model quality (perplexity on WikiText-2), and ZK_cost(A, P) is the circuit constraint count under prover P.

Since evaluating quality(A) requires training (hours) and ZK_cost(A, P) requires proving (minutes), we introduce learned proxies to make search tractable.

### 3.2 Search Space

We define a GPT-2-class transformer search space with dimensions:
- d_model ∈ {256, 384, 512, 640, 768, 1024}
- n_heads ∈ {4, 6, 8, 12, 16} (with d_model % n_heads = 0)
- n_layers ∈ {2, 4, 6, 8, 10, 12, 16, 20, 24}
- seq_len ∈ {128, 256, 512} (fixed at 512 for iso-context evaluation)
- ff_mult ∈ {2.0, 4.0}
- activation ∈ {gelu, relu, poly2, linear}
- use_rotary ∈ {True, False}
- use_layernorm ∈ {True, False}
- tie_embeddings ∈ {True, False}

This yields 47,520 valid configurations. For fair comparison with GPT-2 (seq_len=1024), we fix seq_len=512 as the minimum practical context length.

### 3.3 ZK Cost Proxies

**Analytical ZK cost:** Under Halo2+plookup, the dominant cost is matmul constraints. We derive an analytical formula:

```
ZK_cost(A) ≈ n_layers × (attn_cost(d, h, s) + ffn_cost(d, ff, s)) + embedding_cost(V, d)
```

where attn_cost = 4sd² + 2hs²(d/h), ffn_cost = 4fsd², embedding_cost = Vd. This formula takes microseconds to evaluate and eliminates the need for expensive ONNX export (24 seconds per config on CPU).

**Multi-fidelity learned proxy (Approach A):** We train a gradient-boosted ranker (XGBoost) to predict log(ZK_cost) from a feature vector including architectural parameters, ONNX breakdown (used as a feature, not the label), and EZKL compiler observables. Ground truth labels are actual EZKL proving times. The proxy is initialized from analytical estimates and refined as actual proofs are collected.

**Zero-cost proxy (Approach B):** A regression model fitted on architectural features alone, using the ONNX cost estimate as primary feature. Serves as a fast baseline.

### 3.4 Accuracy Proxies

Following LiteTransformerSearch [cite], we use decoder parameter count (non-embedding, non-LM-head parameters) as the primary accuracy proxy. This achieves Spearman ρ = 0.967 on GPT-2-class transformers [LiteTransformerSearch], outperforming gradient-based proxies. Our local validation confirms directional consistency.

### 3.5 ZK-NAS Search Procedure

**Phase 1 (Search, O(1) oracle):** Sample pool_size=100 configurations, score all with decoder_params_score + analytical ZK cost. Select a batch using Expected Hypervolume Improvement (EHVI) acquisition, which computes the expected addition to Pareto hypervolume if each candidate were added. This requires no actual proof evaluations.

**Phase 2 (Calibration, O(N_cal) proofs):** Train a stratified sample of N_cal=30 configurations for 500 gradient steps. Compute Spearman correlation between each zero-cost proxy and calibration PPL. Identifies which proxy most reliably predicts LM quality in our search space.

**Phase 3 (Final training, O(N_final) full runs):** Fully train the top-K Pareto-frontier configurations identified by Phase 1, plus GPT-2 baseline, until convergence. Report actual WikiText-2 perplexity vs ZK cost.

### 3.6 Contribution: Proof-Budget Efficiency

The key contribution of Phases 1-3 is **proof-budget efficiency**: Phase 1 uses zero proofs; Phase 2 uses ~30 short-training runs; Phase 3 uses ~6 full training runs. Total: ~36 oracle calls vs. thousands required by random exhaustive search. In deployment, the savings are even more dramatic when oracle = actual EZKL proofs (10-30 min each).

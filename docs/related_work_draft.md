# ZK-NAS Related Work Draft

## 2. Related Work

### 2.1 Zero-Knowledge Machine Learning

The problem of verifying neural network inference using zero-knowledge proofs has attracted significant attention. Early work [cite zkml EuroSys 2024] established compilation toolchains for ONNX models into ZK circuits and demonstrated that standard computer vision models (ResNet-20, MobileNet) can be proved in minutes with Halo2-based provers. **zkLLM** [ACM CCS 2024, arXiv:2404.16109] introduced specialized lookup arguments (tlookup, zkAttn) for proving transformer inference and achieved verification of 13B-parameter LLM outputs in under 15 minutes. **zkGPT** [USENIX Security 2025] applies constraint fusion and circuit squeeze to achieve GPT-2 inference proofs in under 25 seconds. **NANOZK** [VerifAI@ICLR 2026] uses layerwise proofs and selective verification to achieve 5.7× speedup over EZKL for transformer layers. **TeleSparse** [PoPETS 2025] applies structured sparsification to reduce proof cost.

A key observation across this body of work: **all prior work fixes the transformer architecture and optimizes the proof system.** Our work asks the complementary question: for a given proof system (EZKL/Halo2+plookup), which transformer architectures are cheapest to prove? This represents the first systematic study of architecture-space optimization for ZK-ML.

### 2.2 Hardware-Aware Neural Architecture Search

Hardware-aware NAS [ProxylessNAS, MnasNet, FBNet] optimizes architecture choices to minimize latency on specific hardware targets. The proxy problem is central: training thousands of candidate architectures is impractical, so efficient proxies for hardware cost (lookup tables, regression models) are essential. Our work applies this paradigm to ZK proving systems, with the key difference that ZK cost is qualitatively different from GPU latency: it is dominated by algebraic constraints rather than memory bandwidth, making the architectural cost structure fundamentally different.

**LiteTransformerSearch** [arXiv:2203.02094] is the closest prior work in spirit: it uses decoder parameter count as a proxy for LM quality and applies NAS to GPT-2-class architectures on WikiText-103. Their finding — that decoder parameter count achieves Spearman ρ = 0.967 with converged PPL, outperforming gradient-based proxies — motivates our choice of decoder_params_score as the accuracy proxy in ZK-NAS.

### 2.3 Zero-Cost NAS Proxies

Training-free accuracy predictors have been extensively studied for vision models [SNIP, GraSP, SynFlow, NASWOT, NEAR]. **NEAR** [Javaheripi & Koushanfar, ICLR 2025] achieves activation-agnostic proxy via effective rank of pre/post-activation matrices. However, as shown by Serianni & Kalita [ACL 2023] and our own validation, training-free metrics often underperform simple parameter counting for transformer LMs, where network expressivity correlates strongly with scale. We therefore use decoder_params_score (log non-embedding parameters) as the primary accuracy proxy, validated by LiteTransformerSearch's results.

No prior work has studied zero-cost proxies for ZK proving cost — our proxy learning framework is the first to treat this as a supervised learning problem with actual proof costs as labels.

### 2.4 NAS for Secure Computation

**NASS** [arXiv 2020] applies NAS to minimize latency under garbled circuit-based multi-party computation — a different security model from ZK proofs. The cost structure differs fundamentally: MPC cost is dominated by AND gate counts (analogous to multiplicative depth in ZK), while ZK/PLONK cost under Halo2 is dominated by the number of rows in the constraint matrix (dominated by matmul count). Our work focuses specifically on ZK proofs, which require stronger security guarantees (non-interactive, publicly verifiable) than MPC.

**NeuroChainOps** [Research Square 2025] uses ZK proofs for verifiability of the federated NAS process itself, not for optimizing model architectures against ZK proving cost — an orthogonal use.

"""Per-operator ZK constraint costs (vendored from zkml_bench.metrics.zk_proxy.OP_COSTS).

Vendored per-operator weights used by the author-defined operation-count
stand-in in ``experiments/operator_attribution.py``. These weights support an
explicit heuristic formalization, not an EZKL-measured or universal cost model.
Units are the source proxy's claimed constraints per output element (or per MAC
for MatMul/Gemm/Conv) under Halo2+plookup.
"""

OP_COSTS: dict[str, int] = {
    # Non-linear: comparison-based
    "Relu": 2,
    "LeakyRelu": 3,
    "Clip": 4,
    "Abs": 2,
    "Sign": 2,
    "HardSigmoid": 4,
    "HardSwish": 6,
    "MaxPool": 3,
    "ArgMax": 20,
    "ArgMin": 20,
    # Non-linear: transcendental
    "Sigmoid": 16,
    "Tanh": 16,
    "Exp": 16,
    "Log": 16,
    "Sqrt": 16,
    "Erf": 24,
    "Softmax": 32,
    "LogSoftmax": 32,
    # Trigonometric (same lookup-table tier as Exp/Log; used by RoPE)
    "Sin": 16,
    "Cos": 16,
    "Atan": 24,
    "Atan2": 24,
    # Negation — linear (single constraint)
    "Neg": 1,
    # Normalization
    "LayerNormalization": 24,
    "InstanceNormalization": 24,
    "GroupNormalization": 24,
    "MeanVarianceNormalization": 24,
    "LpNormalization": 20,
    # Semi-linear
    "BatchNormalization": 1,
    # Linear: MAC-based (cost × MAC count, not element count)
    "Gemm": 1,
    "MatMul": 1,
    "Conv": 1,
    # Linear: element-wise
    "Mul": 1,
    "Div": 4,
    "Add": 0,
    "Sub": 0,
    # Reduction
    "ReduceSum": 0,
    "ReduceMean": 1,
    "ReduceMax": 10,
    "ReduceMin": 10,
    "ReduceProd": 1,
    # Pooling
    "AveragePool": 1,
    "GlobalAveragePool": 1,
    # Shape/indexing (free)
    "Flatten": 0,
    "Reshape": 0,
    "Transpose": 0,
    "Squeeze": 0,
    "Unsqueeze": 0,
    "Concat": 0,
    "Split": 0,
    "Gather": 2,
    "GatherElements": 2,
    "Pad": 0,
    "Slice": 0,
    "Tile": 0,
    "Expand": 0,
    "Shape": 0,
    "Cast": 0,
    # Conditional
    "Where": 6,
    # Dropout (identity at inference)
    "Dropout": 0,
    # Recurrent (hook-based fallback; after ONNX export these are decomposed)
    "LSTM": 200,
    "GRU": 120,
    "RNN": 30,
    # Upsampling
    "Resize": 2,
    "Upsample": 2,
    # Pow: computed dynamically based on exponent value
    "Pow": 1,  # placeholder; see _compute_pow_cost_with_value
}

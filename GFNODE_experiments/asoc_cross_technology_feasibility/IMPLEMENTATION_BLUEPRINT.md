# Minimal implementation blueprint

Selected formulation: `SYNCHRONIZED_MULTI_OUTPUT`.

1. Reuse `CleanDataProtocol` separately for each technology without changing its preprocessing.
2. Intersect each split's valid window start timestamps, then build one joint index containing the three row indices.
3. Use one weather-feature input tensor and a target tensor `[batch, 3, 144]`; preserve target-valid masks per head.
4. Reuse ModernTCN-style convolutional backbone and attach exactly three linear H144 heads.
5. Compute masked loss independently per technology before averaging; validation alone selects checkpoint.
6. Formal screening estimate: 3 seeds × one synchronized model; include three independently trained single-technology ModernTCN comparators only if the study question demands the comparison (12 runs total).

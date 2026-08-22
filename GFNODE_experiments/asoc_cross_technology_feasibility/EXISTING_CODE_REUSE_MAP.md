# Existing-code reuse map

| Capability | Evidence | Classification |
|---|---|---|
| Regular timestamp reindex, split, Train-only KNN/IF/scalers, windows | `asoc_clean_decision/asoc_clean_decision.py:60-180` | Direct reuse |
| Seed, validation-only early stopping, prediction and H144 prefix metrics | `asoc_clean_decision/asoc_clean_decision.py:40-52,357-420` | Direct reuse |
| Checkpoint/epoch JSONL/status recovery pattern | `asoc_discrete_viability/benchmark.py:48-66` | Interface adaptation |
| ModernTCN block | `asoc_discrete_viability/benchmark.py:31-35` | Interface adaptation |
| Multi-technology synchronized index, shared backbone and three heads | `cross_technology.py` | Must add |
| Per-technology masked loss and inverse metrics | not present | Must add for formal experiment |
| latency measurement | not implemented in clean benchmark | Must add for formal experiment |

# Scheme A Multisite M2

## Execution record

Source: e60f3482248cc657397db2b665ea0f8957dee7c5 on origin/research/scheme-a-multisite-data-confirmation-r1. Work branch: research/scheme-a-multisite-frozen-training. Draft PR base: research/scheme-a-multisite-data-confirmation-r1.

The user authorized exactly 24 GPU runs: two sites, four frozen compact implementations, seeds 42/43/44. The M1-R configuration file is unchanged. Its stage and training_this_round=false describe historical M1-R; the separate M2 runner executes the next-round authorization. No manuscript, original Scheme A evidence or original checkpoint is modified.

Preflight: M1-R 43/43 passed (52.214 seconds), M2 ordinary arrays 17/17 passed (36.259 seconds), failed/errors/skipped all zero. Exact 96 support records and matrix match. Additional non-training checks passed: preprocessor state serialization roundtrip, independent synthetic metric arithmetic, and exact independent raw-label reconstruction for both sites. These additional checks are not counted as extra formal unit tests.

GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB; PyTorch 2.7.1+cu118, CUDA runtime 11.8. Float32, no AMP, four CPU threads, cuDNN deterministic=True and benchmark=False. Python/NumPy/CPU/CUDA seeds are set per run. Training budget and model configuration remain exactly M1-R.

Local artifacts are confined to ignored .local/multisite_results. Each run stores only best_validation.pt, training history and ordinary metadata; no last checkpoint. Processor state is stored once per site. All 24 Validation-best checkpoints are fixed and recorded before any Test loader is built. Test is a held-out performance split previously used for data-quality auditing, not a completely unseen external validation set.

Finite negative power/GHI and unusual operating observations remain in the data. NIST uses fixed EST/LST and conservative [T-5,T) aggregation; Yulara uses native local coordinates, unrounded off-grid exclusion and +5-minute availability. Both nonblocking documentation limitations from M1-R remain applicable.

Primary: INVERTED_VARIATE_TRAJECTORY, three-seed mean and sample SD, matched comparisons against Last-value and Daily. Other models/ranks are secondary; best-of-four is a post hoc descriptive envelope. The two external sites remain separate from Alice Springs's 17-channel results and its 12/9/2/1/0 and Daily 22/24 evidence.

## Completion evidence

Training is in progress. No Test performance or final success claim is available yet. Final tables, artifact tests and independent evidence will be recorded after all runs complete. No neural-win condition is imposed.

## Table conventions

Deterministic persistence rows are repeated under the three seed identifiers solely to preserve paired CSV keys. They are not three fitted models or independent baseline repetitions; their sample SD is zero. Neural summaries average per-seed metrics, not predictions. Primary and supplementary_daily_matched are separate analysis rows because Daily availability changes the common target-point intersection. Undefined quantities are empty CSV values, not epsilon substitutions.

Recovery checks validate ordinary identity/configuration, checkpoint selection metadata, saved labels/origins/target starts and masks, and checkpoint re-forward agreement. Partial training artifacts are reported as STALE_ARTIFACT rather than silently restarted. There is no automatic extra run or hyperparameter adjustment.

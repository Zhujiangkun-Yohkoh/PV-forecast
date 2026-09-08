# Post hoc supplementary analysis plan — 2026-09-08

Written before computing new scores. Existing Test has been inspected. Frozen neural results remain immutable.

R01/R02: lead 1..144 on complete-H144 common origins; cumulative H12/48/96/144 on both original horizon-specific and common-H144 support. Daily finite-lag intersection identical for methods. No score-based origin filtering.

R03/R07: power-active = true target > 1% Train max. Low-power is the complement, not night. Per-seed counts/SSE/SAE/signed sums; additive full MSE contributions are subset SSE/Nfull. Compute per-seed RMSE before averaging seeds.

R06: non-overlapping chronological origin blocks anchored at Test split midnight; primary 48h, sensitivities 24/72h; 2000 replicates; RNG 20260908; resample same blocks for methods and seeds separately per system. Empty blocks retained; sum SSE/count before each seed RMSE then average seed skill. Report percentile 2.5/97.5 intervals without p-values. Conditional on observed period, fitted models, and block scheme; residual block-boundary correlation and short Alice period remain limits.

R08: Ridge A flattens same 72xD neural history after Train-only pipeline. B appends 144 exact target-minus-24h historical power values plus 144 missing masks. All lag times are before origin. Train-only median imputation for day inputs and StandardScaler on A/B designs. Target uses Train-only target scaler. alpha candidates [0.1,1,10,100,1000]; select minimum full-H144 Validation SSE/count; ties select larger alpha. Deterministic multi-output closed-form Ridge via sufficient statistics/eigendecomposition. All five targets, no seed replicas, no Test tuning or neural training. Save new predictions outside Git.

R09: month defined by origin month. Fixed cases: first common-H144 origin on/after day15 10:00 of each external Test month, within month. Also show seed42 best and worst per-origin H144 Daily squared-error improvement, explicitly post hoc extremes, never choose best seed. Each curve from one origin.

R05: completed metadata and available logs only, no invented curves or extended training.

Alice configured duplicate CSV candidates were byte-identical; use first path in frozen resolver precedence. Only explicitly authorized files read.

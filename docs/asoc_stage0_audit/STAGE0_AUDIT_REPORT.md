# ASOC Stage 0 evidence audit

Audit date: 2026-08-22. Mode: read-only; no existing code, data, result, paper or submission file was modified.

## Overall determination

**Data integrity: FAIL. Time-series evaluation: FAIL. Result provenance: FAIL. ODE/claim consistency: FAIL for continuous-time, arbitrary-resolution and vector-convergence claims.** Current results must not be used for submission.

## Five most severe issues

1. **BLOCKER: global future-aware preprocessing.** KNN, Isolation Forest and feature/target MinMax are fit before chronological split (PVforecast16/GFNODE_experiments/gfnode_solo_benchmark.py:44-75,609-615).
2. **BLOCKER: test is validation/tuning data.** Benchmark/fair/seasonal pass test loader as val loader; fair comparison selects LR by test loss (fair_sota_comparison.py:1063-1075,1087-1122).
3. **BLOCKER: split-after-window overlap/no purge.** create_sequences makes overlapping windows before split (gfnode_solo_benchmark.py:78-83,609-615). Shared source positions are 83–215; no embargo exists.
4. **BLOCKER: destroyed time axis.** Night deletion produces non-5-minute adjacent rows (166/202/175 measured pre-IF Sanyo/Hanwha/Qcells), but H*5/60 is used as an elapsed-hour label (gfnode_solo_benchmark.py:605-607).
5. **BLOCKER: tables cannot be traced.** Table 9–13 mismatches/missing sources are itemized in EXPERIMENT_PROVENANCE.csv. Seasonal code retrains Autumn although paper says without retraining (seasonal_robustness_experiment.py:157-216; 4_Experiments_and_Results_Analysis.md:158-160).

## Timestamp/data findings

Raw primary CSVs each contain 44,064 records from 2018-04-01 00:00:00 to 2018-08-31 23:55:00 and raw deltas are exactly five minutes. After executable daytime/PR filtering, timestamps are no longer uniform. The preprocessor returns arrays but does not serialize timestamps or IF retained-row masks (gfnode_solo_benchmark.py:51-76); exact post-IF split timestamp ranges/cross-date counts therefore cannot be recovered and are correctly CANNOT_VERIFY in TIMESTAMP_AND_SPLIT_AUDIT.csv. This is a reproducibility defect, not an assumption.

## ODE findings

z is [B,128]; t appears in the function signature but is never used (gfnode_solo_benchmark.py:272-278). Grid is dimensionless [0,1] with H+1 points (:385), separate model/checkpoint per H (:605-657). The RK4 option 0.1 is not the actual step for H≥12; each adjacent output update uses 1/H. Five-minute physical duration never reaches f. Figure 12 uses checkpoint weights but random 128-D states, random-start Euler curves and a per-H SVD basis; the field omits the residual term (fig_neural_ode_analysis.py:72-84,119-160).

## Result-source findings

- Table 9 source CSV: Full RMSE .247664 versus paper .238; Level4 .232267 versus .252.
- Table 10: only H48 raw CSV located; Sanyo no-Transformer .218748 versus paper .248; no H144 record.
- Table 11: 20260315 lacks H72/H120; latest seven-H CSV gives Sanyo H48 .243971 versus paper .230.
- Table 12: saved Autumn H12 R2=.891299/RMSE=.536154 versus paper .960/.272; separate Autumn checkpoints exist.
- Table 13: located CSV has baseline rows only, no GFNODE; script uses test loss for early stopping/LR selection.

No multi-seed metrics, SD, CI or test statistics were found. “Within one standard deviation” at 4_Experiments_and_Results_Analysis.md:188 is unsupported. The stated 3.6%–27.2% range is algebraically reproducible only from printed Table13 values by unweighted mean of per-cell percentage reductions: iTransformer 3.6428%, DLinear 27.1548%; it is not the reduction of mean RMSE (about 3.43% and 25.92%) and cannot validate an untraceable table. Code defines retention as Autumn_R2/Spring_R2*100 (seasonal_robustness_experiment.py:259-271), but it retrains both seasons.

The profiling script defines H48/B1, 50 warmups and 200 timings but no hardware/software manifest (computational_profiling.py:31-36,115-141). Located old efficiency CSV says 42.88±4.62ms, not paper 106.37ms. The 1.13M parameter count is plausible; 132.77M FLOPs and 106.37ms are CANNOT_VERIFY.

## Required rerun gate

1. Create timestamp-level chronological train/validation/test partitions; fit all data processors on train only and save masks/scalers.
2. Retain a regular five-minute grid or provide actual elapsed delta-t to the model; never infer hours from filtered row count.
3. Build windows within partitions or apply at least lookback+horizon−1 raw-step purge.
4. Tune/early-stop solely on validation; test once. Give all models a prespecified comparable budget.
5. Run multiple seeds; retain per-seed predictions/checkpoints/metrics and report mean±SD/CI.
6. Regenerate Tables 9–13/Figures 10–14 from immutable raw artifacts and an environment/hardware manifest. Use a true frozen spring→autumn test if retaining seasonal transfer.

## ASOC recommendation

**NO-GO.** The barrier is not only novelty; it is data leakage, invalid horizon semantics, test-set selection, and unreproducible/contradictory results. A later conditional submission requires the rerun gate plus a narrower evidence-based claim set.


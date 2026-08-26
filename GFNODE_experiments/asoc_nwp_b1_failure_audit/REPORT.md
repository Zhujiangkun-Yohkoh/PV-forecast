# Stage B1 NWP screening failure-attribution audit

## Executive verdict

**Final classification: `COMMON_LEARNER_UNFIT`.**

No target-scaling, inverse-transform, checkpoint-selection, label alignment, timestamp alignment, horizon alignment, or Validation-scope implementation error was found. However, the shared `HISTORY_ONLY` learner is not a valid minimum-skill reference for testing incremental NWP information: its mean Validation H144 RMSE is 6.3064 kW, versus 2.7170 kW for a causal Last-value Persistence forecast on exactly the same 59,424 origins and labels. At H12 the gap is more extreme: 10.7048 versus 0.5984 kW. The neural predictions also contain physically impossible tails, and the top 1% of point errors accounts for 90.6% of `HISTORY_ONLY` H144 SSE on average.

Therefore the previous label `NWP_INFORMATION_FAIL` must be narrowed to:

> Under the fixed Stage B1 learner, 1.0° three-hourly GFS DSWRF/TCDC representation, and fixed fusion configuration, RAW_NWP did not improve a common predictor that itself failed minimum-skill and tail-stability checks. These results cannot establish that causal GFS—or NWP generally—contains no PV forecasting information.

Operational decision remains unchanged: **stop B2, do not download 2023 NWP, and do not develop NWP fusion v2/v3.** The reason is no longer “NWP information was disproved”; it is that B1 did not provide a valid information-value experiment and the current route is closed by the project’s predeclared stopping discipline.

## Scope and sources

This audit was read-only and used only 2021 Train and 2022 Validation artifacts.

- B1 implementation/config/report: `C:/Users/Zhujiangkun-Yohkoh/Desktop/光伏项目_最新/PVforecast16_nwp_minimal_screen/GFNODE_experiments/asoc_nwp_minimal_screen/`
- Prepared artifact: `.../asoc_nwp_minimal_screen/results/prepared_data.npz`
- Nine run directories: `.../results/{HISTORY_ONLY,RAW_NWP,AGE_LEAD_RELIABILITY}/seed_{42,43,44}/`
- Original PV source, read without modification: `C:/Users/Zhujiangkun-Yohkoh/Desktop/光伏项目_最新/PV_improve_v1/原始Dataset/5min pv active power data/17 Sanyo.csv`
- B0.1 semantic reference: `C:/Users/Zhujiangkun-Yohkoh/Desktop/光伏项目_最新/PVforecast16_nwp_feasibility/GFNODE_experiments/asoc_nwp_forecast_feasibility/`
- Code-only comparators: `GFNODE_experiments/asoc_multirate_information_screen/run_information_screen.py` and `GFNODE_experiments/asoc_discrete_viability/benchmark.py`

No 2023 file, Test prediction, Test metric, optimizer, backward pass, or training loop was invoked.

## A. Artifact and checkpoint consistency

All 9/9 run directories contain `epochs.jsonl`, `last.pt`, `best_validation.pt`, and `validation_H144.npz`. Every saved prediction and label has shape `[59424,144]`, all values are finite, and all models/seeds have elementwise-identical labels, forecast-origin timestamps, and NWP-valid masks. Saved labels equal the H144 targets reconstructed from `prepared_data.npz`; saved origins equal the prepared Validation origins.

For all nine runs, the checkpoint epoch and stored Validation MSE exactly match the minimum logged Validation MSE and epoch. A deterministic 32-origin CPU forward pass from each best checkpoint agrees with the saved GPU predictions to mean absolute differences of 0.000164–0.000723 kW; the largest isolated difference is 0.031582 kW. This small device-dependent numerical difference is negligible beside the observed 100–400 kW tail values and supports, but does not mathematically prove bitwise, that the saved predictions came from the recorded best checkpoint.

## B. Target scaling and inverse transform

The target scaler was fit on Train only:

- Train target minimum: 0.000000 kW.
- Train target maximum/range: 6.047201 kW.
- Training transform: `(y_kW - target_min) / target_range`.
- Prediction inverse: `scaled_prediction * target_range + target_min`.

The B1 prediction routine contains one inverse transform, after model inference. Labels are stored directly in physical source units. The saved values and checkpoint re-forward are consistent with one inverse transform, not zero or two. No NaN, Inf, broadcasting error, shape mismatch, or horizon shift was found.

## C. Prediction tails and physical range

The absence of a scaling-code error does not imply a fit-for-purpose predictor. The unconstrained linear H144 head produces extreme out-of-distribution outputs:

| Model | H144 negative predictions | Outside Train target range | Above 2× Train max | H144 prediction extrema (kW) | Mean top-1% SSE share |
|---|---:|---:|---:|---:|---:|
| HISTORY_ONLY | 26.66% | 27.59% | 0.69% | -177.93 to 354.78 across seeds | 90.63% |
| RAW_NWP | 18.91% | 19.68% | 0.60% | -376.16 to 360.05 | 94.13% |
| AGE_LEAD_RELIABILITY | 21.01% | 21.74% | 0.53% | -318.45 to 421.39 | 90.15% |

For `HISTORY_ONLY` seed 42, the H144 RMSE is 8.9385 kW while MAE is 1.5956 kW; its maximum absolute error is 350.44 kW, and its top 1% errors contribute 95.90% of SSE. The RMSE comparison is therefore dominated by a small tail rather than typical errors.

The long CSV contains p0, p0.1, p1, p5, p50, p95, p99, p99.9 and p100 for predictions, labels, absolute errors, and squared errors, plus the top 20 point errors for every run.

## D. Minimum-skill Persistence challenge

Last-value Persistence repeats the power observed at the forecast origin through H144. It uses exactly the same Validation origins, H144 labels, and validity mask as all B1 models and never reads future inputs.

| Horizon | Persistence RMSE | Persistence MAE | Persistence range_nRMSE | Persistence R² | HISTORY_ONLY mean RMSE ± SD | HISTORY/Persistence RMSE ratio |
|---|---:|---:|---:|---:|---:|---:|
| H12 | 0.5984 | 0.2734 | 0.0990 | 0.8865 | 10.7048 ± 5.6641 | 17.89× |
| H48 | 1.3877 | 0.7617 | 0.2295 | 0.3862 | 9.3625 ± 4.3425 | 6.75× |
| H96 | 2.2787 | 1.4539 | 0.3768 | -0.4654 | 7.1437 ± 3.0506 | 3.14× |
| H144 | 2.7170 | 1.9151 | 0.4493 | -0.9251 | 6.3064 ± 2.4608 | 2.32× |

The exact H144 `HISTORY_ONLY` seed results are 8.9385, 5.9171 and 4.0634 kW, all worse than Persistence. Its R² values are -19.8351, -8.1304 and -3.3057. Persistence itself becomes weak at long horizon, but that strengthens rather than weakens the conclusion: a valid common learner should at least dominate it at short horizons, where Persistence has R² 0.8865 and the neural baseline is 18 times worse in RMSE.

This diagnostic is necessary because protocol tests and Git provenance establish legality and reproducibility, not predictive competence. Without a minimum-skill reference, failure of a fusion model cannot distinguish absent NWP information from failure of the shared predictor.

## E. 2021→2022 feature drift

The most serious covariate shifts are configuration-level inputs, not target scaling:

- `Active_Energy_Delivered_Received` is cumulative and nonstationary. Train spans 135,869–145,863 with mean 140,802.65; Validation spans 145,863–158,136 with mean 152,235.84. **99.916%** of finite Validation values lie outside the Train min/max, 76.01% exceed |Train z|=3, and 26.71% exceed |z|=5.
- `Performance_Ratio` has a Train p99 of 102.77 but Validation p99 of 33,596.77 and maximum 129,056.59. 2.14% of Validation values exceed |Train z|=10.
- `Wind_Speed` is 100% missing in both Train and Validation, so its scaled value is constant and its missing indicator is always one.
- `Active_Power`, GHI, temperature and humidity have broadly comparable Train/Validation distributions; their outside-Train-range ratios are below 0.2% except for tiny sensor-edge excursions.

Window MAE is positively associated with the origin-time cumulative-energy representation (Spearman 0.085–0.217 across models/seeds), consistent with but not by itself proving that the nonstationary feature participates in the failure. This audit does not delete the feature or rerun a model.

## F. Error-scenario decomposition and seed 42

The seed-42 anomaly is not concentrated in the 27 one-cycle fallback origins. Their H144 RMSE is only 1.33–2.68 kW, while the 59,397 no-fallback origins retain the extreme aggregate errors.

Instead, errors are localized in a few chronological regions:

- October contributes 86.57% of `HISTORY_ONLY` seed-42 H144 SSE, 89.03% for `RAW_NWP`, and 88.43% for `AGE_LEAD_RELIABILITY`.
- December contributes a further 8.01–9.31%; June contributes about 2.7–2.8%.
- Validation continuous segment 152 has H144 RMSE 52.08, 91.74 and 73.24 kW for the three models, respectively; segments 62 and 176 are the next dominant clusters.
- All three model families show the same chronological concentration, which points to shared input/learner instability rather than an NWP-only defect.

The CSV also gives month, forecast-origin hour, daylight/night, 176 continuous segments, fallback status, Train-defined past-variation groups, and all four horizon prefixes for every model/seed, including each scope’s SSE contribution.

## G. Training-process read-only findings

All logged losses are finite. `HISTORY_ONLY` best epochs are 5, 3 and 7, followed by ordinary early stopping. `RAW_NWP` selects epoch 1 for all seeds and stops after six epochs because every subsequent Validation loss is worse; this is a real logged Validation deterioration, not a checkpoint bookkeeping error. Its initial/best scaled Validation MSE values are 6.5587, 0.7358 and 1.0987.

`AGE_LEAD_RELIABILITY` selects epochs 1, 7 and 9. The seed spread is already large in the shared history predictor and is amplified by the residual NWP branch. No learning-rate, patience, initialization, or epoch change was made.

## H. Code-level difference from previously validated ModernTCN uses

The B1 network belongs to the same depthwise/pointwise ModernTCN family, but it is not a fair performance reproduction of the prior `TRAJECTORY_ONLY` task:

- B1 uses 13 history variables, 13 missing indicators and four time features; it includes cumulative energy and pathological PR values. The previous MEAN_ONLY task uses historical power, time encodings and MB channel means/validity.
- B1 target scaling is Train min/range; MEAN_ONLY uses Train mean/SD; the clean benchmark uses a Train-only scaler.
- B1 is cross-year Train 2021 → Validation 2022; MEAN_ONLY splits within 2022.
- B1 predicts H144; MEAN_ONLY predicts H12. Clean benchmark H144 uses different data years/protocol.
- B1 `HISTORY_ONLY` has 683,856 parameters because of its input dimension and H144 head.
- All relevant heads are unconstrained linear outputs.
- B1 early-stops on scaled Validation MSE; MEAN_ONLY uses physical Validation RMSE; clean benchmark uses Validation MSE.
- B1 uses Train median fill plus explicit missing indicators; the clean protocol uses its own Train-only KNN/IF/scaler pipeline.

Accordingly, prior RMSE values are not compared directly here.

## Interpretation and decision

### Was there an implementation error?

**No explicit implementation-invalidating error was found.** The scaler is Train-only, inverse transformation occurs once, checkpoint epochs match logs, predictions are checkpoint-reproducible within CPU/GPU floating tolerance, labels/timestamps/masks are aligned, and only legal Validation samples are evaluated.

### Why `COMMON_LEARNER_UNFIT` rather than `CONFIGURATION_LEVEL_NWP_FAIL`?

`CONFIGURATION_LEVEL_NWP_FAIL` requires a common predictor with reasonable minimum skill and normal tails. B1 fails both prerequisites: every `HISTORY_ONLY` seed loses to Persistence at H144, it catastrophically loses at H12, and 82.3–95.9% of its SSE lies in the top 1% of points. Thus the observed RAW_NWP degradation cannot isolate incremental NWP information.

### Does this reopen the route?

No. This audit corrects causal interpretation, not the project decision. B2 remains stopped, 2023 remains sealed and undownloaded, and no v2/v3 is justified. Reopening would require a new, separately authorized study with a demonstrably stable common predictor before any NWP fusion claim could be tested.

### Unique next recommendation

**Archive Stage B1 as a failed information-value experiment and return to the clean deterministic application-paper route; do not spend additional GPU or data-download effort on Scheme B.**

## Self-check results

Eight ordinary assertions passed:

1. no training function, optimizer, backward pass or gradient update was called;
2. no 2023 artifact or Test metric was read;
3. original PV, B1 code, prepared artifact and results were not written;
4. Persistence used elementwise-identical labels, origins and masks;
5. metrics were recomputed from saved H144 predictions and labels;
6. the long diagnostic CSV was written and reread successfully;
7. tail percentages and SSE contributions are directly recomputable from its source artifacts;
8. outputs were confined to `GFNODE_experiments/asoc_nwp_b1_failure_audit/`.

The original PV file’s size and modification time were checked before and after the source scan and remained unchanged.

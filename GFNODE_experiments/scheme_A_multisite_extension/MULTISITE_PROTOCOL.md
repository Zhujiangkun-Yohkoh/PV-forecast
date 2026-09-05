# Scheme A-M1 frozen external-site protocol

Protocol frozen for review. Training authorization withheld: see REPORT.md. No training runner is supplied in M1.

## Scope and source

Source: origin/manuscript/clean-pv-benchmark-jrse-final-polish at ede66987e56eb8863287624476f8b8ff3e201897. Branch research/scheme-a-multisite-data-confirmation. This source predates local language/affiliation edits in the submission worktree; those edits are not imported. No PR #14 merge, master access, rebase or force push.

Two external facilities: YULARA_COMBINED (Sails in the Desert combined system 3, 106.6 kW) and NIST_GROUND (Gaithersburg). Do not count Yulara constituent arrays as independent sites. This is external replication with separate site-specific fitting, not zero-shot transfer. Do not pool the external 16 site/horizon/scope summaries with the original 24 comparisons.

## Source configuration and protection

Only explicit YULARA_RAW_FILE and NIST_GROUND_2017_DIRECTORY in an ignored local JSON are accepted. No committed absolute paths, raw SHA generation, whole-disk search or old data-root fallback. NIST must contain exactly the expected 365 month/day CSV names. Missing/extra CSVs, duplicate or reversed timestamps require correction. Raw files are read-only; size and mtime_ns are checked before/after. No raw/derived full CSV, checkpoint, prediction array or cache is committed.

## Seven channels and preprocessing

Order: historical_ac_active_power, ambient_temperature, global_horizontal_irradiance, power_missing, temperature_missing, ghi_missing, isolation_forest_flag. Target: future measured AC active power. No time-of-day, year, PR, humidity, NWP or site-specific inputs.

Keep finite negative power/GHI unless the provider defines an invalid code. Non-numeric and infinite values become missing and are counted separately. Candidate -999/-7999 in a common field requires correction, not an invented deletion rule; none is present in the current common fields. InvPAC_kW_Avg is diagnostic only, never a target. No Test-error, shutdown, snow or low-power filtering. Labels are never imputed.

Future preprocessing order: KNNImputer(n_neighbors=5) fit on Train numeric inputs; transform; IsolationForest(n_estimators=100, contamination=0.01, random_state=42) fit on Train imputed numeric inputs; append original missing masks and IF outlier indicator; feature MinMaxScaler fit on all seven Train augmented columns; target MinMaxScaler fit on finite un-imputed Train target. The feature scaler includes indicator columns, following the old augmentation/scaling order. Nothing is fitted in M1; fit-guard tests use recording test doubles.

Explicit changes to original Scheme A: _valid_power and Daily previously excluded negative power; M1 retains finite negatives by user instruction. The old validator assumes 17 channels and a different numeric order, so its data pipeline cannot be called directly. These changes do not alter original code or evidence.

## Time, aggregation and availability

NIST: TIMESTAMP is LST in Data Dictionary v1.0. Verify every ISO offset is -05:00, then use fixed EST without DST. Never localize to America/New_York. Pyra1 is Ground GHI; directly supplied Pyra1_Wm2_Avg is not converted again. PwrMtrP_kW_Avg is AC meter real power. Pyra1_mV_Avg is absent in this download.

NIST five-minute grouping: timezone fixed UTC-05:00; anchor local midnight; origin 2017-01-01 00:00:00-05:00; closed=left; label=right. Bin T uses distinct raw timestamps T-5,T-4,T-3,T-2,T-1 minutes, i.e. [T-5,T). Each variable requires all five distinct observations to be finite; otherwise that variable is missing, even when a partial mean could be computed. No default resample behavior, interpolation or partial-bin averages.

Output T is a conservative availability/end coordinate. If a raw minute timestamp t could denote either interval start or end, that value is not available before t+1 minute; hence the complete bin is available at T. Dictionary TIMESTAMP Max and Train energy diagnostics support end stamping, but the current export's precise convention is not proven. This conservative delay is an explicit assumption, not a claim of exact provider alignment. Target values use the same availability coordinate; physical sub-minute alignment is not established.

Yulara: retain fixed provider local coordinates and label PROVIDER_LOCAL_TIME_OFFSET_UNCONFIRMED. Do not invent UTC or apply DST. Official glossary establishes five-minute AC power averages; the resource-specific section does not supply complete interval definitions. Each regular record is available at raw timestamp+5 minutes, conservatively covering an interval that could start at its timestamp. The derived output is indexed by availability time. Exclude the two off-grid records without rounding; retain their original values and adjacent records in the audit. Do not replace missing grid labels with off-grid measurements without provider evidence.

Only 2017 source records enter this candidate. Keep every five-minute coordinate and missing row. For split containment, a bin ending at split midnight cannot be an input because its measurement interval begins before the split. The first input interval starts at midnight and is available 00:05; with 72 steps the first legal origin is 06:00. This stricter interval-containment convention is intentional.

## Splits and windows

Train: 2017-01-01 00:00:00--2017-08-31 23:59:59. Validation: September 1--30 with the same day boundaries. Test: October 1--December 31. Interpret boundaries on each site's fixed local coordinate. Five-minute grid; lookback=72; output=144; prefixes=12,48,96,144. Input availability <= origin; first target is origin+5 minutes. Neither input measurement intervals nor targets cross a split. Never delete missing rows and stitch time.

A descriptive horizon-eligible origin has a finite real origin power and complete valid target prefix. Inputs can be missing and later Train-imputed. All methods share those origins. Fitting and checkpoint selection use the full-H144 subset with this same finite-origin requirement; old _build_full_h144 did not require origin power, so this is explicitly stricter. Checkpoint objective remains full-H144 Validation global masked MSE (summed SSE / valid count), never short-prefix scores or unweighted batch means.

DATA_AUDIT_SUMMARY.csv includes every site/split/horizon/primary-or-Daily/full-or-daylight support: origin and point counts, first/last origin, month coverage, input-window missing rate, split input/label missing rates and Train daylight threshold. Selected-label missing rate is zero by complete-prefix eligibility; split missing rate includes excluded labels. These are data counts, not Test scores. Any nonfinite future neural prediction must fail its run, not silently improve scores by dropping points.

## Daylight and baselines

Daylight = true future target > 0.01 * corresponding Train maximum, before scaling. It is descriptive and not a deployed detector. Last-value uses real finite origin power, not imputed power. Daily joins each target timestamp to exactly target-24 hours on the fixed coordinate, never row shift. The causal lag can precede the split when an actual record exists in the supplied year; neural windows remain split-local. Missing lag points are removed identically from Daily, Last-value and every neural method. Identical point masks ensure identical outcomes, not equal historical information. Daily versus six-hour history compares information strategies rather than pure architectures.

## Prespecified analyses and budget

Primary model INVERTED_VARIATE_TRAJECTORY is fixed before external Test prediction, based on original 12/24 primary wins and mean rank 1.875. Primary: its three-seed Test means and sample SD, matched RMSE skill vs Last-value and vs Daily, each site/horizon/scope separately. Skill=1-RMSE_model/RMSE_reference on identical support; zero reference error yields undefined skill, not epsilon substitution. SD uses ddof=1; averaging predictions into an ensemble does not replace mean seed metrics.

Secondary: all four models, each seed, ranks, MAE, Train-range nRMSE, bias, R² and full/daylight differences. Retain negative R²; zero target variance yields undefined R². Train range is max-min, not Test range or assumed AC capacity. Descriptive only: post hoc best-of-four model-mean RMSE envelope, never the primary or deployable model. No best-seed reports, Test selection, unplanned win-count p-values or treating 16 dependent summaries as independent tests.

Budget copied from original config: batch 256, AdamW, learning rate 0.001, weight decay 1e-5, max 25 epochs, patience 5, min_delta 1e-8, gradient clipping 1.0, seeds 42/43/44. No new architecture or hyperparameter search; model budgets remain equal. These are next-round candidates only; M1 performs none of these operations. Seven-channel parameter changes follow input dimension, not a new model claim.

## Exact frozen 24-run candidate matrix

| # | Run ID | Site | Model | Seed |
|---|---|---|---|---:|
| 1 | `YULARA_COMBINED__DISCRETE_RECURRENT_TRAJECTORY__seed42` | YULARA_COMBINED | DISCRETE_RECURRENT_TRAJECTORY | 42 |
| 2 | `YULARA_COMBINED__DISCRETE_RECURRENT_TRAJECTORY__seed43` | YULARA_COMBINED | DISCRETE_RECURRENT_TRAJECTORY | 43 |
| 3 | `YULARA_COMBINED__DISCRETE_RECURRENT_TRAJECTORY__seed44` | YULARA_COMBINED | DISCRETE_RECURRENT_TRAJECTORY | 44 |
| 4 | `YULARA_COMBINED__INVERTED_VARIATE_TRAJECTORY__seed42` | YULARA_COMBINED | INVERTED_VARIATE_TRAJECTORY | 42 |
| 5 | `YULARA_COMBINED__INVERTED_VARIATE_TRAJECTORY__seed43` | YULARA_COMBINED | INVERTED_VARIATE_TRAJECTORY | 43 |
| 6 | `YULARA_COMBINED__INVERTED_VARIATE_TRAJECTORY__seed44` | YULARA_COMBINED | INVERTED_VARIATE_TRAJECTORY | 44 |
| 7 | `YULARA_COMBINED__JOINT_PATCH_TRAJECTORY__seed42` | YULARA_COMBINED | JOINT_PATCH_TRAJECTORY | 42 |
| 8 | `YULARA_COMBINED__JOINT_PATCH_TRAJECTORY__seed43` | YULARA_COMBINED | JOINT_PATCH_TRAJECTORY | 43 |
| 9 | `YULARA_COMBINED__JOINT_PATCH_TRAJECTORY__seed44` | YULARA_COMBINED | JOINT_PATCH_TRAJECTORY | 44 |
| 10 | `YULARA_COMBINED__DEPTHWISE_TCN_TRAJECTORY__seed42` | YULARA_COMBINED | DEPTHWISE_TCN_TRAJECTORY | 42 |
| 11 | `YULARA_COMBINED__DEPTHWISE_TCN_TRAJECTORY__seed43` | YULARA_COMBINED | DEPTHWISE_TCN_TRAJECTORY | 43 |
| 12 | `YULARA_COMBINED__DEPTHWISE_TCN_TRAJECTORY__seed44` | YULARA_COMBINED | DEPTHWISE_TCN_TRAJECTORY | 44 |
| 13 | `NIST_GROUND__DISCRETE_RECURRENT_TRAJECTORY__seed42` | NIST_GROUND | DISCRETE_RECURRENT_TRAJECTORY | 42 |
| 14 | `NIST_GROUND__DISCRETE_RECURRENT_TRAJECTORY__seed43` | NIST_GROUND | DISCRETE_RECURRENT_TRAJECTORY | 43 |
| 15 | `NIST_GROUND__DISCRETE_RECURRENT_TRAJECTORY__seed44` | NIST_GROUND | DISCRETE_RECURRENT_TRAJECTORY | 44 |
| 16 | `NIST_GROUND__INVERTED_VARIATE_TRAJECTORY__seed42` | NIST_GROUND | INVERTED_VARIATE_TRAJECTORY | 42 |
| 17 | `NIST_GROUND__INVERTED_VARIATE_TRAJECTORY__seed43` | NIST_GROUND | INVERTED_VARIATE_TRAJECTORY | 43 |
| 18 | `NIST_GROUND__INVERTED_VARIATE_TRAJECTORY__seed44` | NIST_GROUND | INVERTED_VARIATE_TRAJECTORY | 44 |
| 19 | `NIST_GROUND__JOINT_PATCH_TRAJECTORY__seed42` | NIST_GROUND | JOINT_PATCH_TRAJECTORY | 42 |
| 20 | `NIST_GROUND__JOINT_PATCH_TRAJECTORY__seed43` | NIST_GROUND | JOINT_PATCH_TRAJECTORY | 43 |
| 21 | `NIST_GROUND__JOINT_PATCH_TRAJECTORY__seed44` | NIST_GROUND | JOINT_PATCH_TRAJECTORY | 44 |
| 22 | `NIST_GROUND__DEPTHWISE_TCN_TRAJECTORY__seed42` | NIST_GROUND | DEPTHWISE_TCN_TRAJECTORY | 42 |
| 23 | `NIST_GROUND__DEPTHWISE_TCN_TRAJECTORY__seed43` | NIST_GROUND | DEPTHWISE_TCN_TRAJECTORY | 43 |
| 24 | `NIST_GROUND__DEPTHWISE_TCN_TRAJECTORY__seed44` | NIST_GROUND | DEPTHWISE_TCN_TRAJECTORY | 44 |

## Reproduction and authorization

Run with NumPy, pandas, scikit-learn and PyTorch from repository root:

```
python GFNODE_experiments/scheme_A_multisite_extension/audit_multisite_data.py --paths .local/multisite_paths.json
python GFNODE_experiments/scheme_A_multisite_extension/test_multisite_protocol.py --paths .local/multisite_paths.json
```

The local ignored JSON is owner-populated; no data fallback/download. The audit outputs only a small summary CSV. Forward tests use freshly initialized CPU modules at 7 and 17 channels; strict random 17-state rejection is checked in memory, not by opening an old checkpoint. Each parameter tensor is perturbed under inference mode, output dependence checked and values restored. This demonstrates synthetic tensor participation, not gradient trainability. Runtime guards reject actual estimators' fit methods, training helpers, backward, AdamW construction, checkpoint save/load and real-data prediction helpers in the forward path. Existing code is imported inertly for model classes only.

No neural/risk/IF fitting, actual Validation/Test prediction, or manuscript rewrite. Authorization remains false while REPORT.md identifies missing evidence. A correction decision does not authorize these runs; a later READY decision can authorize only this frozen matrix in a subsequent turn.

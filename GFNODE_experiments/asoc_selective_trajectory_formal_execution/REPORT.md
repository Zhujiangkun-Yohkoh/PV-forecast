# Scheme C1-S3 — formal data confirmation and guarded execution

**Final scientific decision: `C1_FORMAL_DATA_FAIL`.**

The new annual sources fail the preregistered readiness boundary, so the guarded GPU path was not entered. No base forecaster, risk estimator, calibration threshold, Final-Test prediction, error, coverage, or AURC was produced.

## Formal source selection and annual audit

| Year | Formal absolute path | Bytes | Unique target-year seconds | First UTC | Last UTC | Missing seconds | Out-of-year | Structural anomalies |
|---:|---|---:|---:|---|---|---:|---:|---|
| 2021 | `C:\Users\Zhujiangkun-Yohkoh\Desktop\光伏项目_最新\PV_improve_v1\原始Dataset\高分辨率气象数据集\C1_fresh_downloads\2021\fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2021.csv` | 1,726,481,334 | 18,403,200 | 2021-06-02 00:00:00 | 2021-12-31 23:59:59 | 13,132,800 | 0 | column=0; glued=0; truncated=0; Data Error=0 |
| 2022 | `C:\Users\Zhujiangkun-Yohkoh\Desktop\光伏项目_最新\PV_improve_v1\GFNODE_experiments\asoc_multirate_redownload_validation\fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2022.csv` | 3,011,901,400 | 31,536,000 | 2022-01-01 00:00:00 | 2022-12-31 23:59:59 | 0 | 0 | column=0; glued=0; truncated=0; Data Error=0 |
| 2023 | `C:\Users\Zhujiangkun-Yohkoh\Desktop\光伏项目_最新\PV_improve_v1\原始Dataset\高分辨率气象数据集\C1_fresh_downloads\2023\fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2023.csv` | 1,774,300,567 | 167,952 | 2023-01-01 00:00:01 | 2023-01-02 22:39:12 | 31,368,048 | 18,489,565 | column=212; glued=1; truncated=211; Data Error=211 |

The 2021 fresh file contains only 2 June–31 December. The 2023 fresh file contains only 1–2 January 2023, then records from 2024 and 2025; its year transition contains a glued timestamp and `Data Error`/truncated rows. The excluded old damaged 2023 file was not used. The 2022 authoritative redownload was reproduced exactly.

All three headers expose `Timestamp_UTC` plus separate `Irradiance_MB0/MB1/MB2 [W/m-2]`; exported Local is ignored and ACST is computed as UTC+09:30. No System Status or quality field is present in these irradiance files, so no unsupported status meaning is inferred.

## MB missingness and five-minute aggregation

| Year | MB0 missing | MB1 missing | MB2 missing | Complete timestamp bins | Three-channel complete | Partial | Empty |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 781,177 | 781,579 | 781,759 | 61,344 | 58,711 | 2,633 | 43,776 |
| 2022 | 3,992 | 3,236 | 3,320 | 105,120 | 99,211 | 5,909 | 0 |
| 2023 | 59 | 58 | 42 | 558 | 464 | 96 | 104,560 |

Aggregation is right-closed `(t-5 min, t]`; partial numeric missingness is retained with per-channel valid fraction/mask, while zero-timestamp bins break continuity. Nothing was interpolated or repaired.

## Five-stage common-origin audit

| Stage | Expected origins | Common legal origins | Strict three-channel origins | First | Last | Segments | Months | Seasons |
|---|---:|---:|---:|---|---|---:|---|---|
| BASE_TRAIN | 105,037 | 58,122 | 54,067 | 2021-06-02T15:25 | 2021-12-31T22:55 | 13 | 6;7;8;9;10;11;12 | spring;summer;winter |
| BASE_MODEL_VALIDATION | 34,477 | 31,750 | 31,315 | 2022-01-01T05:55 | 2022-04-30T22:55 | 7 | 1;2;3;4 | autumn;summer |
| RISK_FIT | 35,341 | 32,509 | 31,676 | 2022-05-01T05:55 | 2022-08-31T22:55 | 6 | 5;6;7;8 | autumn;winter |
| RISK_CALIBRATION | 35,053 | 34,969 | 2,300 | 2022-09-01T05:55 | 2022-12-31T22:55 | 2 | 9;10;11;12 | spring;summer |
| FINAL_TEST | 105,037 | 604 | 0 | 2023-01-01T05:55 | 2023-01-03T08:10 | 1 | 1 | summer |

Each stage excludes 83 calendar positions by construction (71 initial history positions and 12 terminal target positions). The code combines adjacent UTC annual sources before interpreting ACST. It therefore treats the missing first 9.5 ACST hours of a UTC-year source as an explicit boundary effect, not file corruption. In this execution, however, the much larger 2021/2023 source gaps independently fail the full-year conditions.

## Frozen implementation specification

The committed config fixes the 14 causal input channels, `DEPTHWISE_TCN_TRAJECTORY` architecture inherited from C1-S0R, AdamW training protocol, Train-only imputation/scaling, risk target/range, HistGradientBoosting risk model, scope-matched order-statistic calibration, stable-tie AURC, bootstrap design, and all seven formal success conditions. Since data readiness failed, these frozen definitions were not executed or tuned.

## Readiness conditions

- PASS — `fresh_2021_and_2023_exist`
- FAIL — `fresh_2021_and_2023_have_31536000_unique_seconds`
- PASS — `main_interval_is_one_second`
- FAIL — `fresh_year_first_last_utc_are_exact`
- FAIL — `no_structural_anomaly_in_fresh_years`
- PASS — `mb_fields_units_compatible`
- PASS — `formal_2023_source_is_not_excluded_old_file`
- PASS — `all_stages_have_three_array_common_origins`
- FAIL — `base_train_and_final_test_cover_12_months_four_seasons`
- PASS — `risk_fit_and_calibration_cover_planned_months`
- PASS — `final_test_outcomes_not_read_or_generated_before_freeze`
- PASS — `raw_sources_unchanged`
- PASS — `frozen_daylight_thresholds_match`

Failed conditions: `fresh_2021_and_2023_have_31536000_unique_seconds`, `fresh_year_first_last_utc_are_exact`, `no_structural_anomaly_in_fresh_years`, `base_train_and_final_test_cover_12_months_four_seasons`.

## Tests, execution, and protection

- Fixture tests: 13/13 passed; 0 skipped
- Real-array tests: 11/11 passed; 0 skipped
- GPU training: **No (0/9 runs)**
- Risk fitting: **No**
- Final-Test performance access: **No**
- Original PV/NWP size and nanosecond mtime unchanged: **True**

## Seven formal method conditions

All seven conditions are **NOT_EVALUATED**: macro coverage, per-array minimum coverage, macro AURC improvement, array-level AURC direction, matched-Persistence skill, seed-macro accepted-RMSE reduction, and seed-level AURC direction. A data failure is not a method failure.

## Conclusion

`C1_FORMAL_DATA_FAIL`. The failure is data-specific: 2021 is not a full year and the new 2023 export is a mixed-year, structurally damaged file. Under the preregistration, this ends the execution before training. No interpolation, alternative year, repair, C1 v2/v3, or scientific method conclusion is proposed.

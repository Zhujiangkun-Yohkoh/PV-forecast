# Site 17 Sanyo Multirate Information-Gain Screen

## Scope and outcome

All 6/6 prescribed GPU training runs completed (MEAN_ONLY and HF_DYNAMICS, seeds 42/43/44). Under the fixed ModernTCN protocol, the added past-second irradiance dynamics did **not** provide a stable test improvement beyond the three channels' five-minute means and availability indicators. HF_DYNAMICS lost on H12 in all three seeds and its paired mean H12 RMSE change was **-1.401%** (negative means worse). This is an experiment-specific screening result, not a claim that high-frequency irradiance can never be useful, and it does not trigger NODE training.

## Immutable task definition and sources

- Array: DKASC Site 17 Sanyo, rated capacity 6.3 kW; Active_Power is evaluated in kW.
- PV source: `C:/Users/Zhujiangkun-Yohkoh/Desktop/光伏项目_最新/PV_improve_v1/原始Dataset/5min pv active power data/17 Sanyo.csv`.
- High-frequency source: the structurally complete 2022 file validated in commit `15c04eac30a0b09ab72bbff8c1ca47447a759153`, explicitly named in `config.json`. The damaged 2023 export is never searched or read.
- Time basis: only `Timestamp_UNIX [s]`/UTC is used, transformed to the PV clock as UTC+09:30 (ACST). The file's Local field is not read.
- Each high-frequency summary at forecast origin `t` uses only `(t-5 min, t]`. Targets are exactly `t+5 min` through `t+60 min`; no weather observation after `t` is an input.
- Split-before-window dates: Train 2022-01-01--08-31, Validation 2022-09-01--10-31, Test 2022-11-01--12-31. Windows are independently built inside each split.

The final common forecast-origin counts are 66,842 Train, 17,485 Validation, and 17,401 Test. Both conditions use these exact origins and element-identical targets. A sample requires 72 consecutive valid historical PV values and 12 consecutive valid future PV values, but does not require complete high-frequency history. Empty channel-interval features receive a Train-only median fill and mask 0; partial intervals retain their valid fraction.

## Conditions and implementation

Both conditions use the same direct H12 ModernTCN implementation extracted without architectural change from `asoc_discrete_viability/benchmark.py`: 64 channels, four depthwise/pointwise blocks, kernel 5, no dropout, and one direct 12-output head.

MEAN_ONLY has 14 inputs per five-minute step: historical Active_Power; sine/cosine time-of-day and day-of-year; separate MB0/MB1/MB2 means; and each channel's valid fraction and validity mask. HF_DYNAMICS adds, separately for each MB channel, standard deviation, range, first-last change, maximum absolute consecutive difference, and least-squares slope (15 extra inputs). The three MB channels are never averaged.

All imputations and feature/target standardization statistics are fit only on Train timestamps. The ramp threshold, 0.1506998 kW (Train 90th percentile of absolute target-step changes), is also derived only from Train. AdamW uses learning rate 0.001, weight decay 1e-5, batch 256, at most 25 epochs, patience 5, and Validation RMSE checkpoint selection. These are the previously validated ModernTCN settings rather than condition-specific tuning.

## Main test results

Values are mean +/- sample SD across three seeds. nRMSE is RMSE / 6.3 kW.

| Condition | Prefix | RMSE (kW) | nRMSE | MAE (kW) | R2 |
|---|---:|---:|---:|---:|---:|
| MEAN_ONLY | H3 | 0.407580 +/- 0.004657 | 0.064695 +/- 0.000739 | 0.192891 +/- 0.004486 | 0.950193 +/- 0.001139 |
| HF_DYNAMICS | H3 | 0.409443 +/- 0.010211 | 0.064991 +/- 0.001621 | 0.207127 +/- 0.018671 | 0.949720 +/- 0.002523 |
| MEAN_ONLY | H6 | 0.444666 +/- 0.004056 | 0.070582 +/- 0.000644 | 0.209639 +/- 0.002851 | 0.940717 +/- 0.001079 |
| HF_DYNAMICS | H6 | 0.447491 +/- 0.005346 | 0.071030 +/- 0.000849 | 0.220527 +/- 0.013086 | 0.939959 +/- 0.001439 |
| MEAN_ONLY | H12 | 0.486647 +/- 0.004473 | 0.077246 +/- 0.000710 | 0.233442 +/- 0.001897 | 0.928953 +/- 0.001304 |
| HF_DYNAMICS | H12 | 0.493440 +/- 0.000481 | 0.078324 +/- 0.000076 | 0.244272 +/- 0.004137 | 0.926960 +/- 0.000142 |

### Paired seed results

Relative change is `(RMSE_MEAN_ONLY - RMSE_HF_DYNAMICS) / RMSE_MEAN_ONLY * 100%`; positive favors HF_DYNAMICS.

| Seed | H3 RMSE A / B (kW) | H3 change | H6 RMSE A / B (kW) | H6 change | H12 RMSE A / B (kW) | H12 change |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.403082 / 0.405088 | -0.498% | 0.440023 / 0.444578 | -1.035% | 0.481767 / 0.492921 | -2.315% |
| 43 | 0.412381 / 0.402131 | +2.486% | 0.447523 / 0.444234 | +0.735% | 0.487622 / 0.493527 | -1.211% |
| 44 | 0.407277 / 0.421110 | -3.396% | 0.446452 / 0.453661 | -1.615% | 0.490553 / 0.493871 | -0.676% |
| Paired mean +/- SD | -- | -0.469 +/- 2.941% | -- | -0.638 +/- 1.224% | -- | -1.401 +/- 0.836% |

HF_DYNAMICS wins 1/3 seeds at H3, 1/3 at H6, and 0/3 at H12.

## Daylight, ramp, and trajectory diagnostics

| Condition | Prefix | Daylight RMSE (kW) | Ramp RMSE (kW) | First-difference MAE (kW) |
|---|---:|---:|---:|---:|
| MEAN_ONLY | H3 | 0.566570 +/- 0.006994 | 0.862528 +/- 0.007330 | 0.133052 +/- 0.002359 |
| HF_DYNAMICS | H3 | 0.566609 +/- 0.010808 | 0.841978 +/- 0.010791 | 0.139442 +/- 0.004471 |
| MEAN_ONLY | H6 | 0.618345 +/- 0.005746 | 0.915105 +/- 0.009902 | 0.132837 +/- 0.000193 |
| HF_DYNAMICS | H6 | 0.620757 +/- 0.005335 | 0.905034 +/- 0.006262 | 0.139732 +/- 0.007297 |
| MEAN_ONLY | H12 | 0.676781 +/- 0.006023 | 0.970870 +/- 0.009335 | 0.132490 +/- 0.001308 |
| HF_DYNAMICS | H12 | 0.685252 +/- 0.000526 | 0.971259 +/- 0.002716 | 0.138163 +/- 0.002536 |

The ramp subset gives a small short-prefix advantage but no H12 advantage; daylight and trajectory-difference diagnostics do not support stable information gain.

## Parameters and runtime

| Condition | Parameters | Difference | Mean total training time | Mean epoch time | Mean inference latency/sample |
|---|---:|---:|---:|---:|---:|
| MEAN_ONLY | 74,444 | reference | 56.696 s | 2.825 s | 0.0221 ms |
| HF_DYNAMICS | 75,404 | +1.290% | 33.838 s | 2.975 s | 0.0244 ms |

HF_DYNAMICS has a 5.3% higher per-epoch time and 10.3% higher measured per-sample inference time. Its lower total training time is caused by earlier Validation stopping (best epochs 7, 7, 5 versus 15, 13, 17), not lower computational cost.

## Protocol tests and artifact verification

`test_protocol.py` passes all 13 ordinary checks: common origins; identical targets; disjoint splits; no crossing; causal input endpoint; exact output timing; right-closed historical HF interval; Train-only fit/fill; no Test loader in training; Validation-only checkpointing; independent MB channels; masked partial missingness; and Train-only ramp threshold. All six saved Test artifacts have element-identical labels and forecast-origin timestamps. Each run locally contains epoch JSONL, last and best-validation checkpoints, predictions, labels, timestamps, daylight mask, and ramp mask under `results/<condition>/<seed>/`; `results/` is intentionally not committed.

## Screening interpretation

The observed second-level dynamics are not a stable source of incremental H12 accuracy for this fixed Site 17/2022/ModernTCN experiment. The only favorable seed is seed 43 at H3/H6, while H12 degrades in all seeds and the extra dynamics increase MAE and trajectory-difference error. Therefore the evidence does not justify proceeding automatically to a Neural ODE or presenting high-frequency dynamics as an established contribution. Any decision to investigate a different multirate representation would be a new research question, not a continuation triggered by these results.

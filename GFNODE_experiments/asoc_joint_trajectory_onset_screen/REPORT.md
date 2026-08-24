# Site 17 Joint Power-Trajectory and First-Ramp-Onset Screen

## Reviewer verdict

All 6/6 new GPU runs completed without numerical divergence. ONSET_HAZARD consistently improves first-onset AUROC, AUPRC, Brier score, F1 and miss rate over STEP_MULTITASK, including daylight. It therefore models first-event occurrence more appropriately than converting step probabilities into an onset proxy. However, onset-time MAE is worse and both multitask models degrade deterministic power RMSE materially versus the reused TRAJECTORY_ONLY model. The current joint route is **not ready for full baselines, ablation, or paper rewriting**: event-head value is demonstrated, but the required “preserve trajectory accuracy” part is not.

RAMP_AWARE_NCQ interval-width modulation remains FAIL. Nothing here supports improved probability intervals, Neural ODE, cross-site generalization, or deployment.

## Protocol

- Site 17 Sanyo, 2022, ACST derived from UTC+09:30; 5-minute grid; lookback 72; one direct H12 trajectory.
- Train 2022-01-01--08-31: 66,842 windows; Validation 2022-09-01--10-31: 17,485; Test 2022-11-01--12-31: 17,401.
- Inputs are exactly the 14 previously verified MEAN_ONLY fields. No post-origin PV or irradiance and no failed HF_DYNAMICS fields are used.
- Train-only ramp threshold: 0.1506998 kW (absolute power-change 90th percentile); Train step BCE positive weight: 8.9996.
- ModernTCN: channels 64, four blocks, kernel 5, dropout 0. AdamW, LR 0.001, weight decay 1e-5, batch 256, no scheduler, maximum 25 epochs, patience 5, min_delta 1e-8, gradient clipping 1.0, no mixed precision, num_workers 0.
- Standardized power MSE is unchanged. STEP_MULTITASK uses weighted step BCE and lambda 0.2. ONSET_HAZARD uses numerically stable cause-specific first-event NLL and lambda 0.2. Each best checkpoint uses only its Validation composite objective.

## Event distribution shift

| Split | Ramp-step prevalence | First-onset prevalence | Upward onset | Downward onset |
|---|---:|---:|---:|---:|
| Train | 10.000% (80,214/802,104) | 21.629% (14,457/66,842) | 10.444% | 11.185% |
| Validation | 14.092% (29,568/209,820) | 25.468% (4,453/17,485) | 12.365% | 13.103% |
| Test | 17.473% (36,486/208,812) | 32.314% (5,623/17,401) | 17.959% | 14.355% |

The substantial prevalence rise makes Test harder and cautions against treating this single-year split as stationary. No threshold, class weight, or calibration was changed after Train.

## First-onset results by seed

Window event threshold is fixed at 0.5. STEP_MULTITASK AUROC/AUPRC/Brier use the auxiliary `1-product(1-p_step)` score, but its event decision/time comes from the first 0.5-threshold step transition; it is not a proper first-event distribution.

| Model | Seed | Scope | AUROC | AUPRC | Brier | F1 | Miss rate | Time MAE (steps) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| STEP | 42 | Full | .9474 | .8456 | .1996 | .5422 | .4787 | 2.487 |
| STEP | 43 | Full | .9491 | .8604 | .1802 | .5511 | .5070 | 2.750 |
| STEP | 44 | Full | .9546 | .8603 | .1951 | .5501 | .4901 | 2.296 |
| HAZARD | 42 | Full | .9658 | .8858 | .0703 | .8309 | .2095 | 2.792 |
| HAZARD | 43 | Full | .9610 | .8806 | .0701 | .8549 | .1337 | 2.656 |
| HAZARD | 44 | Full | .9576 | .8684 | .0759 | .8362 | .1641 | 2.967 |
| STEP | 42 | Daylight | .8465 | .8456 | .3554 | .5422 | .4787 | 2.487 |
| STEP | 43 | Daylight | .8513 | .8604 | .3240 | .5511 | .5070 | 2.750 |
| STEP | 44 | Daylight | .8675 | .8604 | .3463 | .5503 | .4901 | 2.296 |
| HAZARD | 42 | Daylight | .9002 | .8858 | .1266 | .8309 | .2095 | 2.792 |
| HAZARD | 43 | Daylight | .8863 | .8807 | .1261 | .8550 | .1337 | 2.656 |
| HAZARD | 44 | Daylight | .8761 | .8684 | .1367 | .8362 | .1641 | 2.967 |

## Mean +/- SD and differences

| Scope / metric | STEP_MULTITASK | ONSET_HAZARD | Hazard absolute change | Relative change |
|---|---:|---:|---:|---:|
| Full AUROC | .9504 +/- .0038 | .9615 +/- .0041 | +.0111 | +1.16% |
| Full AUPRC | .8555 +/- .0085 | .8783 +/- .0089 | +.0228 | +2.66% |
| Full Brier | .1916 +/- .0102 | .0721 +/- .0033 | -.1195 | -62.36% |
| Full F1 | .5478 +/- .0049 | .8407 +/- .0126 | +.2929 | +53.47% |
| Full miss rate | .4920 +/- .0142 | .1691 +/- .0381 | -.3228 | -65.62% |
| Full onset-time MAE | 2.511 +/- .228 steps | 2.805 +/- .156 steps | +.294 steps (+1.47 min) | +11.72% worse |
| Daylight AUROC | .8551 +/- .0110 | .8875 +/- .0121 | +.0324 | +3.79% |
| Daylight AUPRC | .8555 +/- .0085 | .8783 +/- .0090 | +.0228 | +2.67% |
| Transition AUROC | .9092 +/- .0081 | .9213 +/- .0120 | +.0121 | +1.34% |
| Transition AUPRC | .8135 +/- .0084 | .8247 +/- .0220 | +.0112 | +1.38% |

Hazard improves both discrimination metrics in all three seeds and daylight, and greatly lowers misses. It does not improve onset timing among jointly detected events.

## Direction

| Model | Up recall | Down recall | Direction accuracy | Up/Down Test samples |
|---|---:|---:|---:|---:|
| STEP_MULTITASK | .4274 +/- .0325 | .2675 +/- .0058 | .7015 +/- .0268 | 3,125 / 2,498 |
| ONSET_HAZARD | .5546 +/- .0832 | .5124 +/- .0187 | .6445 +/- .0196 | 3,125 / 2,498 |

Hazard avoids single-class failure and substantially improves both cause recalls, especially downward. Its lower conditional direction accuracy reflects many more detected events and should be interpreted with the much lower miss rate.

## Lead-time behavior

True first-onset counts by h are 930, 930, 771, 633, 511, 426, 345, 283, 242, 210, 182 and 160. Mean Hazard identification declines from 0.870 at h1 to 0.585 at h12; STEP falls from 0.989 at h1 to roughly 0.37--0.45 for h2--h12. The late-horizon cells are smaller and are not treated as stable standalone advantages. Upward/downward counts for every lead and every seed are retained in `metrics_per_seed.csv`.

## Power trajectory results

| Model | H3 RMSE | H6 RMSE | H12 RMSE | H12 MAE |
|---|---:|---:|---:|---:|
| TRAJECTORY_ONLY | .40758 +/- .00466 | .44467 +/- .00406 | .48665 +/- .00447 | .23344 +/- .00190 |
| STEP_MULTITASK | .43639 +/- .00917 | .46439 +/- .00509 | .50352 +/- .00471 | .26850 +/- .00381 |
| ONSET_HAZARD | .43853 +/- .00536 | .46712 +/- .00390 | .50344 +/- .00363 | .26819 +/- .00100 |

Relative to TRAJECTORY_ONLY, H12 RMSE worsens 3.47% for STEP and 3.45% for HAZARD; H3 worsens 7.07% and 7.59%. Hazard and STEP have essentially identical H12 RMSE (-0.00008 kW Hazard advantage). H12 daylight RMSE is .67678/.69565/.69679; ramp-step RMSE .97087/.97327/.97198; onset-window RMSE .80823/.82595/.82526 for TRAJECTORY_ONLY/STEP/HAZARD. First-difference MAE is .13249/.14155/.13654 kW. Thus joint training does not preserve the best deterministic trajectory.

## Runs, parameters, and cost

| Model/seed | Best epoch | Actual epochs | Stop | Finite | Training s |
|---|---:|---:|---|---|---:|
| STEP 42/43/44 | 5 / 6 / 8 | 10 / 11 / 13 | early stopping | yes | 55.39 / 63.83 / 78.84 |
| HAZARD 42/43/44 | 7 / 6 / 5 | 12 / 11 / 10 | early stopping | yes | 79.22 / 72.24 / 66.04 |

Shared backbone has 19,136 parameters and the common power head 55,308. STEP event head has 55,308 parameters (129,752 total); Hazard event head has 165,924 (240,368 total), 85.25% more total parameters than STEP. Mean epoch time is 5.79 s STEP and 6.57 s Hazard; inference is 0.0388 and 0.0392 ms/sample. The event gain costs a much larger head but little inference latency.

Fixed 10-bin reliability rows are saved for every model/seed. Mean full-Test ECE is approximately .237 for the STEP auxiliary any-step score and .038 for Hazard. No post-hoc calibration was applied.

## Fairness and verification

All three models share element-identical Test labels, origin timestamps and masks. TRAJECTORY_ONLY artifacts are reused from `asoc_multirate_information_screen/results/MEAN_ONLY/<seed>/test_predictions.npz`; no deterministic retraining occurred. STEP and Hazard share the exact backbone, power head configuration, input columns, samples, optimizer, loader settings and seeds. Only event head and event loss differ.

`test_protocol.py` passes 19 ordinary tests: output shapes; conditional normalization; event/no-event mass; non-increasing survival; mutually exclusive causes; one first event; correct h1 prior state; Train-only threshold; split/window isolation; causal inputs; no Test loader; Validation-only checkpointing; artifact equality; finite event/no-event NLL; finite forward/backward gradients; mean loss convention; and shared backbone/power-head configuration. Six saved new Test artifacts and all metrics were recomputed without retraining.

## Recommendation

Do **not** enter full baselines or paper rewriting with the current joint objective. Hazard modeling is scientifically justified over step multitask for first-onset detection, but the 3.45% H12 and 7.59% H3 RMSE degradation violates the joint-value premise, while onset timing also worsens. The single next step should be a narrowly scoped, predeclared test of gradient-conflict mitigation that preserves the already validated trajectory model; do not add new encoders, NODEs, probability intervals, sites, or broad hyperparameter searches until trajectory preservation is demonstrated.

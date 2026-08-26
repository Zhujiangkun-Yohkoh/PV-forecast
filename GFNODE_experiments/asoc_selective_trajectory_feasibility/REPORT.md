# Scheme C1-S0R — Scope-Matched Selective PV Calibration Correction

## Corrected screening decision

**C1_GO — screening evidence only.** After correcting the calibration population, the C1 screen satisfies the preregistered daylight H12 checks. This does **not** establish a publishable method and does not authorize further training. It means only that the corrected feasibility screen should be reviewed by the supervisor before deciding whether to establish a strict Train/Model-Validation/Risk-Calibration/Test protocol.

No deep model was trained or fine-tuned in C1-S0R. Existing best checkpoints were used only under `torch.inference_mode()` to reconstruct Validation predictions; the nine preregistered CPU `HistGradientBoostingRegressor` risk estimators were refit with unchanged parameters.

## Confirmed C1-S0 implementation error

The original code calculated every risk threshold from all 8,743 `RISK_CALIBRATION` windows and then evaluated daylight coverage. Thus the previously reported “80% daylight” threshold was actually an 80% **full-timeline** threshold. Nighttime low-risk observations changed the score distribution, so applying that threshold to daylight windows yielded only **58.75% mean Test daylight coverage** (58.99%, 58.48%, 58.78% by seed). It was not valid to describe those results as a nominal 80% daylight operating point.

C1-S0R constructs a calibration-membership mask and intersects it with the matching Validation scope before taking each quantile. Full thresholds use all `RISK_CALIBRATION` windows; daylight thresholds use only the 4,260 origin-daylight windows in `RISK_CALIBRATION`. Test scores never enter threshold calculation. Daylight remains causal and unchanged: observed power at forecast origin greater than **0.063 kW**.

The correction changes the official H12 daylight result from **58.75% ± 0.26%** coverage to **78.09% ± 1.52%**. The old result is retained here only to document the corrected error and is no longer a primary result.

## Artifacts, data split, and fairness

- Task: Site 17 Sanyo, 2022, 5-minute grid, lookback 72, direct H12 trajectory, evaluated at H3/H6/H12.
- Source experiment: `C:/Users/Zhujiangkun-Yohkoh/Desktop/光伏项目_最新/PVforecast16/GFNODE_experiments/asoc_multirate_information_screen`.
- Prepared data: `results/prepared_data.npz` under that source experiment.
- Existing Test predictions: `results/MEAN_ONLY/{42,43,44}/test_predictions.npz`.
- Existing best checkpoints: `results/MEAN_ONLY/{42,43,44}/best_validation.pt`; best epochs 15, 13, and 17.
- Validation: 17,485 time-ordered origins; first 8,742 are `RISK_FIT`, final 8,743 are `RISK_CALIBRATION`.
- Test: 17,401 origins, including 8,972 origin-daylight windows.
- Three seeds have element-identical Test labels and origins. ModernTCN and Persistence use identical accepted masks.
- Train target range for normalized trajectory loss: **5.993634 kW**. Train-only high-change threshold: **0.691913 kW**.

The frozen deep checkpoint previously used the complete original Validation split for checkpoint selection. Therefore this remains a feasibility screen. A later confirmatory study would require a newly established four-part protocol; C1-S0R itself does not provide that confirmation.

## Scope-matched thresholds and realized coverage

At target coverage 0.8, the H12 thresholds and realized coverage were:

| Seed | Full threshold | Full calibration coverage | Full Test coverage | Daylight threshold | Daylight calibration coverage | Daylight Test coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.088077 | 79.995% | 78.858% | 0.144693 | 80.000% | 76.906% |
| 43 | 0.086900 | 79.995% | 78.593% | 0.143789 | 80.000% | 77.552% |
| 44 | 0.085438 | 79.995% | 78.748% | 0.146248 | 80.000% | 79.804% |

The full and daylight masks and thresholds are materially different. All three corrected daylight Test coverages lie in the preregistered [70%, 90%] interval. Calibration coverage differs from its target only by empirical-quantile discreteness/ties.

For the full risk model, the scope-matched H12 daylight operating points were:

| Target coverage | Test coverage mean ± SD | Accepted RMSE, kW | RMSE reduction | Matched Persistence RMSE, kW | High-error acceptance | High-change acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 50.99% ± 1.36% | 0.2801 ± 0.0175 | 58.52% ± 2.35% | 0.6054 ± 0.0059 | 21.61% | 10.22% |
| 70% | 70.01% ± 0.88% | 0.4313 ± 0.0113 | 36.12% ± 1.14% | 0.7041 ± 0.0034 | 48.19% | 33.11% |
| 80% | 78.09% ± 1.52% | 0.5037 ± 0.0181 | 25.41% ± 2.09% | 0.7674 ± 0.0090 | 61.91% | 48.25% |
| 90% | 87.03% ± 2.27% | 0.5802 ± 0.0200 | 14.07% ± 2.22% | 0.8366 ± 0.0093 | 77.50% | 67.55% |

`high_error_acceptance_rate` is the fraction of actual Test high-error windows within the scope that are nevertheless accepted. It is not precision, false discovery rate, or calibration error.

## Corrected H12 daylight results by seed

| Seed | Test coverage | Accepted count | Unselected RMSE, kW | Accepted RMSE, kW | RMSE reduction | Accepted MAE, kW | Matched Persistence RMSE, kW |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 76.906% | 6,900 | 0.6687 | 0.4881 | 27.01% | 0.2948 | 0.7598 |
| 43 | 77.552% | 6,958 | 0.6765 | 0.4995 | 26.16% | 0.2969 | 0.7649 |
| 44 | 79.804% | 7,160 | 0.6803 | 0.5236 | 23.04% | 0.3033 | 0.7774 |
| Mean ± SD | 78.087% ± 1.521% | — | 0.6752 ± 0.0059 | 0.5037 ± 0.0181 | 25.41% ± 2.09% | 0.2983 ± 0.0044 | 0.7674 ± 0.0090 |

ModernTCN is better than Last-value Persistence on the identical accepted daylight origins in 3/3 seeds. This comparison prevents the selective gain from being attributed solely to accepting trivially persistent windows.

## Risk ranking and simple causal challenges

H12 daylight ranking metrics for the full risk model were:

| Metric | Mean ± sample SD | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|---:|
| Spearman | 0.7947 ± 0.0124 | 0.7805 | 0.8008 | 0.8029 |
| AUROC | 0.9040 ± 0.0275 | 0.8724 | 0.9185 | 0.9213 |
| AUPRC | 0.9272 ± 0.0037 | 0.9262 | 0.9313 | 0.9242 |
| Risk–coverage AUC, lower better | 0.04063 ± 0.00145 | 0.04189 | 0.04096 | 0.03904 |

At their own scope-matched 80% daylight thresholds:

| Risk method | Test coverage | Accepted RMSE, kW | RMSE reduction |
|---|---:|---:|---:|
| Full risk model | 78.09% ± 1.52% | 0.5037 ± 0.0181 | 25.41% ± 2.09% |
| Recent variation | 85.08% ± 0.00% | 0.5669 ± 0.0055 | 16.04% ± 0.09% |
| Model–Persistence disagreement | 86.81% ± 0.38% | 0.6136 ± 0.0149 | 9.13% ± 1.74% |

The full estimator improves risk–coverage AUC by **13.45%** and accepted RMSE by **11.16%** relative to the best simple causal score; both exceed the required 5% incremental criterion. The realized coverages differ across methods, so accepted-RMSE comparison must be read together with risk–coverage AUC rather than as a perfectly coverage-matched head-to-head test.

## Oracle headroom

At exact 80% Test daylight coverage, oracle selection reduces H12 RMSE by **44.32% ± 0.07%** across seeds. This remains only a future-loss upper bound and is not deployable. It establishes headroom but does not substitute for calibrated selection.

## Corrected natural-day cluster bootstrap

The old implementation sampled only days containing accepted windows and compared against a fixed full-sample RMSE. C1-S0R instead samples with replacement from **all 61 Test daylight natural days**, keeps the fixed acceptance mask, and recomputes unselected RMSE, accepted RMSE, improvement, Persistence skill, and coverage within every replicate. No replicate recalibrates a threshold; no replicate lacked accepted observations.

| Seed | Accepted RMSE reduction, mean [95% CI] | Matched Persistence skill, mean [95% CI] | Realized coverage, mean [95% CI] |
|---:|---:|---:|---:|
| 42 | 26.83% [21.77%, 32.14%] | 35.73% [31.52%, 40.20%] | 76.86% [71.86%, 81.68%] |
| 43 | 25.94% [20.75%, 30.95%] | 34.68% [30.43%, 39.61%] | 77.56% [72.48%, 82.35%] |
| 44 | 22.90% [17.84%, 27.92%] | 32.79% [27.75%, 37.94%] | 79.83% [75.10%, 84.53%] |

The intervals support positive selective improvement and positive matched-Persistence skill in all seeds, while also showing the uncertainty induced by day-level temporal clustering.

## Prespecified decision checks

| Check | Corrected result | Status |
|---|---:|---|
| Three H12 daylight Test coverages in [0.70, 0.90] | 0.7691 / 0.7755 / 0.7980 | Pass |
| Mean accepted-RMSE reduction at least 10% | 25.41% | Pass |
| At least 2/3 seeds improve | 3/3 | Pass |
| Mean Spearman at least 0.50 | 0.7947 | Pass |
| Mean AUROC at least 0.75 | 0.9040 | Pass |
| ModernTCN no worse than matched Persistence in at least 2/3 seeds | 3/3 | Pass |
| Full estimator at least 5% better than best simple score by AURC or accepted RMSE | 13.45% AURC; 11.16% RMSE | Pass |

Accordingly, the prior `C1_NO_GO_SIGNAL_WEAK` conclusion is **withdrawn for this implementation error** and replaced with `C1_GO` at screening level. The permitted wording is: **C1 screening passes after correcting calibration scope; supervisor review is required before deciding whether to enter a strict four-part formal protocol.**

## Literature boundary

`LITERATURE_OVERLAP_MATRIX.csv` was not changed because the correction is statistical, not methodological. The prior novelty conclusion remains: no reviewed single paper triggered the exact `NOVELTY_BLOCKED` combination, but the gap is narrow because selective time-series forecasting, bounded abstention, PV confidence forecasting, and time-series conformal risk control already exist. No claim of inventing selective forecasting is defensible at this stage.

## Tests and implementation checks

- **7/7 focused regression tests passed** in `test_protocol.py`: scope-specific threshold sources, distinct masks, in-scope calibration coverage, Test exclusion from quantiles, paired bootstrap recomputation, and all-day sampling frame.
- **13/13 runtime array checks passed**: no deep-training API, fit/calibration isolation, scope-matched masks, actual calibration coverage checks, Test isolation, causal feature timestamps, artifact identity, distinct masks, all-day bootstrap frame, output confinement, source-file immutability, and required metric fields.
- `metrics.csv` contains 4,013 data rows and was re-imported successfully for structural verification.
- No optimizer, backward pass, training epoch, new checkpoint, or deep-model weight update occurred.

## Reviewer interpretation

Selective forecasting can still manufacture an attractive error number by rejecting difficult cases. The corrected result is credible only because it reports realized coverage, rejected-set behavior, high-error acceptance, an all-day paired bootstrap, and matched Persistence on the exact accepted origins. Even so, the 2022 Test period has already been used for this feasibility decision, and the original deep checkpoint used the full Validation period. These facts prevent confirmatory or publication-level claims.

The next action is **not** to train immediately. The supervisor should first decide whether the corrected screening evidence justifies one preregistered formal study with genuinely separated Model-Validation and Risk-Calibration periods plus a previously untouched Test period. Scheme A, master, the original worktree, source artifacts, and the literature matrix remain unchanged.

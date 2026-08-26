# Scheme C1-S0 — Forecastability-Aware Risk-Controlled Selective PV Trajectory Forecasting

## Final decision

**C1_NO_GO_SIGNAL_WEAK**

The screen found substantial oracle headroom and a strong past-only error-ranking signal, but the preregistered 80% acceptance threshold calibrated on the later half of Validation transferred to only **58.75% ± 0.26% daylight coverage** on Test. This is outside the required 70%–90% interval for all three seeds. The accepted-set RMSE reduction is therefore obtained with materially more abstention than the advertised 80% operating point and cannot be treated as risk-controlled 80%-coverage performance. Under the prespecified decision rules, C1 is closed; no C1 v2/v3 or formal four-segment experiment is recommended.

## Scope, artifacts, and fairness

- Task: Site 17 Sanyo, 2022, 5-minute grid, lookback 72, direct H12 (60-minute) trajectory; evaluated at H3/H6/H12.
- Deep forecast: existing `MEAN_ONLY / TRAJECTORY_ONLY ModernTCN`, seeds 42/43/44. No deep model was trained or fine-tuned.
- Prepared source: `C:/Users/Zhujiangkun-Yohkoh/Desktop/光伏项目_最新/PVforecast16/GFNODE_experiments/asoc_multirate_information_screen/results/prepared_data.npz`.
- Test sources: `.../results/MEAN_ONLY/{42,43,44}/test_predictions.npz`.
- Checkpoints: `.../results/MEAN_ONLY/{42,43,44}/best_validation.pt`, best epochs 15/13/17.
- Validation predictions were absent and were reconstructed in memory from the existing best checkpoints under `torch.inference_mode()`; no new checkpoint or prediction artifact was saved.
- The 17,485 Validation origins were sorted and split without shuffling into 8,742 `RISK_FIT` and 8,743 `RISK_CALIBRATION` origins. Test contained 17,401 origins, including 8,972 origin-daylight cases.
- Predictions, labels, origins, and task configuration were checked seed-by-seed. Test labels and origins were element-identical across all three seeds. ModernTCN and Last-value Persistence always used exactly the same accepted origins.
- The original deep checkpoint had already used all Validation data for checkpoint selection. This makes C1-S0 a feasibility screen, not a confirmatory selective-forecasting experiment. A successful screen would have required a future Train/Model-Validation/Risk-Calibration/Test design; the screen did not pass.

The Train target range used for normalized trajectory loss was **5.993634 kW**. High-error labels used the seed- and horizon-specific 80th percentile of `RISK_FIT` normalized trajectory loss; H12 thresholds were 0.038867, 0.044241, and 0.048093 for seeds 42, 43, and 44. The causal high-change proxy threshold was the Train 90th percentile of past-12-step maximum absolute PV change, **0.691913 kW**.

## Fixed risk estimators and causal features

Nine CPU risk regressors were fit: one fixed `HistGradientBoostingRegressor` for each seed and H3/H6/H12. All parameters match `config.json`; no search or post-result feature changes were made. The target was `log1p(L_h)`.

Features used only information available at forecast origin: clock/calendar terms; causal origin-daylight status; past PV statistics at 12/36/72 steps; separate MB0/MB1/MB2 past-12 mean, dispersion, variation, and valid-ratio features; and forecast-derived disagreement/trajectory-shape features. The MB channels were not averaged. Two prespecified simple scores were evaluated: past-12 PV maximum absolute change and ModernTCN–Persistence mean absolute disagreement.

## Oracle headroom

At exact 80% Test daylight coverage, oracle selection reduced H12 RMSE by **44.32% ± 0.07%** across seeds. Unselected ModernTCN H12 daylight RMSE was 0.6687/0.6765/0.6803 kW; oracle accepted RMSE was 0.3722/0.3763/0.3793 kW. The corresponding matched Persistence RMSE was 0.6705/0.6715/0.6730 kW. Thus the `C1_NO_GO_NO_HEADROOM` condition is not applicable.

The oracle is a diagnostic upper bound based on future realized loss. It is neither deployable nor comparable to a fixed calibration threshold without matching realized coverage.

## Risk-ranking ability

For Test daylight H12, the full risk model achieved:

| Metric | Mean ± sample SD | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|---:|
| Spearman with actual normalized loss | 0.7947 ± 0.0124 | 0.7805 | 0.8008 | 0.8029 |
| AUROC for fixed high-error event | 0.9040 ± 0.0275 | 0.8724 | 0.9185 | 0.9213 |
| AUPRC | 0.9272 ± 0.0037 | 0.9262 | 0.9313 | 0.9242 |
| Risk–coverage AUC (lower better) | 0.04063 ± 0.00145 | 0.04189 | 0.04096 | 0.03904 |

The signal is directionally stable in 3/3 seeds and exceeds the prespecified Spearman/AUROC requirements. H3/H6/H12 daylight Spearman means were 0.7466/0.7750/0.7947; AUROC means were 0.8705/0.8887/0.9040.

Relative to simple causal scores at H12 daylight:

| Method | Spearman | AUROC | Risk–coverage AUC | 80%-threshold accepted RMSE (kW) | Actual coverage |
|---|---:|---:|---:|---:|---:|
| Full risk model | 0.7947 | 0.9040 | 0.04063 | 0.3335 | 58.75% |
| Recent variation | 0.6620 | 0.8345 | 0.04694 | 0.4466 | 64.42% |
| Model–Persistence disagreement | 0.2282 | 0.5986 | 0.06810 | 0.6132 | 69.91% |

The full model improves risk–coverage AUC by **13.45%** and accepted RMSE by **25.32%** relative to the better simple challenge. Its complexity therefore has empirical ranking value. That value does not rescue the failed coverage-transfer requirement.

## Fixed calibration thresholds: 50/70/80/90%

The following H12 daylight results use thresholds computed only on `RISK_CALIBRATION`; Test was not re-ranked to force a desired coverage.

| Calibration quantile | Test coverage mean ± SD | Accepted RMSE kW | RMSE reduction vs full | Matched Persistence RMSE kW | False-safe rate | High-change acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 5.72% ± 2.86% | 0.1030 ± 0.0241 | 84.77% ± 3.48% | 0.3396 ± 0.0952 | 0.24% | 0.00% |
| 70% | 40.65% ± 0.79% | 0.2295 ± 0.0030 | 66.00% ± 0.74% | 0.5758 ± 0.0160 | 12.43% | 3.64% |
| 80% | 58.75% ± 0.26% | 0.3335 ± 0.0042 | 50.60% ± 0.76% | 0.6307 ± 0.0037 | 30.82% | 16.98% |
| 90% | 77.56% ± 1.34% | 0.4996 ± 0.0171 | 26.02% ± 1.93% | 0.7643 ± 0.0078 | 61.00% | 47.13% |

The nominal-to-realized coverage shift is large and monotone: the threshold transports poorly from September–October calibration to November–December daylight conditions. At the prespecified 80% threshold the actual coverages were 58.99%, 58.48%, and 58.78%. Therefore the apparently large 50.60% RMSE reduction is not evidence of an 80%-coverage system.

On the full timeline, the same 80% threshold yielded 78.73% ± 0.13% coverage and 56.38% ± 0.56% RMSE reduction. The daylight failure is obscured by abundant low-risk nighttime origins, illustrating why full-timeline selective metrics alone are unsafe for PV claims.

## Matched Persistence and block-bootstrap uncertainty

At the fixed 80% calibration threshold and the identical accepted daylight origins, ModernTCN H12 RMSE was 0.3320/0.3383/0.3303 kW, while Persistence RMSE was 0.6267/0.6339/0.6316 kW. ModernTCN is better in 3/3 seeds. The matched comparison rules out the interpretation that the accepted subset merely makes Persistence equally sufficient.

Natural-day moving-block bootstrap (1,000 replicates per seed) gave:

- accepted-set RMSE reduction versus unselected ModernTCN: seed 42 50.18% [44.31, 55.85], seed 43 49.79% [43.05, 55.73], seed 44 51.36% [44.76, 57.65];
- matched-Persistence skill: seed 42 46.86% [41.17, 52.30], seed 43 46.44% [40.84, 52.38], seed 44 47.71% [41.25, 53.79].

These intervals quantify the accepted subset actually produced by the fixed thresholds. They do not repair its coverage shortfall.

## Literature overlap and novelty threat

Fifteen high-threat works were checked at method level. No single paper in the matrix simultaneously contains all of: PV multi-horizon trajectories, a past-only difficulty score, explicit forecast-level acceptance, calibrated risk/coverage control, and a matched-coverage Persistence plus daylight/high-change audit. Accordingly, the exact blocking rule `NOVELTY_BLOCKED` is **not** triggered.

However, the algorithmic gap is narrow and high risk:

- *Bounded-Abstention Multi-horizon Time-series Forecasting* (2026) already formalizes full/partial/interval abstention, conditional-risk selection, and calibration-set coverage constraints for structured horizons ([arXiv](https://arxiv.org/abs/2602.04714), Sections 2–4).
- *Selective Time Series Forecasting via Metalearning* (2026) already predicts empirical forecast-error percentiles from recent-lag structural features to reject difficult forecasts ([arXiv](https://arxiv.org/abs/2606.23448)).
- PV confidence and conformal forecasting are established separately, including confidence-based separation of solar irradiance forecasts and conformal PV intervals ([IET](https://doi.org/10.1049/iet-rpg.2018.5354), [Solar Energy Advances](https://doi.org/10.1016/j.seja.2024.100059)).
- Time-series/non-exchangeable conformal calibration and general conformal risk control are also established ([NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html), [ICLR 2024](https://openreview.net/forum?id=33XGfHLtZg), [arXiv 2310.01262](https://arxiv.org/abs/2310.01262)).

Thus the defensible difference would have been an application-specific combination and a stringent matched operational audit—not a broad claim of inventing selective multi-horizon forecasting. Since the empirical screen fails the preregistered coverage requirement, that narrow difference is not worth formal model development.

## Protocol self-checks

All **12/12** ordinary checks passed:

1. no deep-model training, optimizer, backward, or gradient update call;
2. risk estimators fit only on `RISK_FIT`;
3. acceptance thresholds computed only on `RISK_CALIBRATION`;
4. Test labels absent from fitting, feature selection, and threshold calculation;
5. every risk feature ends at or before forecast origin;
6. ModernTCN and Persistence share labels, origins, and accepted masks;
7. Test labels/origins are identical across seeds;
8. full and daylight masks are distinct;
9. bootstrap unit is natural day;
10. all outputs remain in the C1 directory;
11. all source artifact sizes and modification times are unchanged;
12. both CSVs are readable and the metric CSV includes fields required to recompute primary summaries.

GPU use was limited to no-gradient checkpoint inference on an NVIDIA GeForce RTX 3060 Laptop GPU. The nine risk estimators ran on CPU. There were no deep-model training runs.

## Reviewer interpretation and route closure

Selective evaluation can manufacture impressive-looking error reductions by rejecting exactly those windows where the forecaster fails. The proper question is not whether accepted RMSE is lower—it almost inevitably is under a useful ranking—but whether coverage is calibrated, operationally stable, and paired with disclosure of rejected cases and a same-sample baseline. Here, ranking is strong, yet the 80% calibration threshold rejects about 41% of daylight Test origins. Reporting only its 50.60% RMSE reduction would materially overstate practical reliability.

The final conclusion is therefore **C1_NO_GO_SIGNAL_WEAK**, specifically a failure of out-of-period coverage transfer under the fixed calibration rule, not absence of oracle headroom or absence of all forecastability signal. C1 should not enter a formal four-part protocol, C1 v2/v3, conformal recalibration variant, or new model development. The unique next recommendation is to close C1 and keep selective-risk findings as an internal limitation analysis; Scheme A and other worktrees remain untouched.

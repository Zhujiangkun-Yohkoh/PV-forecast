# Site 17 Sanyo Ramp Signal Validity and Simple-baseline Challenge

## Reviewer verdict

The existing ramp head contains a real predictive signal and clearly exceeds all four causal simple baselines on ramp-step discrimination. The advantage remains in daylight. However, onset detection is materially weaker than step detection, and full-timeline scores receive a modest benefit from abundant nighttime non-ramp observations. This supports studying joint power-trajectory and ramp-event prediction as a research question, but does not rescue the failed RAMP_AWARE_NCQ interval-width modulation.

## Event and protocol

Ramp step is `1[|P(t+h)-P(t+h-1)| >= 0.1506998 kW]`; h=1 compares against the last observed power at the forecast origin. The threshold is the Train-only 90th percentile. Train step prevalence is 10.0004%; derived onset prevalence is 3.2925%. Onset is diagnostic only and does not replace the original task.

All inputs, Test origins, labels and timestamps come from commits 265cd618 and dd218c7. The four baselines use only information at or before the origin. TIME_OF_DAY_LOGISTIC uses only origin sine/cosine time-of-day and is fit on Train with fixed settings. No Test threshold, calibration, deep training, GPU training, or model selection occurs.

## Ramp-step challenge (all 12 leads pooled)

| Model | Scope | AUROC | AUPRC | F1@0.5 | ECE | Spearman |
|---|---|---:|---:|---:|---:|---:|
| RAMP_AWARE_NCQ_HEAD | full_timeline | 0.9345 +/- 0.0024 | 0.6786 +/- 0.0094 | 0.6652 +/- 0.0108 | 0.1379 +/- 0.0151 | 0.8513 +/- 0.0118 |
| LAST_CHANGE | full_timeline | 0.8832 | 0.6013 | -- | -- | 0.8782 |
| RECENT_MAX_6 | full_timeline | 0.9178 | 0.6551 | -- | -- | 0.8803 |
| IRRADIANCE_CHANGE | full_timeline | 0.8874 | 0.5691 | -- | -- | 0.8791 |
| TIME_OF_DAY_LOGISTIC | full_timeline | 0.8566 | 0.4297 | -- | -- | 0.8233 |
| RAMP_AWARE_NCQ_HEAD | daylight | 0.8416 +/- 0.0055 | 0.6787 +/- 0.0094 | 0.6654 +/- 0.0108 | 0.2655 +/- 0.0294 | 0.4802 +/- 0.0091 |
| LAST_CHANGE | daylight | 0.7337 | 0.6068 | -- | -- | 0.4666 |
| RECENT_MAX_6 | daylight | 0.8123 | 0.6594 | -- | -- | 0.5281 |
| IRRADIANCE_CHANGE | daylight | 0.7424 | 0.5778 | -- | -- | 0.4715 |
| TIME_OF_DAY_LOGISTIC | daylight | 0.6534 | 0.4300 | -- | -- | 0.1513 |
| RAMP_AWARE_NCQ_HEAD | sunrise_sunset_transition | 0.8800 +/- 0.0071 | 0.5151 +/- 0.0206 | 0.5108 +/- 0.0079 | 0.1768 +/- 0.0191 | 0.7113 +/- 0.0344 |
| LAST_CHANGE | sunrise_sunset_transition | 0.7668 | 0.4104 | -- | -- | 0.6831 |
| RECENT_MAX_6 | sunrise_sunset_transition | 0.8157 | 0.4593 | -- | -- | 0.6511 |
| IRRADIANCE_CHANGE | sunrise_sunset_transition | 0.7784 | 0.3592 | -- | -- | 0.6810 |
| TIME_OF_DAY_LOGISTIC | sunrise_sunset_transition | 0.8135 | 0.3560 | -- | -- | 0.7786 |

## Step versus onset

| Task | Scope | Head AUROC | Head AUPRC | Head F1@0.5 | Best baseline AUROC |
|---|---|---:|---:|---:|---:|
| ramp_step | full_timeline | 0.9345 +/- 0.0024 | 0.6786 +/- 0.0094 | 0.6652 +/- 0.0108 | 0.9178 |
| ramp_step | daylight | 0.8416 +/- 0.0055 | 0.6787 +/- 0.0094 | 0.6654 +/- 0.0108 | 0.8123 |
| ramp_step | sunrise_sunset_transition | 0.8800 +/- 0.0071 | 0.5151 +/- 0.0206 | 0.5108 +/- 0.0079 | 0.8157 |
| ramp_onset | full_timeline | 0.8400 +/- 0.0025 | 0.1490 +/- 0.0017 | 0.2560 +/- 0.0047 | 0.8126 |
| ramp_onset | daylight | 0.6726 +/- 0.0051 | 0.1490 +/- 0.0017 | 0.2561 +/- 0.0047 | 0.6324 |
| ramp_onset | sunrise_sunset_transition | 0.8036 +/- 0.0044 | 0.1487 +/- 0.0030 | 0.2452 +/- 0.0029 | 0.7284 |

## Lead-time behavior

| h | Step prevalence | Head AUROC | Head AUPRC | Best baseline AUROC |
|---:|---:|---:|---:|---:|
| 1 | 17.5047% | 0.9510 +/- 0.0014 | 0.7540 +/- 0.0092 | 0.9479 |
| 2 | 17.4990% | 0.9459 +/- 0.0019 | 0.7292 +/- 0.0092 | 0.9385 |
| 3 | 17.4932% | 0.9418 +/- 0.0021 | 0.7075 +/- 0.0096 | 0.9339 |
| 4 | 17.4875% | 0.9391 +/- 0.0023 | 0.6926 +/- 0.0107 | 0.9292 |
| 5 | 17.4818% | 0.9363 +/- 0.0025 | 0.6801 +/- 0.0107 | 0.9246 |
| 6 | 17.4760% | 0.9338 +/- 0.0029 | 0.6700 +/- 0.0118 | 0.9199 |
| 7 | 17.4703% | 0.9317 +/- 0.0028 | 0.6628 +/- 0.0093 | 0.9147 |
| 8 | 17.4645% | 0.9298 +/- 0.0030 | 0.6547 +/- 0.0089 | 0.9102 |
| 9 | 17.4588% | 0.9278 +/- 0.0034 | 0.6462 +/- 0.0082 | 0.9050 |
| 10 | 17.4530% | 0.9267 +/- 0.0032 | 0.6427 +/- 0.0042 | 0.9012 |
| 11 | 17.4473% | 0.9250 +/- 0.0026 | 0.6356 +/- 0.0017 | 0.8967 |
| 12 | 17.4415% | 0.9239 +/- 0.0027 | 0.6312 +/- 0.0029 | 0.8916 |

## Interpretation

1. The head has a clear advantage over the strongest simple baseline for ramp steps, including daylight observations.
2. Onset scores are lower than step scores, showing that the head partly recognizes ongoing ramp regimes rather than exclusively anticipating new events.
3. Full-timeline AUROC is somewhat helped by nighttime negatives; daylight and transition-only results are the more conservative evidence.
4. Discrimination generally declines with lead time, so this is not horizon-invariant event prediction.
5. The signal is sufficient to motivate a separately designed joint trajectory/event study, but not to claim that the existing probabilistic intervals are improved.
6. RAMP_AWARE_NCQ width modulation remains FAIL: classification strength cannot overwrite its worse pinball and Winkler results.

## Limits

No evidence here establishes calibrated uncertainty, cross-site/year generalization, causal ramp mechanisms, operational value, or superiority to trained event-forecasting baselines. Sunrise/sunset transition scope is fixed in advance as ACST 05:00--09:00 and 16:00--20:00, not selected on Test.

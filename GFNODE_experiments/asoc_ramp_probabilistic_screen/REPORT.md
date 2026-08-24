# Site 17 Ramp-aware Probabilistic Trajectory Screen

## Outcome

All 6/6 prescribed GPU runs completed. RAMP_AWARE_NCQ learned a strong ramp-risk signal (H12 AUROC 0.9345 +/- 0.0024; Spearman correlation with absolute future power change 0.8513 +/- 0.0118) and moved aggregate coverage closer to 80%. However, it did **not** improve probabilistic quality: against STANDARD_NCQ it had higher mean pinball loss and worse Winkler score on the full Test set and on the ramp subset. The evidence therefore supports ramp-risk predictability, but not the hypothesis that the implemented risk-conditioned width modulation improves ramp-period probabilistic forecasts.

No Neural ODE, high-frequency dynamics features, automatic gate, Test calibration, hyperparameter search, or subsequent experiment was started.

## Protocol and provenance

- Data preparation, ACST mapping, Train-only preprocessing, feature order, and forecast-origin indices are reused from commit `265cd6183ff374850786df8f30844b8dec043044`.
- Site 17 Sanyo, 2022; Train Jan--Aug, Validation Sep--Oct, Test Nov--Dec; lookback 72 and one direct H12 output.
- Samples: 66,842 Train, 17,485 Validation, 17,401 Test. All three conditions have element-identical Test labels and origin timestamps.
- Inputs are exactly the 14 MEAN_ONLY inputs: historical Active_Power, four cyclic time features, separate MB0/MB1/MB2 five-minute means, valid fractions, and masks. Failed HF_DYNAMICS fields are excluded.
- Ramp threshold is the Train-only 90th percentile, 0.1506998 kW. Train ramp class imbalance gives BCE positive weight 8.9996. Validation/Test do not set either value.
- Both probability models share the same 64-channel, four-block, kernel-5, dropout-0 ModernTCN and three structurally ordered heads. RAMP_AWARE_NCQ alone adds the predicted ramp head and 12-step nonnegative width multipliers. Inference never receives true ramp labels.
- AdamW, learning rate 0.001, weight decay 1e-5, batch 256, maximum 30 epochs, patience 6. Checkpoints use only Validation mean pinball loss. The ramp BCE coefficient is fixed at 0.2.

## Full-Test probability results

Mean +/- sample SD across seeds; intervals are q10--q90.

| Model | Prefix | Pinball | Coverage | Width (kW) | Winkler | Calibration error |
|---|---:|---:|---:|---:|---:|---:|
| STANDARD_NCQ | H3 | 0.059268 +/- 0.002543 | 0.7883 +/- 0.0164 | 0.6141 +/- 0.0973 | 0.8758 +/- 0.0405 | 0.0164 +/- 0.0081 |
| RAMP_AWARE_NCQ | H3 | 0.062851 +/- 0.001638 | 0.8022 +/- 0.0246 | 0.6398 +/- 0.0749 | 0.9359 +/- 0.0261 | 0.0177 +/- 0.0119 |
| STANDARD_NCQ | H6 | 0.064325 +/- 0.001929 | 0.7899 +/- 0.0560 | 0.6607 +/- 0.0918 | 0.9507 +/- 0.0360 | 0.0397 +/- 0.0304 |
| RAMP_AWARE_NCQ | H6 | 0.067112 +/- 0.001601 | 0.8083 +/- 0.0113 | 0.6864 +/- 0.0857 | 0.9990 +/- 0.0269 | 0.0087 +/- 0.0108 |
| STANDARD_NCQ | H12 | 0.071022 +/- 0.002000 | 0.7857 +/- 0.0518 | 0.6985 +/- 0.0893 | 1.0389 +/- 0.0353 | 0.0445 +/- 0.0051 |
| RAMP_AWARE_NCQ | H12 | 0.073871 +/- 0.001528 | 0.8083 +/- 0.0274 | 0.7509 +/- 0.1023 | 1.0938 +/- 0.0293 | 0.0178 +/- 0.0195 |

The non-crossing parameterization worked exactly: crossing rate is 0 in every seed, prefix, and scope.

## H12 results by seed

| Model | Seed | Pinball | Coverage | Width | Winkler | q50 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| STANDARD_NCQ | 42 | 0.072944 | 0.7609 | 0.7794 | 1.0757 | 0.4970 |
| STANDARD_NCQ | 43 | 0.068952 | 0.7508 | 0.6027 | 1.0053 | 0.4986 |
| STANDARD_NCQ | 44 | 0.071170 | 0.8453 | 0.7134 | 1.0359 | 0.4986 |
| RAMP_AWARE_NCQ | 42 | 0.072217 | 0.7991 | 0.6723 | 1.0600 | 0.4981 |
| RAMP_AWARE_NCQ | 43 | 0.075231 | 0.7867 | 0.7140 | 1.1097 | 0.4977 |
| RAMP_AWARE_NCQ | 44 | 0.074165 | 0.8391 | 0.8666 | 1.1118 | 0.4969 |

The apparent coverage benefit is not uniformly accompanied by a proper-score benefit: only seed 42 improves full-Test pinball and Winkler.

## Ramp and non-ramp H12 diagnosis

| Model / subset | Pinball | Coverage | Width (kW) | Winkler |
|---|---:|---:|---:|---:|
| STANDARD / ramp | 0.237665 +/- 0.002255 | 0.7325 +/- 0.0452 | 2.1614 +/- 0.2318 | 3.4188 +/- 0.0509 |
| RAMP_AWARE / ramp | 0.239426 +/- 0.001128 | 0.7503 +/- 0.0498 | 2.2821 +/- 0.3003 | 3.4738 +/- 0.0406 |
| STANDARD / non-ramp | 0.035739 +/- 0.002899 | 0.7969 +/- 0.0609 | 0.3887 +/- 0.0592 | 0.5351 +/- 0.0535 |
| RAMP_AWARE / non-ramp | 0.038819 +/- 0.001987 | 0.8206 +/- 0.0250 | 0.4268 +/- 0.0604 | 0.5899 +/- 0.0344 |

Both models naturally produce much wider ramp intervals than non-ramp intervals. Ramp-aware modulation further increases ramp width by 0.121 kW and coverage by 1.78 percentage points, but worsens ramp pinball by 0.00176 and Winkler by 0.0551. Thus probability quality is not mainly improved in ramps; interval widening trades sharpness for modest coverage.

## Ramp-head quality

| Prefix | AUROC | AUPRC | Brier | Precision | Recall | F1 | Spearman(p, abs change) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H3 | 0.9463 +/- 0.0018 | 0.7311 +/- 0.0095 | 0.1182 +/- 0.0065 | 0.5310 +/- 0.0145 | 0.9524 +/- 0.0069 | 0.6817 +/- 0.0103 | 0.8550 +/- 0.0133 |
| H6 | 0.9413 +/- 0.0022 | 0.7070 +/- 0.0105 | 0.1210 +/- 0.0080 | 0.5253 +/- 0.0160 | 0.9467 +/- 0.0102 | 0.6755 +/- 0.0109 | 0.8534 +/- 0.0130 |
| H12 | 0.9345 +/- 0.0024 | 0.6786 +/- 0.0094 | 0.1250 +/- 0.0093 | 0.5156 +/- 0.0155 | 0.9380 +/- 0.0114 | 0.6652 +/- 0.0108 | 0.8513 +/- 0.0118 |

The fixed 0.5 threshold gives high recall but modest precision. No Test-driven threshold selection was performed.

## Per-step coverage / width

Each cell is mean coverage / mean width in kW across seeds.

| Step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STANDARD | .853/.533 | .683/.634 | .829/.675 | .772/.694 | .840/.702 | .762/.727 | .812/.725 | .858/.725 | .790/.733 | .731/.735 | .715/.752 | .783/.748 |
| RAMP_AWARE | .775/.572 | .807/.659 | .824/.689 | .803/.722 | .822/.717 | .818/.760 | .844/.790 | .823/.803 | .833/.796 | .796/.839 | .771/.797 | .784/.868 |

## q50 point performance and deterministic reference

H12 q50 RMSE is 0.498058 +/- 0.000912 kW for STANDARD and 0.497551 +/- 0.000599 kW for RAMP_AWARE. Their mean R2 values are 0.92559 and 0.92574. Point performance is numerically reasonable, but both are worse than the reused deterministic MEAN_ONLY H12 RMSE of 0.486647 +/- 0.004473 kW: paired degradations are 2.350% and 2.247%, respectively. The probability models are not compared with the deterministic model on coverage or pinball because it has no interval output.

The three deterministic artifacts existed locally and were reused without retraining. Their labels and timestamps match all probability runs element by element.

## Capacity and runtime

| Model | Parameters | Mean training time | Mean epoch time | Inference ms/sample |
|---|---:|---:|---:|---:|
| STANDARD_NCQ | 185,060 | 76.42 s | 3.409 s | 0.0255 |
| RAMP_AWARE_NCQ | 240,392 | 82.38 s | 3.562 s | 0.0274 |

RAMP_AWARE adds 55,332 parameters (+29.90%). Best Validation epochs were STANDARD 16/17/16 and RAMP_AWARE 23/12/16 for seeds 42/43/44.

## Verification and limits

`test_protocol.py` passes 14 ordinary checks covering common artifacts, split/window isolation, causal input timing, structured non-crossing output, output shapes, Train-only ramp threshold/scaling, correct first-step ramp definition, label-free forward modulation, Validation-only checkpointing, exclusion of HF_DYNAMICS inputs, and absence of a Test loader in training. Six probability artifacts and 72 metric rows were independently counted after training.

This screen cannot support claims of improved proper probabilistic scores, superior ramp-period uncertainty quality, calibrated deployment-ready intervals, generalization beyond Site 17/2022, or superiority to external probabilistic baselines. It does support the narrower claim that future ramp occurrence is predictable from the selected six-hour history, while showing that this simple multiplicative width modulation does not convert that signal into consistent probabilistic-score gains.

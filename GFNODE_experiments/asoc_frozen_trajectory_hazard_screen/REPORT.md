# Frozen trajectory + independent first-onset hazard screen

## Outcome

All 3/3 planned GPU runs completed without non-finite loss or gradient. The frozen model reproduced every saved `TRAJECTORY_ONLY` Test prediction exactly: maximum and mean absolute power-output differences were **0.0 kW** for seeds 42, 43 and 44; all power RMSE/MAE differences were also 0.0. The implementation therefore answers the isolation question cleanly.

The event result is mixed and does **not** justify continuing the joint trajectory–event research route. The frozen head reduced the mean miss rate relative to `STEP_MULTITASK` (0.333 vs 0.492), but it did not retain the clear discrimination advantage of the end-to-end hazard model. Its full/daylight AUROC was 0.9420/0.8325, below both `STANDARD_ONSET_HAZARD` (0.9615/0.8875) and `STEP_MULTITASK` (0.9504/0.8551). Its AUPRC was only marginally above STEP (0.8602 vs 0.8555) and below STANDARD (0.8783). Under the stated reviewer rule, this is not a sufficiently clear event advantage over the simpler STEP model. The recommended decision is to terminate this event-model sequence and return to the deterministic `TRAJECTORY_ONLY` route.

## Protocol and execution

- Site/year: Site 17 Sanyo, 2022; capacity 6.3 kW.
- Clock: authoritative high-frequency UTC converted to ACST by UTC+09:30.
- Splits: Train 2022-01-01–2022-08-31; Validation 2022-09-01–2022-10-31; Test 2022-11-01–2022-12-31.
- Lookback/Horizon: 72 × 5 minutes / 12 × 5 minutes.
- Fair origins: Train 66,842; Validation 17,485; Test 17,401.
- Ramp threshold: **0.1506997943 kW**, the Train-only 90th percentile of absolute power change. It was not re-estimated on Validation or Test.
- Optimizer: AdamW, learning rate 0.001, weight decay 1e-5; batch 256; at most 25 epochs; patience 5; gradient clipping 1.0; no mixed precision; zero workers; no scheduler.
- Selection: Validation first-event NLL only. The training function has no Test-loader argument.
- Hazard head: one `Linear(4608, 36)`, reshaped to `[batch,12,3]`, identical in structure and size to the existing hazard head.
- Freezing: backbone and power-head parameters had `requires_grad=False`, were absent from the optimizer, stayed in evaluation mode, received no gradients, and were compared elementwise before/after training. There are no backbone buffers in this architecture; the generic buffer comparison also passed.

Run details:

| Seed | Best epoch | Epochs | Stop | Best Validation NLL | Train seconds | Mean epoch seconds |
|---:|---:|---:|---|---:|---:|---:|
| 42 | 6 | 11 | early stopping | 1.038512 | 48.41 | 4.394 |
| 43 | 14 | 19 | early stopping | 1.032335 | 82.10 | 4.315 |
| 44 | 12 | 17 | early stopping | 1.028745 | 78.70 | 4.623 |

No run showed numerical divergence or a non-finite gradient. All **24/24** ordinary protocol tests passed after training.

## Power trajectory identity

The two rows below apply identically to `TRAJECTORY_ONLY` and `FROZEN_TRAJECTORY_HAZARD` (mean ± sample SD across seeds).

| Scope | Horizon | RMSE kW | MAE kW |
|---|---:|---:|---:|
| Full | H3 | 0.40758 ± 0.00466 | 0.19289 ± 0.00449 |
| Full | H6 | 0.44467 ± 0.00406 | 0.20964 ± 0.00285 |
| Full | H12 | 0.48665 ± 0.00448 | 0.23344 ± 0.00190 |
| Daylight | H12 | 0.67678 ± 0.00602 | 0.42330 ± 0.00346 |
| Ramp-step | H12 | 0.97087 ± 0.00943 | 0.72639 ± 0.00873 |
| First-onset-near | H12 | 0.80823 ± 0.00833 | 0.56375 ± 0.00502 |

The H12 trajectory first-difference MAE was 0.13249 ± 0.00130 kW for both. Full Test array checks found maximum absolute difference = 0.0 kW, mean absolute difference = 0.0 kW, RMSE difference = 0.0 kW and MAE difference = 0.0 kW for every seed.

## First-onset results

Mean ± sample SD across seeds:

| Model | Scope | AUROC | AUPRC | Brier | F1 | Recall | Miss rate | Time MAE (steps) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| STEP_MULTITASK | Full | 0.9504±0.0038 | 0.8555±0.0085 | 0.1916±0.0102 | 0.5478±0.0049 | 0.5080±0.0142 | 0.4920±0.0142 | 2.5109±0.2279 |
| STANDARD_ONSET_HAZARD | Full | 0.9615±0.0041 | 0.8783±0.0089 | 0.0721±0.0033 | 0.8407±0.0126 | 0.8309±0.0381 | 0.1691±0.0381 | 2.8051±0.1559 |
| POWER_ANCHORED_HAZARD | Full | 0.9614±0.0035 | 0.8798±0.0127 | 0.0709±0.0036 | 0.8481±0.0136 | 0.8501±0.0357 | 0.1499±0.0357 | 2.8491±0.1375 |
| FROZEN_TRAJECTORY_HAZARD | Full | 0.9420±0.0110 | 0.8602±0.0237 | 0.1001±0.0060 | 0.7505±0.0067 | 0.6673±0.0056 | 0.3327±0.0056 | 2.7712±0.0674 |
| STEP_MULTITASK | Daylight | 0.8551±0.0110 | 0.8555±0.0085 | 0.3419±0.0162 | 0.5479±0.0049 | 0.5080±0.0142 | 0.4920±0.0142 | 2.5109±0.2279 |
| STANDARD_ONSET_HAZARD | Daylight | 0.8875±0.0121 | 0.8783±0.0090 | 0.1298±0.0060 | 0.8407±0.0127 | 0.8309±0.0381 | 0.1691±0.0381 | 2.8051±0.1559 |
| POWER_ANCHORED_HAZARD | Daylight | 0.8875±0.0101 | 0.8798±0.0126 | 0.1275±0.0065 | 0.8481±0.0137 | 0.8501±0.0357 | 0.1499±0.0357 | 2.8491±0.1375 |
| FROZEN_TRAJECTORY_HAZARD | Daylight | 0.8325±0.0303 | 0.8607±0.0231 | 0.1801±0.0107 | 0.7505±0.0067 | 0.6673±0.0056 | 0.3327±0.0056 | 2.7712±0.0674 |

For Frozen versus Standard, full AUROC changed by −0.01948 (−2.03%) and AUPRC by −0.01810 (−2.06%); daylight AUROC changed by −0.05506 (−6.20%) and AUPRC by −0.01763 (−2.01%). Versus STEP, Frozen full AUROC changed by −0.00842, while AUPRC changed by only +0.00469; daylight AUROC changed by −0.02266 and AUPRC by +0.00521. The miss-rate improvement over STEP is real (−0.1593 absolute), but discrimination is not consistently superior.

Frozen direction results were upward recall 0.4401±0.0250, downward recall 0.4182±0.0484 and direction accuracy 0.6449±0.0071. Neither direction collapsed completely, but both recalls were below Standard's mean recalls. Onset-time MAE was 2.771±0.067 steps (13.86 minutes) among jointly detected positive windows: slightly below Standard, but worse than STEP and still too imprecise for a claim of accurate onset timing. Per-seed and H1–H12 lead-time rows are retained in `metrics_per_seed.csv` and should not be overinterpreted where counts are small.

The common Test population was 17,401 windows with 5,623 first-onset positives (32.31%). Daylight included 9,655 windows and all 5,623 positives (58.24% prevalence); sunrise/sunset included 7,134 windows and 2,742 positives (38.44%). The near-identical Full/Daylight AUPRC values arise because all Test onset positives satisfy the target-power daylight mask, although the extra nighttime negatives materially increase Full AUROC. The masks are independently constructed and their sample counts differ.

## Efficiency and provenance

- Backbone: 19,136 parameters; frozen power head: 55,308; hazard head: 165,924.
- Total inference parameters: 240,368; trainable parameters: 165,924 (69.03%). Frozen parameters remain present at inference.
- Mean inference latency: 0.0356 ms/sample (three-run mean on the current GPU environment).
- Mean training time: 69.74 s/run; mean epoch time: 4.444 s. This is cheaper in optimization state and backward computation than Standard Hazard because only the head is trainable, although the large flattened linear head still dominates parameter count.
- Every reused artifact matched the same Test labels and forecast-origin timestamps elementwise. Inputs end at the forecast origin; no future PV or irradiance is used.
- Local checkpoints, logs and arrays are under `results/FROZEN_TRAJECTORY_HAZARD/<seed>/` and are intentionally excluded from Git.

## Reviewer judgment

The frozen deterministic representation contains useful first-onset information, but not enough to preserve the end-to-end hazard model's event advantage. Power fidelity is perfect by construction, yet event AUROC degrades below STEP on both Full and Daylight scopes, and AUPRC is only marginally higher with substantial seed variation. The method can be described as a low-interference diagnostic risk head, but the evidence is insufficient to define a publishable two-stage trajectory-plus-event method or to justify full baselines, ablation, or cross-time validation.

**Recommendation: terminate the joint trajectory–event research direction and retain `TRAJECTORY_ONLY` as the candidate deterministic forecasting route.** Do not add another event-head version, Adapter, extra encoder, or new loss. The previous `RAMP_AWARE_NCQ` width-modulation result remains FAIL; no claim about probabilistic intervals, cross-site transfer, cross-year robustness, climate generalization, or deployment is supported.

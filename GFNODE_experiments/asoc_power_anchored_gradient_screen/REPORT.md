# Site 17 Power-Anchored Hazard Gradient Screen

## Final reviewer verdict

All 3/3 prescribed POWER_ANCHORED_HAZARD GPU runs completed without numerical failure. Negative shared-backbone gradient conflict is real and persistent, occurring in roughly 38%--42% of batches. Asymmetric projection slightly improves power RMSE while preserving first-onset discrimination, but the recovery is far too small: H12 RMSE remains 3.16% worse than TRAJECTORY_ONLY and H3 remains 7.04% worse. Onset-time MAE also remains worse than STEP_MULTITASK.

The power-anchored projection therefore does **not** solve the core joint-training problem. Per the predeclared final-rescue rule, terminate this shared joint-training route. Do not proceed to GradNorm, MGDA, dynamic weighting, adapters, shared-private encoders, further loss variants, NODE, full baselines, ablation, or paper redesign.

RAMP_AWARE_NCQ interval-width modulation remains FAIL.

## Fixed protocol

- Site 17 Sanyo, 2022, UTC-derived ACST UTC+09:30; 5-minute grid; lookback 72; direct H12 output.
- Windows: Train 66,842; Validation 17,485; Test 17,401. Train/Validation/Test dates and Train-only preprocessing are unchanged from commits 265cd618 and 94ef5ad.
- Train-only ramp threshold 0.1506998 kW. First-onset and upward/downward labels, masks, origins and timestamps are unchanged.
- ModernTCN channels 64, four blocks, kernel 5, dropout 0; power head and hazard head exactly match STANDARD_ONSET_HAZARD.
- AdamW, LR 0.001, weight decay 1e-5, batch 256, no scheduler, max 25 epochs, patience 5, min_delta 1e-8, clip norm 1.0, no mixed precision, num_workers 0.
- Loss remains standardized power MSE + 0.2 cause-specific first-event NLL. Validation checkpoint objective is unchanged.
- Only shared-backbone gradient merging changes: power gradients remain untouched; conflicting event gradients are projected; non-conflicting event gradients remain unchanged. Heads receive only their own task gradients.

## Gradient-conflict evidence

| Seed | Epochs | Mean conflict rate | First 3 epochs | Last 3 epochs | Raw cosine mean | Projection/event norm ratio | Mean power/event gradient norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 39.50% | 40.46% | 37.02% | 0.0640 | 7.13% | 0.30 / 0.26 approximately |
| 43 | 11 | 41.67% | 43.26% | 38.55% | 0.0368 | 5.28% | 0.27 / 0.24 approximately |
| 44 | 10 | 38.17% | 39.06% | 36.51% | 0.0505 | 5.18% | 0.27 / 0.24 approximately |
| Mean +/- SD | -- | 39.78% +/- 1.78% | 40.92% +/- 2.11% | 37.36% +/- 1.08% | 0.0504 +/- 0.0136 | 5.86% +/- 1.11% | -- |

Conflict is not restricted to initialization: it persists through the final epochs, although it declines modestly. Raw cosine is positive on average because non-conflicting batches predominate slightly; negative batches have materially negative medians (stored per epoch in JSONL). The projection correction is only about 5%--7% of event-gradient norm, consistent with the small observed power recovery. Conflict exists, but direct negative alignment is not the dominant explanation for the full trajectory degradation.

## Power trajectory by seed

| Model | Seed | H3 RMSE | H6 RMSE | H12 RMSE | H12 MAE |
|---|---:|---:|---:|---:|---:|
| TRAJECTORY_ONLY | 42 | .40308 | .44002 | .48177 | .23125 |
| TRAJECTORY_ONLY | 43 | .41238 | .44752 | .48762 | .23452 |
| TRAJECTORY_ONLY | 44 | .40728 | .44645 | .49055 | .23455 |
| STANDARD_HAZARD | 42 | .43426 | .46297 | .49979 | .26823 |
| STANDARD_HAZARD | 43 | .43679 | .47072 | .50706 | .26916 |
| STANDARD_HAZARD | 44 | .44455 | .46766 | .50346 | .26716 |
| POWER_ANCHORED | 42 | .42648 | .45775 | .49617 | .26562 |
| POWER_ANCHORED | 43 | .44225 | .47126 | .50774 | .28676 |
| POWER_ANCHORED | 44 | .44009 | .46515 | .50215 | .26172 |

## Mean power results and recovery

| Model | H3 RMSE | H6 RMSE | H12 RMSE | H12 MAE |
|---|---:|---:|---:|---:|
| TRAJECTORY_ONLY | .40758 +/- .00466 | .44467 +/- .00406 | .48665 +/- .00447 | .23344 +/- .00190 |
| STANDARD_HAZARD | .43853 +/- .00536 | .46712 +/- .00390 | .50344 +/- .00363 | .26819 +/- .00100 |
| POWER_ANCHORED | .43627 +/- .00855 | .46472 +/- .00676 | .50202 +/- .00579 | .27136 +/- .01347 |

Against STANDARD_HAZARD, projection improves RMSE by 0.00226 kW (0.52%) at H3, 0.00240 kW (0.51%) at H6, and 0.00142 kW (0.28%) at H12. The improvement is not seed-stable at H3/H6: seed 43 worsens. Against TRAJECTORY_ONLY, anchored RMSE remains worse by 0.02869 kW (7.04%) at H3, 0.02006 kW (4.51%) at H6 and 0.01537 kW (3.16%) at H12.

H12 diagnostic RMSE for TRAJECTORY_ONLY / STANDARD / ANCHORED is:

- Daylight: .67678 / .69679 / .69389 kW.
- Ramp-step: .97087 / .97198 / .96988 kW.
- First-onset windows: .80823 / .82526 / .82103 kW.
- First-difference MAE: .13249 / .13654 / approximately .13567 kW.

Projection gives small diagnostic improvements but does not close the general trajectory gap.

## First-onset results by seed

| Seed | Scope | Anchored AUROC | Anchored AUPRC | Brier | F1 | Miss rate | Time MAE steps |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | Full | .9654 | .8944 | .0679 | .8465 | .1711 | 2.857 |
| 43 | Full | .9599 | .8725 | .0698 | .8624 | .1087 | 2.708 |
| 44 | Full | .9590 | .8725 | .0748 | .8353 | .1698 | 2.983 |
| 42 | Daylight | .8990 | .8944 | .1223 | .8465 | .1711 | 2.857 |
| 43 | Daylight | .8831 | .8726 | .1255 | .8626 | .1087 | 2.708 |
| 44 | Daylight | .8803 | .8725 | .1348 | .8353 | .1698 | 2.983 |

## Event preservation relative to prior models

| Metric | STEP_MULTITASK | STANDARD_HAZARD | POWER_ANCHORED |
|---|---:|---:|---:|
| Full AUROC | .9504 +/- .0038 | .9615 +/- .0041 | .9614 +/- .0035 |
| Full AUPRC | .8555 +/- .0085 | .8783 +/- .0089 | .8798 +/- .0127 |
| Daylight AUROC | .8551 +/- .0110 | .8875 +/- .0121 | .8875 +/- .0101 |
| Daylight AUPRC | .8555 +/- .0085 | .8783 +/- .0090 | .8798 +/- .0126 |
| Miss rate | .4920 +/- .0142 | .1691 +/- .0381 | .1499 +/- .0357 |
| Onset-time MAE steps | 2.511 +/- .228 | 2.805 +/- .156 | 2.849 +/- .138 |
| Up recall | .4274 +/- .0325 | .5546 +/- .0832 | .5845 +/- .0817 |
| Down recall | .2675 +/- .0058 | .5124 +/- .0187 | .5155 +/- .0086 |
| Direction accuracy | .7015 +/- .0268 | .6445 +/- .0196 | .6509 +/- .0212 |

Hazard discrimination, low miss rate and both directions are preserved; no single-class failure occurs. Onset-time MAE remains worse than STEP and slightly worsens relative to STANDARD_HAZARD.

## Full versus Daylight scope audit

The masks are genuinely different:

- Full: 17,401 windows, 5,623 first-onset positives, prevalence 32.31%.
- Daylight: 9,655 windows, 5,623 positives, prevalence 58.24%.
- Masks are not equal; 7,746 Full windows are excluded by Daylight.
- Daylight is defined diagnostically as **any future true H12 target power > 0.063 kW**, not by the origin clock. It is an evaluation mask, never an input.

All first-onset positives satisfy this Daylight definition, so Full and Daylight contain the same positive events but different negatives. Nearly identical AUPRC is therefore not an array reuse bug: the removed nighttime negatives mostly receive scores below the positives and have almost no effect on precision-recall ranking. AUROC changes substantially because its negative-pair comparisons do change. Unified reevaluation of all four models confirms separate arrays and masks.

## Runs, parameters and compute

| Seed | Best epoch | Actual epochs | Stop | Finite | Training seconds |
|---:|---:|---:|---|---|---:|
| 42 | 7 | 12 | early stopping | yes | 94.60 |
| 43 | 6 | 11 | early stopping | yes | 84.48 |
| 44 | 5 | 10 | early stopping | yes | 76.91 |

POWER_ANCHORED and STANDARD_HAZARD both contain 240,368 parameters: backbone 19,136, power head 55,308, hazard head 165,924. Projection adds no parameters and changes no inference graph. Mean anchored epoch time is 7.73 s versus 6.57 s standard, a 17.7% training overhead from separate gradient extraction/projection. Measured inference is 0.0350 versus 0.0392 ms/sample; this small reversed difference is timing noise, not an inference benefit.

## Verification

`test_protocol.py` passes 19 ordinary checks covering identical parameterization/initialization; independent task gradients; conflict and non-conflict projection behavior; unchanged power gradient; isolated head gradients; exact shared merge; no accumulation; finite event/no-event loss and gradients; no Test loader; Validation-only checkpoints; common artifacts; separate Full/Daylight masks; Train-only threshold; and forward/backward execution.

All four models use element-identical Test labels, timestamps, origins and stored masks. Three new anchored artifacts and 345 compact metric rows were independently recomputed. No Test threshold search, calibration, model adjustment, or extra training was performed.

## Final recommendation

Terminate the joint shared-backbone training route. Gradient conflict is measurable, and projection preserves event quality, but it recovers only a small fraction of lost power accuracy at nontrivial training cost while onset timing remains worse. The next research decision should return to the already stronger deterministic trajectory model rather than create another multitask-gradient or architecture variant.

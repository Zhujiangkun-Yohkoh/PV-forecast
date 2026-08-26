# Scheme C1-S1 Formal Five-Stage Protocol

## Status and scope

This document preregisters the formal protocol only. It does not fit a base forecaster or risk estimator, compute candidate Final-Test C1 errors, download data, or authorize GPU execution. The formal route may start only after the data conditions below have been verified without outcome inspection.

The fixed task is causal selective forecasting of a direct H12 Site-level PV trajectory on a regular 5-minute clock. Each input contains 72 observations ending at the forecast origin; H3, H6, and H12 are prefixes of the same H12 output. Risk features may use only information available at or before the origin.

## Final-Test eligibility audit

### What was inspected

The audit inspected only file existence, timestamp range and order, five-minute regularity, column structure, value missingness, timezone evidence, and the ability to align the three arrays with MB0/MB1/MB2. It did not load C1 predictions for 2023/2024, calculate their errors, fit risk models, calculate risk scores, set thresholds, or construct risk–coverage curves.

The Git history for `asoc_selective_trajectory_feasibility` contains C1-S0/S0R commits `3e825d5`, `085c5bd`, `3280b9f`, `ce9c249`, `50f7e16`, and `a20cf58`. Their task data are Site 17 in 2022. Searches across experiment reports found structural audits of 2023 and sealed-test declarations in the NWP work, but no 2023/2024 C1 risk score, acceptance threshold, realized coverage, selective RMSE/MAE, risk–coverage curve, hyperparameter choice, or decision. Therefore 2023 is **C1-method-unseen**, not globally untouched.

### PV evidence

The three authoritative five-minute files span the full calendar endpoints in 2023, with zero duplicate or inverted parsed timestamps. They contain gaps and missing power, so windows must be built separately inside continuous valid segments.

| Array | 2023 timestamps | valid Active Power | missing timestamps vs calendar | valid-power segments | L72+H144 diagnostic windows |
|---|---:|---:|---:|---:|---:|
| Site 17 Sanyo | 101,797 | 96,941 | 3,323 | 2,773 | 72,047 |
| Site 25 Hanwha | 101,833 | 95,908 | 3,287 | 2,642 | 71,456 |
| Site 38 Q CELLS | 101,812 | 95,981 | 3,308 | 2,639 | 71,609 |
| Common valid timestamps | 91,296 | — | — | 4,284 | 69,875 |

The L72+H144 counts above are conservative continuity diagnostics from the earlier long-horizon convention; the formal H12 task will have at least as many eligible windows after the same segmented construction. No prediction outcome was used to select a segment.

### Irradiance evidence and eligibility verdict

The validated 2022 second-level file is structurally complete in UTC (31,536,000 records; 3,011,901,400 bytes), with channel-level NaNs retained by masks. The local file labelled as 2023 second-level irradiance is not a full 2023 resource: it contributes only 560 five-minute bins from 2023-01-01 09:05 to 2023-01-03 07:40, followed by a malformed transition and an approximately 446-day jump. It cannot support a Final Test. The 2024 portions of that damaged export are also discontinuous and are not an eligible fallback.

**Candidate Final Test:** calendar 2023 remains the only preregistered candidate, conditional on a fresh authoritative full-year UTC second-level MB0/MB1/MB2 export passing the same structural and time-alignment checks as the 2022 redownload. It must be described as `C1-method-unseen candidate Final Test`, never as globally untouched.

## Fixed five-stage time protocol

The only authorized chronology is below. All boundaries are defined on the raw ACST PV clock after mapping authoritative irradiance UTC to ACST (`UTC+09:30`). A later implementation must record exact first/last eligible origins after segmented window construction, but may not move these calendar boundaries.

| Order | Stage | Raw-time boundary (ACST) | Permitted use |
|---:|---|---|---|
| A | `BASE_TRAIN` | 2021-01-01 00:00 through 2021-12-31 23:55 | Fit all preprocessing and the base forecaster only |
| B | `BASE_MODEL_VALIDATION` | 2022-01-01 00:00 through 2022-04-30 23:55 | Early stopping and base-checkpoint selection only |
| C | `RISK_FIT` | 2022-05-01 00:00 through 2022-08-31 23:55 | Fit the fixed risk estimator only |
| D | `RISK_CALIBRATION` | 2022-09-01 00:00 through 2022-12-31 23:55 | Set fixed scope-matched acceptance thresholds only |
| E | `FINAL_TEST` | 2023-01-01 00:00 through 2023-12-31 23:55 | One final evaluation after code, rules, metrics, and success criteria are frozen |

This is a **five-stage** protocol, not a four-part protocol. The chronology is strictly A < B < C < D < E. Each stage independently identifies continuous valid segments and constructs lookback=72/H12 windows inside them. A window may not cross a missing interval, segment boundary, or stage boundary. Separate disjoint segments are never joined by interpolation or reindex-based concealment.

All feature scalers, target scalers, fill values, missingness transformations, and any other fitted preprocessing are fitted on `BASE_TRAIN` only. The base checkpoint is selected using `BASE_MODEL_VALIDATION` only. `BASE_MODEL_VALIDATION` cannot enter risk fitting. The risk estimator uses `RISK_FIT` only. Scope-matched thresholds use `RISK_CALIBRATION` only. No `FINAL_TEST` object may be passed to a fit, checkpoint, feature-selection, prior-estimation, or threshold function.

The same eligible origins, labels, validity masks, and accepted masks must be used by the base model and Last-value Persistence within every comparison. Final Test is not opened until the implementation, fixed feature set, configuration, metrics, bootstrap, and success rule are reviewed and frozen in ordinary version control.

## Model and experiment matrix

The existing source class is a compact stack of an input 1×1 convolution, repeated depthwise temporal convolutions plus 1×1 convolutions and GELU, and a flattened linear H12 output. It does not implement the full official ModernTCN architecture. Its formal name is therefore:

`DEPTHWISE_TCN_TRAJECTORY`

No new base forecaster or deep baseline is authorized. If all three arrays pass the same five-stage data protocol, the formal deep-training matrix is exactly three arrays × seeds 42, 43, and 44 = **9 runs**. H3/H6/H12 are evaluated from each single H12 trajectory.

The risk estimator is fixed to the current `HistGradientBoostingRegressor` configuration. The three risk methods are fixed:

- `FULL_RISK_MODEL`
- `RECENT_VARIATION`
- `MODEL_PERSISTENCE_DISAGREEMENT`

Last-value Persistence is not fitted and is evaluated only on the identical accepted origins. There is no hyperparameter search, C1 v2/v3, probabilistic interval, NODE, dynamic threshold, or new risk model.

## Origin-daylight and thresholds

The primary scope is `origin-daylight`: observed Active Power at the forecast origin exceeds 1% of the verified array-specific reference. The intended references are Site 17 = 6.3 kW, Site 25 = 5.83 kW, and Site 38 = 5.9 kW, subject to final metadata/unit verification before execution. Corresponding thresholds are 0.0630, 0.0583, and 0.0590 kW. This is not a claim that the entire future H12 trajectory is daylight.

The nominal 80% operating threshold is a finite-sample order statistic computed separately for each array, seed, horizon, risk method, and scope from `RISK_CALIBRATION` only. Test scores never modify it. No distribution-free, finite-sample, or conformal guarantee is claimed.

## Formal evaluation

For every array, seed, and H3/H6/H12 prefix, report:

- realized origin-daylight coverage;
- accepted and rejected RMSE and MAE;
- unselected RMSE and MAE;
- risk–coverage AUC (AURC; lower is better);
- matched-Persistence RMSE/MAE and skill on identical accepted origins;
- Spearman correlation between risk score and fixed trajectory loss;
- AUROC and AUPRC with the high-error prevalence and the `RISK_FIT`-defined high-error threshold;
- sample, accepted, rejected, and high-error counts.

Array-level results are primary. Macro summaries are secondary and may not conceal failure of an array.

AURC is the primary incremental comparison between `FULL_RISK_MODEL` and simple risk scores. If two methods' realized Test coverage differs by more than two percentage points, their accepted-RMSE percentages must not be used as a direct method-superiority claim. Their AURC and a coverage-matched descriptive interpolation, if reported, must be clearly distinguished from the fixed operating point.

The primary uncertainty analysis is a **continuous seven-day moving-block bootstrap**, 1,000 replicates. Blocks are sampled from the complete chronological origin sequence, preserving within-block overlap and dependence; acceptance masks and thresholds are fixed before resampling. Every replicate recomputes unselected, accepted, rejected, and matched-Persistence metrics. Natural-day cluster bootstrap is retained only as sensitivity analysis because overlapping H12 windows cross day boundaries and adjacent days are dependent.

## Formal ordinary-test design (not implemented in S1)

Before any GPU authorization, the implementation must pass ordinary tests for:

1. **Future sentinel:** replacing every power and irradiance value after an origin leaves that origin's risk features element-identical.
2. **Threshold isolation:** replacing every Final-Test score leaves the `RISK_CALIBRATION` threshold element-identical.
3. `BASE_MODEL_VALIDATION` never enters risk fitting.
4. `RISK_FIT` never enters threshold calibration.
5. No Final-Test loader/object is accepted by any fit, checkpoint-selection, or threshold API.
6. `risk_fit_fraction` genuinely controls the split or is removed from configuration; the formal calendar boundary remains authoritative.
7. The high-change quantile is read from configuration and estimated only from the authorized fit stage.
8. Each array uses its independently verified 1% reference threshold.
9. Base model, Persistence, and every risk method use element-identical origins, labels, and accepted masks for a stated comparison.
10. All features, risks, predictions, losses, and metrics are finite.
11. Every input-source timestamp is no later than the forecast origin.
12. Every window lies wholly within one continuous segment and one stage.

## Preregistered Final-Test decision boundary

Success requires all of the following at the fixed nominal 80% origin-daylight operating point:

1. macro realized coverage in [0.75, 0.85], with no array below 0.70;
2. macro AURC improvement of `FULL_RISK_MODEL` over the best simple risk score of at least 5%;
3. the AURC improvement direction holds for at least two of three arrays;
4. at least two of three arrays outperform matched Persistence on accepted origins;
5. accepted RMSE is at least 10% below unselected RMSE and the improvement direction holds for at least two of three seeds;
6. every failure is reported without a C1 v2/v3 response.

Only if every condition passes may manuscript writing and necessary ablation be considered. Otherwise C1 closes and Scheme A remains the fallback. These are preregistered reporting boundaries, not a software release gate.

## Execution authorization

The protocol design is authorized, but GPU execution is **`NOT_AUTHORIZED_UNTIL_DATA_AND_PROTOCOL_CONFIRMED`**. The blocking evidence is the missing full-year 2021 second-level source needed for consistent base training and the corrupt/incomplete local 2023 second-level source needed for Final Test. After the two authoritative exports are supplied, they must pass read-only structure, UTC/ACST, channel, missingness, and segmented-alignment checks before any training begins.

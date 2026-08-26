# Scheme A submission-critical correction report

## Decision

**GO_FOR_CORRECTED_BENCHMARK_APPLICATION_MANUSCRIPT; NO-GO for an algorithm-superiority manuscript.** The corrected experiment supports a leakage-aware application benchmark, but not a claim that any evaluated architecture is new or uniformly superior. Last-value persistence is beaten in most, but not all, combinations; supplementary daily persistence remains stronger than the best neural result in 22 of 24 array--horizon--scope combinations on its slightly different eligible sample set.

## Corrections implemented

- Fixed the seven original numerical inputs, in order: `Performance_Ratio`, `Weather_Temperature_Celsius`, `Weather_Relative_Humidity`, `Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`, `Radiation_Global_Tilted`, and `Radiation_Diffuse_Tilted`.
- Confirmed that the previous 15-channel tensor comprised seven numerical values, their seven missing-value masks, and one Isolation Forest indicator. It contained neither cyclical time features nor historical `Active_Power`.
- Added causal historical `Active_Power` and its missing-value mask. The corrected tensor has eight numerical values, eight masks, and one Isolation Forest indicator (17 channels). Future power is never an input.
- Fit the Active Power imputer/scaler, all other imputers/scalers, and Isolation Forest on Train only.
- Replaced equal averaging of batch validation means with global masked MSE: accumulated target SSE divided by accumulated valid-target count.
- Kept the original date splits, lookback 72, H144 output, seeds 42/43/44, AdamW settings, epoch budget, and Validation-only checkpoint selection. The training API has no Test loader argument.
- Replaced misleading family names with implementation-level descriptions: **Discrete recurrent decoder**, **Inverted-variate Transformer**, **Joint-patch Transformer**, and **Depthwise convolutional TCN**. The published iTransformer, PatchTST, and ModernTCN papers remain architectural inspirations, not implementation-identity claims.
- Made horizon-specific valid origins the primary analysis. H12/H48/H96/H144 require L72 and only the evaluated target prefix to be valid, although every neural model still outputs H144. Full-H144-common prefixes are retained as a secondary sensitivity analysis.
- Constructed Last-value Persistence on exactly the same origins, labels, and masks as each neural comparison. Daily Persistence remains supplementary because its 24-hour lag changes eligible counts.

## Execution and tests

- Real GPU runs: **36/36 completed** (4 models x 3 arrays x 3 seeds).
- Numerical divergence/non-finite outputs: **none**.
- Ordinary protocol tests: **15/15 passed**.
- Device: NVIDIA GeForce RTX 3060 Laptop GPU; float32.
- Local `results/` contains checkpoints, predictions, labels, origins, masks, and epoch logs and is intentionally excluded from Git.

## Primary common sample counts

Counts are identical for every neural model and Last-value Persistence within each array/horizon. `valid_target_count` is for the full timeline; daylight counts use true target power above 1% of the Train maximum and are evaluation-only.

| Array | Horizon | Origins | Valid targets | Daylight origins | Daylight targets |
|---|---:|---:|---:|---:|---:|
| Sanyo | H12 | 6,589 | 79,068 | 3,201 | 36,190 |
| Sanyo | H48 | 5,905 | 283,440 | 3,389 | 129,528 |
| Sanyo | H96 | 5,024 | 482,304 | 3,629 | 215,712 |
| Sanyo | H144 | 4,160 | 599,040 | 3,869 | 259,345 |
| Hanwha | H12 | 6,625 | 79,500 | 3,225 | 36,421 |
| Hanwha | H48 | 6,049 | 290,352 | 3,522 | 132,649 |
| Hanwha | H96 | 5,281 | 506,976 | 3,881 | 228,420 |
| Hanwha | H144 | 4,521 | 651,024 | 4,225 | 287,368 |
| Qcells | H12 | 6,463 | 77,556 | 3,173 | 36,504 |
| Qcells | H48 | 5,491 | 263,568 | 3,071 | 123,447 |
| Qcells | H96 | 4,227 | 405,792 | 2,927 | 184,959 |
| Qcells | H144 | 2,996 | 431,424 | 2,800 | 187,950 |

## Qcells H12 correction

The old evidence used 3,019 neural origins but 2,996 Last-value Persistence origins, so the comparison was not sample matched. It also contained only 42 daylight target points (0.116% of the old 36,228 neural target points), producing an unrepresentative near-zero full-timeline persistence error.

The corrected primary comparison has 6,463 common origins and 77,556 common target points. It contains 3,173 daylight origins and 36,504 daylight target points: **49.095% of origins** and **47.067% of target points**, respectively. Corrected Qcells H12 full-timeline RMSE is 0.471 kW for Last-value Persistence versus 0.327 kW for the Inverted-variate Transformer and 0.416 kW for the Depthwise convolutional TCN. The Discrete recurrent decoder (0.923 kW) and Joint-patch Transformer (0.532 kW) remain worse than Last-value Persistence. The previous 0.004 kW persistence result is withdrawn.

## Corrected ranking and persistence results

Across the 24 primary array x horizon x scope combinations, including Last-value Persistence:

| Method | RMSE wins | Mean rank | Arithmetic mean RMSE skill vs Last | Positive-skill combinations |
|---|---:|---:|---:|---:|
| Inverted-variate Transformer | 12 | 1.875 | 0.549 | 24/24 |
| Depthwise convolutional TCN | 9 | 2.167 | 0.515 | 24/24 |
| Joint-patch Transformer | 2 | 2.458 | 0.493 | 22/24 |
| Discrete recurrent decoder | 1 | 3.667 | 0.399 | 22/24 |
| Last-value Persistence | 0 | 4.833 | 0 | reference |

The arithmetic mean of per-combination ratio-based skill is not the same quantity as the mean absolute or Train-range-normalized error. Mean Train-range nRMSE across the 24 combinations is 0.135 for the Inverted-variate Transformer, 0.143 for the Joint-patch Transformer, 0.144 for the Depthwise convolutional TCN, 0.161 for the Discrete recurrent decoder, and 0.329 for Last-value Persistence.

Daily Persistence is a stringent supplementary challenge. Because the exact 24-hour lag slightly changes sample availability, it is not ranked in the primary table. On its eligible sample set, it beats the best neural result in 22/24 combinations; only Hanwha H12 full and daylight favor the Depthwise convolutional TCN. This result must remain prominent in the paper.

Restricting all prefixes to complete H144 windows lowers the macro RMSE of every method (for example, 0.808 kW versus 0.771 kW for the Inverted-variate Transformer in primary versus sensitivity summaries). The sensitivity therefore demonstrates sample-selection effects and cannot remain the sole primary analysis.

## Efficiency

| Method | Parameters | Mean batch-1 latency (ms) | Throughput (samples/s) | Mean run training time (s) |
|---|---:|---:|---:|---:|
| Discrete recurrent decoder | 99,362 | 31.023 | 1,116 | 563.88 |
| Inverted-variate Transformer | 194,960 | 0.496 | 377,131 | 23.66 |
| Joint-patch Transformer | 148,112 | 0.499 | 421,929 | 18.80 |
| Depthwise convolutional TCN | 683,024 | 0.670 | 197,426 | 24.24 |

These are descriptive measurements on one GPU and exclude loading and disk I/O.

## Disposition of previous conclusions

### Retained

- Train-only preprocessing, split-local windows, Validation-only checkpoint selection, explicit masks, and persistence challenges are necessary.
- Rankings depend on array, forecast horizon, and evaluation scope.
- This work is an evaluation/application study, not a new architecture contribution or cross-site generalization study.
- Daily seasonal persistence is materially stronger than Last-value Persistence and must not be hidden.

### Modified

- Neural models do outperform Last-value Persistence on average after causal history power is supplied and samples are matched; this is expressed separately as absolute error and ratio-based skill.
- The strongest learned implementation is the Inverted-variate Transformer in this corrected benchmark, not the prior model labeled ModernTCN.
- Test is described only as excluded from preprocessing fitting, training, and checkpoint selection; it is not described as untouched or evaluated once at project level.

### Withdrawn

- The old 18/24 neural-win claim for “ModernTCN.”
- The conclusion that no neural model beats Last-value Persistence on average.
- The old Qcells H12 full/daylight results and any ranking based on their unmatched and tiny daylight sample.
- Claims that the three compact implementations are complete official iTransformer, PatchTST, or ModernTCN reproductions.

## Reviewer-style judgment

The corrected evidence is sufficient for a transparent benchmark/application manuscript if the paper foregrounds sample fairness, the causal power-history correction, horizon-specific eligibility, and both persistence references. It is not sufficient for a new-algorithm paper or a universal neural-superiority claim. The strongest publishable finding is conditional: learned H144 trajectory models consistently beat Last-value Persistence after correction, yet Daily Persistence remains the dominant supplementary comparator in 22/24 settings. A paper that reports both facts is defensible; a paper that suppresses the second is not.

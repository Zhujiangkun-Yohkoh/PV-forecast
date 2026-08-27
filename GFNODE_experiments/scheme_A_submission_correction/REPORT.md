# Scheme A submission-final evidence audit

## Final decision

**SCHEME_A_EVIDENCE_READY_FOR_SUBMISSION_PACKAGE.** This means that the numerical evidence and manuscript descriptions are internally consistent enough to prepare a submission package. It does not authorize filling unconfirmed authorship, CRediT, funding, indexing, or submission-system declarations.

No neural-network training, optimizer step, backward pass, epoch loop, checkpoint modification, or Test-driven model change was performed in this audit.

## Artifact and code audit

- Completed runs found: **36/36**.
- Required files found per run: `completed.json`, `test_H144.npz`, and `best_validation.pt`.
- Saved labels, `target_valid`, forecast origins, and target starts agree elementwise with the current 17-channel protocol for all 36 runs.
- Every checkpoint loads strictly into its expected 17-input model, and its model, array, seed, parameter count, best epoch, and Validation global MSE agree with `completed.json`.
- All 36 best checkpoints reproduced their saved predictions by no-gradient GPU inference within the declared floating-point tolerance.
- Lookback 72, output H144, Train/Validation/Test dates, and H144 Validation selection agree with the active configuration.
- `run_one()` now validates all identity, array, shape, checkpoint, and protocol fields before reuse. A mismatch raises `STALE_ARTIFACT`; evidence-only execution never falls back to training.
- Protected raw PV files, checkpoints, prediction arrays, and completion markers retained their original sizes and modification times.

## Tests

- Ordinary tests not requiring local results: **16/16 passed**.
- Full tests requiring all 36 local artifacts: **9/9 passed**.
- Skipped tests: **0**.
- The test suite itself uses inference only and does not execute backward or an optimizer.

## Primary horizon-specific evidence

All models are trained and selected on complete H144 Train/Validation windows. Horizon-specific eligibility is used only for the primary Test prefixes. Last-value Persistence and every neural method use identical origins, labels, and point masks in each primary comparison.

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

Qcells H12 remains corrected at 6,463 common origins and 77,556 full-timeline targets. Its 36,504 daylight targets are 47.067% of full target points. Full/daylight Last-value RMSE is 0.471/0.682 kW; Inverted-variate is 0.327/0.457 kW; Depthwise TCN is 0.416/0.599 kW.

The independently verified 24-combination primary ranking remains:

| Method | RMSE wins | Mean rank | Arithmetic mean RMSE skill vs Last |
|---|---:|---:|---:|
| Inverted-variate Transformer | 12 | 1.875 | 0.549 |
| Depthwise convolutional TCN | 9 | 2.167 | 0.515 |
| Joint-patch Transformer | 2 | 2.458 | 0.493 |
| Discrete recurrent decoder | 1 | 3.667 | 0.399 |
| Last-value Persistence | 0 | 4.833 | 0 (reference) |

The arithmetic mean of ratio-based skill is not the same statistic as macro mean RMSE or macro Train-range nRMSE.

## Daily Persistence: corrected same-mask comparison

The previous statement used Daily Persistence on its own valid points while neural values came from a different mask. It is retained only as correction history and is not evidence for a win count.

The new analysis is named `supplementary_daily_matched`. For every array, horizon, and scope, the actual finite Daily-Persistence point mask is intersected with label and daylight validity and then applied unchanged to Daily Persistence, Last-value Persistence, and all 12 neural predictions. Thus every method has the same origins, labels, target points, and daylight points within a comparison.

| Array | H | Scope | Origins | Points | Daily RMSE | Best neural (RMSE) | Winner |
|---|---:|---|---:|---:|---:|---|---|
| Sanyo | 12 | Full | 6,589 | 79,005 | 0.175 | Discrete recurrent (0.207) | Daily |
| Sanyo | 12 | Daylight | 3,201 | 36,190 | 0.259 | Depthwise TCN (0.291) | Daily |
| Sanyo | 48 | Full | 5,905 | 283,197 | 0.160 | Joint-patch (0.615) | Daily |
| Sanyo | 48 | Daylight | 3,389 | 129,528 | 0.236 | Joint-patch (0.864) | Daily |
| Sanyo | 96 | Full | 5,024 | 481,821 | 0.129 | Inverted-variate (0.826) | Daily |
| Sanyo | 96 | Daylight | 3,629 | 215,712 | 0.193 | Inverted-variate (1.134) | Daily |
| Sanyo | 144 | Full | 4,160 | 598,317 | 0.118 | Inverted-variate (0.775) | Daily |
| Sanyo | 144 | Daylight | 3,869 | 259,345 | 0.179 | Inverted-variate (1.057) | Daily |
| Hanwha | 12 | Full | 6,625 | 79,392 | 0.185 | Depthwise TCN (0.176) | Neural |
| Hanwha | 12 | Daylight | 3,225 | 36,421 | 0.273 | Depthwise TCN (0.231) | Neural |
| Hanwha | 48 | Full | 6,049 | 289,920 | 0.193 | Inverted-variate (0.575) | Daily |
| Hanwha | 48 | Daylight | 3,522 | 132,649 | 0.286 | Inverted-variate (0.817) | Daily |
| Hanwha | 96 | Full | 5,281 | 506,112 | 0.204 | Depthwise TCN (0.804) | Daily |
| Hanwha | 96 | Daylight | 3,881 | 228,420 | 0.303 | Depthwise TCN (1.136) | Daily |
| Hanwha | 144 | Full | 4,521 | 649,735 | 0.203 | Depthwise TCN (0.743) | Daily |
| Hanwha | 144 | Daylight | 4,225 | 287,368 | 0.305 | Depthwise TCN (1.039) | Daily |
| Qcells | 12 | Full | 6,463 | 77,530 | 0.281 | Inverted-variate (0.327) | Daily |
| Qcells | 12 | Daylight | 3,173 | 36,504 | 0.409 | Inverted-variate (0.457) | Daily |
| Qcells | 48 | Full | 5,491 | 263,470 | 0.281 | Inverted-variate (0.808) | Daily |
| Qcells | 48 | Daylight | 3,071 | 123,447 | 0.410 | Inverted-variate (1.160) | Daily |
| Qcells | 96 | Full | 4,227 | 405,598 | 0.255 | Inverted-variate (1.137) | Daily |
| Qcells | 96 | Daylight | 2,927 | 184,959 | 0.378 | Inverted-variate (1.647) | Daily |
| Qcells | 144 | Full | 2,996 | 431,150 | 0.244 | Depthwise TCN (0.789) | Daily |
| Qcells | 144 | Daylight | 2,800 | 187,950 | 0.369 | Depthwise TCN (1.152) | Daily |

**Fair result:** Daily Persistence outperforms the best neural implementation in **22/24 matched comparisons**; the best neural implementation wins **2/24**, both at Hanwha H12. The numerical count is unchanged from the historical statement, but its evidentiary basis is now valid because every method uses the same point mask. `corrected_metrics.csv` additionally records MAE, Train-range nRMSE, per-seed RMSE skill relative to Daily Persistence, three-seed means, and sample SD for every matched combination.

## Efficiency source correction

Efficiency rows are read directly from the 36 `completed.json` files. Mean values are:

| Method | Parameters | Mean latency (ms) | Throughput (samples/s) | Peak memory (MiB) |
|---|---:|---:|---:|---:|
| Discrete recurrent decoder | 99,362 | 32.204 | 1,116 | 121.12 |
| Inverted-variate Transformer | 194,960 | 0.558 | 377,131 | 27.31 |
| Joint-patch Transformer | 148,112 | 0.535 | 421,929 | 26.47 |
| Depthwise convolutional TCN | 683,024 | 0.706 | 197,426 | 35.27 |

The manuscript table was updated to these artifact-derived values. The prior rounded latency values are withdrawn. A separate NumPy/Pandas implementation subsequently verified all submission-critical metric rows without importing the production evaluation functions.

## Independent submission evidence verification

`independent_verify_evidence.py` directly read the 36 saved NPZ files, 36 run metadata files, the three source Active Power series, and `corrected_metrics.csv`. It independently rebuilt the horizon-specific and Daily-matched masks, Last-value and exact-lag Daily Persistence, RMSE, MAE, Train-range nRMSE, rankings, and skills. It did not import or call the production metric or mask functions.

- Comparisons: **4,414/4,414 passed**.
- Maximum absolute difference: **8.51e-12**.
- Maximum relative difference: **3.23e-11**.
- Primary win counts: 12 Inverted-variate, 9 Depthwise TCN, 2 Joint-patch, 1 Discrete recurrent, and 0 Last-value.
- Daily-matched outcome: **22/24 Daily Persistence**, with the two neural wins still limited to Hanwha H12 full/daylight.
- Qcells H12: 6,463 origins, 77,556 full targets, and 36,504 daylight targets.
- Training, optimizer, backward, and checkpoint loading: **not executed**.
- The source data, checkpoints, and prediction artifacts retained their exact sizes and nanosecond modification times.

## Manuscript disposition

- **Retained:** title, primary ranking, Last-value Persistence skills, Qcells H12 correction, and application/benchmark positioning.
- **Strengthened:** Daily Persistence is now a same-mask result rather than an unmatched contextual comparison.
- **Corrected:** Methods explicitly state complete-H144 Train/Validation checkpoint selection and horizon-specific Test-only eligibility; regular-grid missing rows, masks, and Train-only imputation are explicit; efficiency values come directly from run metadata.
- **Withdrawn:** any inference based on different Daily and neural point masks, including the former wording “on its own eligible sample set.”
- **Unchanged boundary:** the four neural models are compact project implementations, not official full reproductions or new architectures; Test has been repeatedly inspected during development and is not an independent external confirmation.

## Reviewer-style judgment

The evidence is internally coherent for a transparent benchmark/application submission package. The result is scientifically challenging rather than favorable: the learned models beat Last-value Persistence, but Daily Persistence wins 22/24 strictly matched supplementary comparisons. This must remain prominent in the abstract, results, discussion, limitations, and conclusion. The manuscript is not evidence for algorithmic novelty, universal neural superiority, or cross-site generalization.

No new training is justified or authorized by this audit. Remaining work is human submission metadata and editorial positioning, not numerical repair.

## Final independent verification and submission package

**SCHEME_A_SUBMISSION_PACKAGE_READY.** The submission package was prepared without neural-network training or checkpoint modification.

- Original ordinary protocol tests: **16/16 passed**.
- Completed-artifact tests: **9/9 passed**.
- New independent-evidence tests: **12/12 passed**.
- Skipped tests: **0**.
- Independent numerical comparisons: **4,414/4,414 passed**.
- PDF: **12 pages**, all reported fonts embedded; no missing citations, undefined references, overfull boxes, or visual page-boundary failures were found.
- Table IV memory unit: corrected from MB to **MiB** with values unchanged, matching the `bytes / 1024**2` implementation.
- Official JRSE/AIP requirements were checked on **2026-08-27**. JRSE remains a hybrid journal; its official charge page states no page charges and lists optional Author Select open access at USD 3,800. Current Clarivate indexing must still be confirmed by the author or library.
- The package intentionally does not assert author approval, exclusive submission, CRediT roles, funding, Conflict of Interest, ethics applicability, ORCID, reviewer nominations, or fee authorization.

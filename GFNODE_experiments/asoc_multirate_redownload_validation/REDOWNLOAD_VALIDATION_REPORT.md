# DKASC 2022 Multirate Redownload Validation

## Decision

**B. 2022_PARTIAL_PERIOD_USABLE**

The redownloaded second-level file is structurally intact and contains every UTC second of calendar year 2022. It is not `2022_FULL_YEAR_USABLE` for strict alignment to the timezone-free Site 17 PV clock because:

1. the exported `Local` field is consistently UTC+09:00, whereas official DKASC notices identify Alice Springs event time as ACST and civil ACST is UTC+09:30;
2. physical phase and independent MB-channel correlation measurements show that the PV clock aligns with UTC+09:30, not the exported Local field;
3. a UTC-calendar-2022 export cannot supply the first 9.5 hours of local-calendar 2022 after that correction;
4. 5,909 of 105,120 second-derived 5-minute intervals contain at least one missing MB value, although no timestamp itself is missing.

The data are sufficiently sound to enter a carefully masked, segmented multirate information-gain screen. They are not suitable for an unqualified “complete local 2022” claim.

## Input-location note

The newly downloaded files were found in the requested validation directory, not in the stated `原始Dataset/高分辨率气象数据集` directory. They were treated as pre-existing, read-only inputs and were not moved, renamed, copied, or modified:

- `fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2022.csv`
- corresponding 2022 five-minute irradiance, meteorology, and wind-vector CSVs.

The older high-resolution directory still contains the previously audited corrupted 2023 file. This path discrepancy is recorded so that later code does not accidentally select the old file.

## File structure

### Second-level irradiance

| Item | Result |
|---|---:|
| File size | 3,011,901,400 bytes |
| Physical lines | 31,536,001 including header |
| Parseable data records | 31,536,000 |
| First/last UTC | 2022-01-01 00:00:00 / 2022-12-31 23:59:59 |
| First/last exported Local | 2022-01-01 09:00:00 / 2023-01-01 08:59:59 |
| Dominant interval | 1 second; all 31,535,999 adjacent differences |
| Missing timestamps | 0 |
| Duplicate / inverse timestamps | 0 / 0 |
| Column-count / quote anomalies | 0 / 0 |
| Glued records / `Data Error` | 0 / 0 |
| MB0 / MB1 / MB2 missing values | 3,992 / 3,236 / 3,320 |
| UTC–Local offset | +09:00 for all 31,536,000 records |

Thus the redownload is not corrupted at the file-structure or network-transfer level. Value-level NaNs remain and are counted independently from structural integrity.

### Five-minute companion exports

The three companion CSVs each contain 105,121 parseable records, no duplicate/inverse timestamps, no structural anomalies, and a dominant five-minute interval. Their UTC range is 2021-12-31 15:00 through 2022-12-31 15:00 and exported Local range is 2022-01-01 00:00 through 2023-01-01 00:00, again with an invariant +09:00 offset. The five-minute irradiance file has one missing value in each MB channel; the meteorology and wind files have no MB fields and no structural errors.

## Calendar-month coverage

Every month contains exactly its expected number of second timestamps and has a longest timestamp gap of zero seconds.

| Month | Expected/actual seconds | MB0 valid | MB1 valid | MB2 valid | All three valid | Complete / partial / empty 5-min bins | Longest complete run |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jan | 2,678,400 / 2,678,400 | 2,678,400 | 2,678,400 | 2,678,398 | 2,678,398 | 8,926 / 2 / 0 | 21,740 min |
| Feb | 2,419,200 / 2,419,200 | 2,419,200 | 2,419,199 | 2,419,199 | 2,419,198 | 8,062 / 2 / 0 | 20,925 min |
| Mar | 2,678,400 / 2,678,400 | 2,677,947 | 2,677,947 | 2,677,948 | 2,677,947 | 8,925 / 3 / 0 | 34,940 min |
| Apr | 2,592,000 / 2,592,000 | 2,592,000 | 2,592,000 | 2,592,000 | 2,592,000 | 8,640 / 0 / 0 | 43,200 min |
| May | 2,678,400 / 2,678,400 | 2,678,399 | 2,678,399 | 2,678,399 | 2,678,399 | 8,927 / 1 / 0 | 38,155 min |
| Jun | 2,592,000 / 2,592,000 | 2,592,000 | 2,592,000 | 2,592,000 | 2,592,000 | 8,640 / 0 / 0 | 43,200 min |
| Jul | 2,678,400 / 2,678,400 | 2,678,400 | 2,678,398 | 2,678,398 | 2,678,398 | 8,926 / 2 / 0 | 17,890 min |
| Aug | 2,678,400 / 2,678,400 | 2,678,383 | 2,678,383 | 2,678,381 | 2,678,363 | 8,895 / 33 / 0 | 21,935 min |
| Sep | 2,592,000 / 2,592,000 | 2,591,218 | 2,591,374 | 2,591,337 | 2,590,507 | 7,372 / 1,268 / 0 | 1,455 min |
| Oct | 2,678,400 / 2,678,400 | 2,677,342 | 2,677,610 | 2,677,584 | 2,676,336 | 7,151 / 1,777 / 0 | 255 min |
| Nov | 2,592,000 / 2,592,000 | 2,591,253 | 2,591,408 | 2,591,412 | 2,590,548 | 7,349 / 1,291 / 0 | 205 min |
| Dec | 2,678,400 / 2,678,400 | 2,677,466 | 2,677,646 | 2,677,624 | 2,676,624 | 7,398 / 1,530 / 0 | 385 min |

Annual totals are 31,536,000/31,536,000 timestamps, 31,528,718 jointly valid channel-seconds, 99,211 fully valid 5-minute bins, 5,909 partial bins, and zero timestamp-empty bins. April and June are completely valid at the three-channel/second level; September–December contain the densest value missingness.

## Independent MB-channel characterization

No channel is averaged before feature calculation. Each 5-minute interval independently yields mean, standard deviation, minimum, maximum, range, first-last change, maximum absolute first difference, least-squares slope, and valid count.

Selected annual distributions across five-minute bins:

| Channel | Mean irradiance mean / min / max | Mean within-bin std | Mean range | Mean max abs difference | Mean valid count |
|---|---:|---:|---:|---:|---:|
| MB0 | 248.62 / 0 / 1399.09 W/m² | 12.74 | 45.59 | 7.73 | 299.962 |
| MB1 | 268.32 / 0 / 1296.06 W/m² | 13.22 | 47.41 | 7.83 | 299.969 |
| MB2 | 269.73 / 0 / 1074.59 W/m² | 16.27 | 56.92 | 10.03 | 299.968 |

Channel relationships at one-second resolution:

| Pair | Pearson correlation | Difference mean ± SD | Difference min / max |
|---|---:|---:|---:|
| MB0−MB1 | 0.9835 | −19.70 ± 70.31 W/m² | −965 / 872 |
| MB0−MB2 | 0.8952 | −21.12 ± 174.09 W/m² | −983 / 1,228 |
| MB1−MB2 | 0.9133 | −1.42 ± 159.95 W/m² | −880 / 1,105 |

There are zero seconds where at least two channels fall outside the broad physical audit range [−20, 1600] W/m². A descriptive cross-channel disagreement rule—exactly one channel more than `max(100 W/m², 25% of the per-second median)` from the median—flags 6,180,385 seconds. These are disagreement flags, not automatically sensor faults: the three MB instruments can have different orientations/responses, so no values are removed on this basis.

All three channels are usable with their own validity masks. MB0 and MB1 are most mutually consistent; MB2 remains scientifically useful as an independent channel rather than being averaged away.

## Time-basis resolution

The file itself proves only that its exported Local column equals UTC+09:00. It does not prove that this is Alice Springs civil time. Available DKASC official notices use **ACST** for Alice Springs events, and the Northern Territory uses ACST (UTC+09:30) without daylight saving. Relevant official pages:

- https://dkasolarcentre.com.au/download/notes-on-the-data
- https://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase
- https://dkasolarcentre.com.au/glossary

Four candidates were measured against Site 17 without using a model or selecting by forecast performance. Correlation lags were measured independently for MB0, MB1 and MB2.

| Candidate | Applied clock offset | MB0 / MB1 / MB2 peak lag | Peak correlations | Median irradiance sunrise vs PV start | Median irradiance noon vs PV peak |
|---|---:|---:|---:|---:|---:|
| PV timestamps vs high-frequency UTC | 0 | −24 / −24 / −24 (search boundary) | negative | irradiance near 00:00 vs PV 07:00 | 03:10/03:05/03:10 vs 12:35 |
| UTC+09:00 | +540 min | −6 / −6 / −6 | 0.9835 / 0.9985 / 0.9084 | 06:30/06:30/06:45 vs 07:00 | 12:10/12:05/12:05 vs 12:35 |
| UTC+09:30 | +570 min | **0 / 0 / 0** | 0.9835 / 0.9985 / 0.9084 | 07:00/07:00/07:15 vs 07:00 | 12:40/12:35/12:35 vs 12:35 |
| File Local | +540 min | −6 / −6 / −6 | same as +09:00 | same as +09:00 | same as +09:00 |

Recommendation: treat the PV timestamps as ACST and map high-frequency data from authoritative UTC using **UTC+09:30**. Do not use the exported Local column without an explicit 30-minute correction. This recommendation is based first on official civil-time semantics and then corroborated by sunrise/noon phase and zero-lag results—not selected merely because correlation is high.

No DST jump is present or expected under ACST.

## Site 17 usable observations and windows

Under the recommended UTC+09:30 mapping:

- Site 17 PV timestamps in 2022: 102,901;
- valid `Active_Power`: 102,890;
- mapped three-channel-complete high-frequency bins within the PV calendar: 99,122;
- PV-valid and high-frequency-complete common bins: 96,892;
- daylight common-complete bins: 44,388;
- contiguous lookback=72, H12 windows: **67,400**;
- no counted window crosses a missing/incomplete interval.

The longest strict common-complete sequence is **2022-03-26 20:30 through 2022-04-26 21:00** on the recommended PV/ACST clock, lasting 44,675 minutes. This is the recommended single continuous validation period. For a broader information-gain screen, all 67,400 segmented windows across 2022 may be used with splits defined before any outcome inspection.

## Readiness and limitations

The data can proceed to a minimal multirate information-gain performance screen, provided that:

- UTC is the authoritative source clock and is converted explicitly to ACST;
- MB0/MB1/MB2 remain separate with channel-specific masks;
- only observations at or before each forecast origin are used;
- windows are segmented at every PV absence, invalid target, or incomplete required high-frequency interval;
- Test data do not influence time mapping, feature selection, or thresholds.

Recommended first screen: Site 17 Sanyo, the strict 2022-03-26 20:30–2022-04-26 21:00 interval for a continuous prototype, all three MB channels plus their independent past-only within-bin statistics, lookback 72, and H12 one-hour prediction. Full-year segmented screening is defensible after the prototype passes.

Unsupported claims at this stage include: complete local-calendar-2022 coverage; validity of the exported Local field as civil time; universal sensor-anomaly labels from channel disagreement; multi-year/seasonal generalization; performance gain from multirate information; continuous-time superiority; arbitrary-resolution forecasting; or operational availability of future irradiance.

## Ordinary self-checks

The validation asserts that:

- sizes and nanosecond modification times of all four resource inputs and Sanyo are unchanged before/after the full scan;
- generated outputs are restricted to this directory;
- monthly expected seconds sum to 31,536,000 and complete + partial + empty bins reproduce each calendar month;
- MB0, MB1 and MB2 have independent counters and features; all four time candidates contain three independent channel lag measurements;
- malformed timestamps return UNKNOWN/`None` behavior;
- truncated rows are never padded or assigned guessed values;
- each second-derived feature bin ends after its maximum source timestamp, preventing future-input leakage.

No hash, contract, freeze, baseline, automated gate, interpolation, repaired data copy, model, or training tensor was created.

## Git constraint

The requested output directory is outside the only existing Git repository (`PVforecast16`). `PV_improve_v1` and its parents are not Git worktrees and have no `origin`. Committing these four files to `origin/master` would require either copying them into another directory, initializing/modifying a repository, or changing the requested output location—all explicitly outside the authorized boundary. The four files are therefore complete locally, but Git commit/push is blocked pending a user decision on the repository path.

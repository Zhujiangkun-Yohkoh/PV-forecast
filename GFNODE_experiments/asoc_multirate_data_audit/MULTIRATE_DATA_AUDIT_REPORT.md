# DKASC Multirate Data Feasibility Audit

## Executive decision

For the intended 2021–2023 multirate study, the current download supports **C: re-download of the high-resolution irradiance data is required**. The four authoritative PV exports remain usable for regular 5-minute research (**B**) after explicit malformed-record handling, but the file labelled `irradiancedata_1sec_2023.csv` contains only 560 local 5-minute bins in 2023 (2023-01-01 09:05 to 2023-01-03 07:40), followed by a malformed transition and an approximately 446-day jump to March 2024. It cannot support a substantive 2021–2023 continuous/multirate experiment.

No neural network was trained. No source row was changed, skipped silently, interpolated, inserted, deleted, or written back.

## Files and parseability

Eight real CSV files were scanned line-by-line. Detailed headers, byte sizes, physical line counts, timestamp distributions and per-column missingness are in `DATA_FILE_INVENTORY.csv`.

| File | Physical lines | Valid/recovered records | Fully parseable | Structural findings |
|---|---:|---:|---|---|
| `17 Sanyo.csv` | 1,767,913 | 1,767,912 | No | 1 glued line containing two complete records; 1 unrecoverable truncated tail (`"2025-08-`); 1 odd-quote line |
| `23 Calyxo.csv` | 1,669,412 | 1,669,411 | No | 2 glued lines; each has a truncated first record and a complete recoverable suffix |
| `25 Hanwha.csv` | 999,916 | 999,915 | No | 2 glued lines; includes the known 2025-08-23 05:30 truncated record followed by a complete 05:35 record |
| `38 Q CELLS.csv` | 1,621,061 | 1,621,061 | No | 2 glued lines; 1 truncated record fragment; 2 complete suffix/records recoverable |
| 5-minute irradiance resource CSV | 549,579 | 549,578 | Yes | No structural error detected |
| `irradiancedata_1sec_2023.csv` | 18,657,728 | 18,657,519 | No | 2 glued/truncated transitions with 2 complete suffix records recoverable; 208 literal `Data Error`/column-count-invalid lines unrecoverable |
| 5-minute meteorology CSV | 549,579 | 549,578 | Yes | No structural error detected |
| 5-minute wind-vector CSV | 549,579 | 549,578 | Yes | No structural error detected |

Across all files there are **218 structurally abnormal physical lines**: 9 glued lines, one additional truncated Sanyo tail, and 208 unrecoverable error lines. They contain **8 truncated record fragments**, **9 complete records recoverable in memory**, and **209 standalone unrecoverable physical records** (208 high-frequency error lines plus the Sanyo tail). There are **0 duplicate headers, 0 duplicate timestamps, and 0 timestamp inversions** among parsed records.

These categories overlap by design: a glued physical line can contain both one MALFORMED fragment and one recovered complete record. Truncated values are never guessed. Natural empty cells are counted separately in `missing_counts` and `missing_ratios`; they are not classified as CSV damage.

### Known Hanwha case

At physical line 948,028, the first record begins at `2025-08-23 05:30:00` and stops after the first part of the irradiance fields. The complete `2025-08-23 05:35:00` record is concatenated immediately afterward. The audit marks 05:30 as MALFORMED and accepts only the independently field-complete 05:35 suffix in memory. Nothing is written back.

### Structural damage relevance to 2021–2023

The PV glued/truncated problems identified with timestamps occur in 2025, outside the target years; the PV exports nevertheless have ordinary value missingness and timestamp gaps during 2021–2023, which are retained explicitly. The decisive problem is the high-resolution irradiance export: its 2023 run terminates during 2023-01-03, at a corrupted transition resembling `02/01/2023 22:24/03/2024 00:00:00...`. The first record is not recoverable; the complete March 2024 suffix is recoverable. Thus almost all of 2023 and all of 2021–2022 lack second-level irradiance in the supplied high-resolution file.

## Time bases, rates and units

### PV exports

- Timestamp format: `YYYY-MM-DD hh:mm:ss`; no timezone or UTC offset is encoded.
- Dominant interval: exactly 5 minutes.
- No daylight-saving jump is visible as a duplicated/inverted timestamp. Because no timezone is encoded, timezone semantics cannot be independently proven from these files.
- `Active_Power` maps to DKASC's **5-Min-Avg kW** definition: AC power in kW based on 1-second values averaged over 5 minutes.
- `Active_Energy_Delivered_Received` is a cumulative delivered-minus-received energy quantity in kWh, not instantaneous power.
- horizontal and tilted radiation fields are W/m² and are 10-second samples averaged over 5 minutes according to the DKASC glossary.

### Resource exports

- The large irradiance file explicitly labels UTC and Local timestamps at second resolution and irradiance as `[W/m-2]`. Its dominant parsed interval is **1 second** (18,657,303 adjacent one-second differences), not 5 seconds.
- The three smaller resource files have minute-format timestamps and a dominant interval of **5 minutes**. They are already aggregates and are not high-frequency substitutes.
- The downloaded resource files consistently encode Local = UTC +09:00. No offset change or DST transition was detected. This differs from the conventional Alice Springs civil offset; the audit reports the file values as written and does not silently correct them.
- The three irradiance channels `Irradiance_MB0/MB1/MB2` explicitly use W/m²; unit temperatures use °C and voltages use V.
- None of the eight downloaded files contains a `System availability` field. Therefore official logger availability cannot be compared programmatically with observed missingness in this download.

Authoritative definitions:

- DKASC glossary: https://dkasolarcentre.com.au/glossary
- DKASC data notes: https://dkasolarcentre.com.au/download/notes-on-the-data
- Alice Springs download page: https://dkasolarcentre.com.au/download?location=alice-springs

The glossary states that irradiance is power per area in W/m², AC power is 5-minute-average kW, delivered-minus-received energy is cumulative kWh, and System availability represents onsite data-logger connectivity.

## Array identity

All four requested PV exports are locally present; none is `NOT_DOWNLOADED`.

| Site | Official identity | Rating | Local file |
|---|---|---:|---|
| 17 | Sanyo, HIT hybrid silicon, fixed | 6.3 kW | present |
| 23 | Calyxo, CdTe, fixed | 5.4 kW | present |
| 25 | Hanwha Solar, poly-Si, fixed | 5.83 kW | present |
| 38 | Hanwha Q CELLS, mono-Si, fixed | 5.9 kW | present |

Official source pages:

- Site 17: https://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase
- Site 23: https://dkasolarcentre.com.au/source/alice-springs/dka-m15-a-phase
- Site 25: https://dkasolarcentre.com.au/source/alice-springs/dkasc-alice-springs-25-hanwha-q-cells-poly-si-fixed
- Site 38: https://dkasolarcentre.com.au/source/alice-springs/dka-m19-b-phase

## 2021–2023 coverage

Each calendar year contains 105,120 expected 5-minute timestamps. PV source-row/value counts are:

| Array | 2021 rows / valid | 2022 rows / valid | 2023 rows / valid |
|---|---:|---:|---:|
| Sanyo | 100,539 / 100,534 | 102,901 / 102,890 | 101,797 / 96,941 |
| Calyxo | 100,539 / 99,074 | 102,901 / 100,501 | 101,810 / 95,794 |
| Hanwha | 100,541 / 99,076 | 102,901 / 100,501 | 101,833 / 95,908 |
| Q CELLS | 100,539 / 99,074 | 102,901 / 100,501 | 101,812 / 95,981 |

For all arrays, **2022 is the best of the three years** by both timestamp presence and valid-power count. Sanyo has the highest valid-power coverage in every year and is the best minimal-prototype array.

The second-level file contributes no 5-minute bins in 2021 or 2022 and only 560 bins in 2023; 544 contain exactly 300 valid second-level irradiance observations and 16 are partial. This is 0.53% row coverage and 0.52% complete-bin coverage of calendar year 2023.

The true PV/second-level common local range for every array is **2023-01-01 09:05 through 2023-01-03 07:40**, 560 five-minute intervals. Within that narrow range:

- PV row/value valid: 560/560 for each array;
- high-resolution row present: 560/560;
- complete 300-observation intervals: 544 (97.14%);
- partial intervals: 16 (2.86%);
- completely empty intervals inside this short bounded range: 0;
- PV-valid but high-resolution-incomplete: 16;
- high-resolution-valid but PV-invalid: 0;
- longest continuous complete segment: 925 minutes (15 h 25 min).

Outside that bounded run, the interval distribution exposes an approximately 446-day discontinuity into March 2024. It must not be hidden by limiting coverage calculations to observed rows.

Day/night coverage is reported separately in the summary using a fixed, descriptive rule: past-interval mean irradiance >20 W/m² is daylight. Intervals with no high-resolution observations are not assigned to either class. System-availability agreement is `CANNOT_VERIFY` because the field is absent.

## Past-only high-frequency information screen

For each forecast origin `t`, irradiance features use only records in `(t-5 min, t]`: mean, standard deviation, minimum, maximum, range, first-last change, maximum absolute first difference, slope, coefficient of variation, and valid count. Targets are the next 5-minute ramp and the change from `t` to `t+1 hour`. No observation after `t` is used as an input.

Across the 560 usable origins, Spearman correlations between past-window features and absolute future-one-hour ramp were:

| Array | 5-min mean | Standard deviation | Range |
|---|---:|---:|---:|
| Sanyo | 0.835 | 0.845 | 0.847 |
| Calyxo | 0.844 | 0.855 | 0.857 |
| Hanwha | 0.828 | 0.840 | 0.842 |
| Q CELLS | 0.822 | 0.836 | 0.838 |

After linearly controlling for the 5-minute mean, range retains only a modest Pearson association with absolute one-hour ramp (`r=0.136–0.142`); standard deviation retains `r=0.110–0.116`. Ramp-group summaries also show higher within-bin range/std in the ramp group than the moderate group, and near-zero values in the stable/night-heavy group.

Interpretation: **there is a preliminary indication of information beyond the 5-minute mean, but not decisive evidence**. The incremental relationship is modest, the sample is only 560 origins spanning fewer than two days, solar level strongly confounds both mean and volatility, and no seasonal/weather diversity exists. This screen justifies checking a corrected download; it does not justify a multirate modeling claim.

## Leakage boundary

The legitimate main task is to use high-resolution observations timestamped no later than the forecast origin to predict future PV power. Using measured irradiance from within the future prediction interval would be an oracle/ex-post task and cannot be presented as operational forecasting. The audit code enforces and self-tests `max_source_time <= forecast_origin` for every constructed feature aggregate.

## Answers to the required questions

1. **Can every file be parsed completely?** No. Three 5-minute resource files are structurally clean; all four PV exports and the second-level irradiance file contain at least one structural problem.
2. **How many problems?** 218 abnormal physical lines, including 9 glued lines, 8 truncated record fragments, 209 standalone unrecoverable records, 9 recovered complete records, and 0 duplicate timestamps. Categories overlap as explained above.
3. **Does structural damage affect 2021–2023 research?** PV structural incidents are outside 2021–2023, but the second-level file's corrupted transition and 446-day gap make the intended period unusable.
4. **True common range?** 2023-01-01 09:05 to 2023-01-03 07:40 local-column time.
5. **Actual high-frequency rate?** 1 second. The smaller resource exports are 5-minute aggregates.
6. **Which arrays have sufficient coverage?** All four are adequate candidates for regular 5-minute work; Sanyo is strongest. None has sufficient paired second-level coverage for a substantive multirate study with this download.
7. **Best year?** 2022 for PV quality. For second-level irradiance, 2021/2022 have no data and 2023 has only 0.53% coverage, so no year is scientifically adequate.
8. **Extra high-frequency information?** Tentative and modest beyond the mean; partial correlations are about 0.11–0.14, based on an inadequate two-day sample.
9. **Supported task?** A: no for a substantive study. B: yes for regular 5-minute prediction. C: yes—re-download is required specifically for the intended multirate 2021–2023 route.
10. **Minimal prototype?** After re-download, use Site 17 Sanyo, a continuous period selected solely by source completeness within 2022 or another fully covered year, common past-only MB0/MB1/MB2 irradiance plus explicitly available past PV/weather variables, and an initial one-hour (H12) horizon. With the current file, only a non-scientific parser smoke prototype is possible on 2023-01-01 09:05 to 2023-01-03 07:40.
11. **Unsupported claims:** robust multirate forecasting across 2021–2023; seasonal high-frequency generalization; continuous-time benefit; asynchronous-observation robustness; operational use of future measured irradiance; DST-aware local-time behavior; System-availability/missingness agreement; or general high-frequency added value from this two-day sample.

## Ordinary self-checks

`python audit_multirate_data.py --self-test` verifies:

- outputs are restricted to this audit directory;
- all eight inventory records resolve to real source files and byte sizes match the sources;
- the audit run compares source size and nanosecond modification time before and after scanning and aborts on change;
- unparseable timestamps return `None`/UNKNOWN behavior; no timestamps are fabricated;
- a truncated prefix is not accepted as a full record, while an independently complete suffix may be recovered;
- all descriptive/correlation records are `past_only`, with a direct maximum-source-time assertion;
- for every non-empty common coverage group, complete + partial + no-weather intervals exactly reproduces expected intervals.

These are ordinary assertions, not hashes, contracts, freezes, baselines, or gates.

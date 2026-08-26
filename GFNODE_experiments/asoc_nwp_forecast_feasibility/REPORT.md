# Stage B0.1 — Operational GFS causality, temporal semantics, split and novelty correction

## Final verdicts

- **causal availability verdict:** `VALIDATED_PREVIOUS_COMPLETED_CYCLE_6H`. For every origin, select the latest nominal cycle satisfying `cycle + 6 h <= origin`; if objects are absent, fall back only in 6-hour decrements. Six hours is a predeclared conservative use policy, not the actual historical publication timestamp.
- **archive coverage verdict:** `PILOT_COMPLETE`. Continuous pilot objects: 608/608 successful, 0 failed; origin mappings: 2016/2016 NWP-valid.
- **GRIB temporal-semantics verdict:** `VALIDATED_FROM_MESSAGE_METADATA`. Instantaneous variables are interpolated only within one issued cycle; DSWRF uses interval-average support; APCP is converted from interval accumulation to an interval rate.
- **split representativeness verdict:** `FULL_2023_LEGAL_SEGMENTS_DEFINED`. Test is all strict legal 2023 segments, not a selected 45-day block.
- **literature novelty verdict:** `NARROW_GAP_REMAINS`. No checked single paper contains all four elements jointly, but each broad component has strong prior art.
- **B1 readiness verdict:** `B1_READY`.

No neural network training, optimizer, backward pass, or checkpoint operation was performed. Existing PV and pre-existing NWP files were not modified.

## 1. Corrected operational availability policy

`availability_policy = PREVIOUS_COMPLETED_CYCLE_6H`

For each 5-minute forecast origin in UTC:

`selected_cycle = max(cycle: cycle + 6 h <= forecast_origin)`

`forecast_age = forecast_origin - selected_cycle`

`valid_time = selected_cycle + forecast_lead`

The entire H144 future NWP trajectory comes from this one selected cycle. Missing objects cause fallback to `selected_cycle - 6 h`, then earlier cycles; they never permit a newer cycle, ERA5, or future measured weather. This policy avoids relying on unavailable historical posting timestamps. A publication-time manifest is needed only for a future claim that the method uses the “latest actually available forecast.”

The 2022-09-01 00:00 through 2022-09-07 23:55 ACST pilot produced 2,016 origin mappings, 0 total fallback events and an NWP-valid rate of 100.000%.

## 2. Continuous pilot download

- UTC cycle dates: 2022-08-31 through 2022-09-07; four cycles/day; leads f006–f024 hourly.
- Requested lead objects: 608; successful: 608; failed: 0; success rate: 100.000%.
- Exact validated byte-range GRIB payload: 3,163,949,390 bytes; official IDX objects: 24,992,652 bytes; total pilot object bytes: 3,188,942,042. The final validation rerun transferred 0 bytes because complete local objects were reused read-only.
- AWS byte ranges isolate the seven requested GRIB messages, not a spatial sub-grid; each selected global field is decoded in memory and only the nearest Alice Springs grid value is retained in the audit CSV.
- Extraction/download wall-time sum for the final validation pass: 21.1 s.
- Extrapolated selected-message volume for 2021-03-23 through 2023-12-31: approximately 373.5 GiB; allow about 1.5× this value for working disk and indexes.

## 3. GRIB time semantics and 5-minute alignment

| Variable | Observed semantics at f006 | 5-minute treatment |
|---|---|---|
| TMP 2 m | ('instant', 6.0, 6.0, 'K', 'UNKNOWN') | Linear interpolation between valid times inside the selected cycle. |
| RH 2 m | ('instant', 6.0, 6.0, '%', 'UNKNOWN') | Linear interpolation inside the selected cycle. |
| U/V 10 m | ('instant', 6.0, 6.0, 'm s**-1', 'UNKNOWN') / ('instant', 6.0, 6.0, 'm s**-1', 'UNKNOWN') | Component-wise linear interpolation inside the selected cycle. |
| TCDC | ('instant', 6.0, 6.0, '%', 'UNKNOWN') | Interpolate only when `stepType=instant`; otherwise use interval support. |
| DSWRF | ('avg', 0.0, 6.0, 'W m**-2', 0) | Treat as the average over `(startStep,endStep]`; assign that interval mean, not an instantaneous point. |
| APCP | ('accum', 0.0, 6.0, 'kg m**-2', 1) | Divide accumulation by interval duration and use the resulting rate on `(startStep,endStep]`; never interpolate cumulative totals directly. |

Ground GHI is audit/label-side information only and is prohibited from future model inputs.

## 4. Corrected splits and legal windows

The official GFS v16 operational boundary is recorded as 2021-03-22 12 UTC. Exact AWS boundary probes show that all tested f006/f024 objects for 2021-03-23 four cycles are `AVAILABLE`. Config and report therefore use identical dates:

- Train: 2021-03-23 00:00–2021-12-31 23:55 ACST.
- Validation: 2022-01-01 00:00–2022-12-31 23:55 ACST.
- Test: 2023-01-01 00:00–2023-12-31 23:55 ACST, all legal continuous fragments.

| Split | Site | raw continuous segments | legal segments (>=216 points) | L72+H144 windows | months |
|---|---|---:|---:|---:|---|
| train | Sanyo | 17 | 15 | 78,000 | 2021-03|2021-04|2021-05|2021-06|2021-07|2021-08|2021-09|2021-10|2021-11|2021-12 |
| train | Hanwha | 15 | 12 | 77,186 | 2021-03|2021-04|2021-05|2021-06|2021-07|2021-08|2021-09|2021-10|2021-11|2021-12 |
| train | Qcells | 16 | 13 | 76,969 | 2021-03|2021-04|2021-05|2021-06|2021-07|2021-08|2021-09|2021-10|2021-11|2021-12 |
| train | ALL_THREE_COMMON | 18 | 15 | 76,536 | 2021-03|2021-04|2021-05|2021-06|2021-07|2021-08|2021-09|2021-10|2021-11|2021-12 |
| validation | Sanyo | 12 | 12 | 100,310 | 2022-01|2022-02|2022-03|2022-04|2022-05|2022-06|2022-07|2022-08|2022-09|2022-10|2022-11|2022-12 |
| validation | Hanwha | 13 | 12 | 97,901 | 2022-01|2022-02|2022-03|2022-04|2022-05|2022-06|2022-07|2022-08|2022-09|2022-10|2022-11|2022-12 |
| validation | Qcells | 13 | 12 | 97,901 | 2022-01|2022-02|2022-03|2022-04|2022-05|2022-06|2022-07|2022-08|2022-09|2022-10|2022-11|2022-12 |
| validation | ALL_THREE_COMMON | 15 | 13 | 97,678 | 2022-01|2022-02|2022-03|2022-04|2022-05|2022-06|2022-07|2022-08|2022-09|2022-10|2022-11|2022-12 |
| test | Sanyo | 2,773 | 34 | 72,047 | 2023-01|2023-02|2023-03|2023-04|2023-05|2023-06|2023-07|2023-08|2023-09|2023-10|2023-11|2023-12 |
| test | Hanwha | 2,642 | 29 | 71,456 | 2023-01|2023-02|2023-03|2023-04|2023-05|2023-06|2023-07|2023-08|2023-09|2023-10|2023-11|2023-12 |
| test | Qcells | 2,639 | 29 | 71,609 | 2023-01|2023-02|2023-03|2023-04|2023-05|2023-06|2023-07|2023-08|2023-09|2023-10|2023-11|2023-12 |
| test | ALL_THREE_COMMON | 4,284 | 35 | 69,875 | 2023-01|2023-02|2023-03|2023-04|2023-05|2023-06|2023-07|2023-08|2023-09|2023-10|2023-11|2023-12 |


Each window is built only within one continuous segment and one split. No Test month, fragment, or threshold was selected by prediction error.

## 5. Literature evidence and claim boundary

The matrix retains 30 candidates. Seven highest-threat records now contain manual page/section evidence for issue-time eligibility, forecast age, lead-dependent reliability, dual-stream fusion, and their joint presence. Findings:

- SolarDB/Polasek directly occupies latest-available sampling and explicit forecast age.
- Chen et al. occupies systematic NWP-integration strategies and horizon-dependent empirical selection.
- Cross-Unet directly occupies adaptive historical/forward-weather dual-stream fusion.
- CDG occupies LMD/NWP dual encoders and NWP correction.
- Weather-mode reliability occupies reliability-based forecast-model selection, but its Discussion acknowledges same-day measured-irradiance correction as a real-world limitation.
- NWP-error robustness work occupies state/lead-dependent robustness analysis and observed feature-reallocation behavior.

No single checked work implements all four jointly. The only defensible disposition is `NARROW_GAP_REMAINS`—not first-of-kind, and no model name is assigned.

## 6. B1 boundary

`B1_READY` means the protocol is technically ready for at most one pre-registered minimal GPU screen; it is not evidence that the proposed fusion will outperform a history-only or simple NWP baseline. B1 must preserve the six-hour completed-cycle policy, one-cycle H144 trajectory, NWP-valid masks, full legal 2023 Test fragments, and the variable-specific GRIB semantics above.

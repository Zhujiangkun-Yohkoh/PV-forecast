# Stage B0 — Alice Springs operational GFS feasibility and innovation-threat audit

## Verdict

**Recommendation: CONDITIONAL GO for one minimal training screen, after one missing material is obtained: an authoritative historical GFS availability/publication-time manifest (or provider-side object inventory) covering the intended full study period.** The sampled operational archive is technically usable and contains future-direction information, but a single AWS endpoint did not cover January 2021 and object `Last-Modified` is not a guaranteed operational delivery timestamp. The training task must therefore use nominal cycle plus a documented conservative 30-minute release delay unless a stronger manifest is supplied.

No neural network was trained. No raw PV record was edited, interpolated, filled, renamed, or rewritten.

## 1. PV coverage

The raw DKASC files were parsed physical-line by physical-line; malformed lines were counted rather than silently skipped. Years meeting at least 98% timestamp coverage simultaneously across Site 17/25/38: **none**. Continuous L72+H144 window counts are:

| Year | Sanyo | Hanwha | Qcells |
|---|---:|---:|---:|
| 2021 | 96,640 | 95,826 | 95,609 |
| 2022 | 100,310 | 97,901 | 97,901 |
| 2023 | 72,047 | 71,456 | 71,609 |

There are **not two complete common years** under the strict 98% timestamp-and-valid-power definition. The recommended non-Test-tuned protocol is: Train = 2021-03-01 through 2021-12-31 (AWS GFS v16 archive availability boundary, retaining gaps explicitly); Validation = 2022-01-01 through 2022-12-31 (retaining gaps); Test = the predeclared longest common valid 2023 block, **2023-08-23 03:15:00 through 2023-10-07 12:10:00**, which contains 12,853 strict L72+H144 windows. Windows must be built only inside uninterrupted segments; no imputation may bridge a gap.

PV timestamps are timezone-naive. They are interpreted as ACST based on the project's prior authoritative UTC/ACST audit; GFS uses UTC exclusively and is converted by +09:30.

## 2. GFS archive and causality

The NOAA Open Data/AWS GFS archive was queried by exact `.idx` and GRIB2 URLs. The 2021-01-15 object returned 404, while sampled dates in July 2021, 2022 and 2023 existed. GFS cycles are nominally 00/06/12/18 UTC. The operational matching rule is:

`selected_cycle = latest cycle with cycle_time + 30 min <= forecast_origin_UTC`.

This 30-minute delay is conservative relative to NOAA documented product delays (roughly 8–20 minutes for pressure GRIB products), but it is not a reconstruction of the exact historical posting second. Every downloaded message was checked against GRIB metadata for cycle, lead, valid time, units and nearest grid coordinate. Linear interpolation from issued hourly/3-hourly GFS values to 5-minute timestamps is causal because both interpolation endpoints belong to the same already-issued forecast trajectory; it adds temporal smoothness, not future observation information.

Actual analyzed sample: **6 local days** (2 clear, 2 cloudy, 2 high-change), spanning 2021–2023 and multiple months; two cycles per local day and leads 3/6/9/12 h. An interrupted broader pilot left 0 additional official selected-record subsets in the authorized raw NWP directory; they were not used in statistics and were not deleted or altered. Analyzed subset: 48 unique GRIB files, approximately 254.5 MiB.

## 3. Preliminary information value

At sampled valid times, GFS downward short-wave radiation was compared with the same-time ground GHI and PV power. Results by lead, cycle and scenario are in the inventory. Direction agreement based on successive sampled leads was **0.667 for PV** and **0.722 for ground GHI** across 36 valid changes. This is preliminary descriptive evidence that issued future GFS trajectories contain some directional information unavailable from historical observations alone; it is not a performance claim and is too small for final inference.

All three arrays are co-located and can use identical GFS issue/valid timestamps. Their targets remain array-specific, enabling Site 17 development and Site 25/38 independent evaluation without changing exogenous information.

## 4. Literature overlap and algorithmic novelty threat

The matrix contains 30 candidate works and 17 full-text-level checks. The proposed idea—reliability-adaptive fusion of historical observations and future NWP using issue time, forecast age and lead time—**is not wholly unoccupied**:

- Polasek et al. explicitly use the latest available weather forecast and increasing forecast age.
- the SolarDB/Applied Energy study explicitly analyzes forecast age under uncertain weather forecasts;
- Liu et al. use separate local-measurement and NWP encoders plus NWP correction;
- Chen et al. compare multiple NWP integration strategies and horizon-dependent behavior;
- Cross-Unet adaptively emphasizes forward-looking weather channels alongside historical records;
- weather-mode reliability and NWP-error robustness papers directly threaten a broad “reliability-aware fusion” claim.

No checked paper was found that combines all four elements exactly in this Alice Springs 1–12 h task: operational issue-time eligibility, explicit forecast-age representation, lead-dependent reliability, and adaptive dual-stream fusion. That narrower coupling may be defensible, but only after a formal claim chart and a minimal controlled comparison. Do not name a model or claim novelty yet.

## 5. Scale estimate and next action

Selected-variable byte-range retrieval avoids multi-terabyte full-GRIB downloads. Extrapolating the measured subset volume to four cycles/day, hourly leads 0–18 and 2021-03 through 2023 suggests roughly **80–250 GB download and 120–400 GB working disk**, depending on message compression and whether hourly or 3-hourly leads are retained. On the present connection, plan for several days of download plus 1–3 days of point extraction/validation; exact timing must be measured by a pilot month.

**Only requested additional material:** an authoritative GFS historical object/publication-time inventory for 2021–2023 (NOAA/NODD or provider export), sufficient to prove which cycle products were available when. With that supplied, proceed to at most one pre-registered minimal screen: history-only versus causally available GFS, followed by the single issue-age/lead reliability fusion candidate. Without it, retain `CONDITIONAL GO` and do not train.

## 6. Scientific boundaries

- ERA5 and future measured ground weather are excluded from model inputs.
- Sample-day selection is descriptive and cannot tune a Test threshold or model.
- The sampled correlations do not establish annual performance, deployment readiness, or causal benefit.
- Exact historical delivery time is not recoverable solely from nominal cycle and current object metadata.
- No cross-climate or cross-location generalization is supported.

# Scheme C1-S4 — annual source and executable-readiness review

## Two independent verdicts

- Data: **`C1_FORMAL_DATA_FAIL`**.
- Implementation: **`C1_FORMAL_IMPLEMENTATION_READY_FOR_GPU_REVIEW`** after 28 synthetic production-function tests and 7 real-array tests pass.
- GPU execution: **NOT AUTHORIZED / not performed (0/9)**.
- Scientific method outcome: **NOT EVALUATED**.

## Official-source finding

For both 2021 and 2023 the defensible source verdict is **`OFFICIAL_FULL_YEAR_UNAVAILABLE_OR_UNCONFIRMED`**. The official DKASC NT Solar Resource page lists Alice Springs annual 5-minute and 5-second downloads for both years and states that the Class-A stations collect high-resolution data. The official Fulcrum3D interface separately exposes a user-selected date range and a 1-second irradiance download. Neither public page guarantees that a requested 1-second export contains every second of a calendar year, documents an annual 1-second row limit, or explains these two malformed exports. Therefore portal availability is not evidence of complete 1-second annual coverage. The present files support download/export failure or wrong selection as possibilities, but do not distinguish them from true upstream gaps.

Official pages checked 2026-08-28: https://www.dkasolarcentre.com.au/download?location=nt-solar-resource and https://nt-solar-resource.fulcrum3d.com/download . The DKASC page defines System Status (0=OK, 1=issue), but the selected CSV headers contain no status field.

## Explicit sources and annual union

|Year|Files|Bytes|Unique seconds|First UTC|Last UTC|Missing|Duplicate|Out-of-year|Structure|
|---:|---:|---:|---:|---|---|---:|---:|---:|---|
|2021|1|1,726,481,334|18,403,200|2021-06-02 00:00:00|2021-12-31 23:59:59|13,132,800|0|0|columns 0; glued 0; truncated 0; Data Error 0|
|2022|1|3,011,901,400|31,536,000|2022-01-01 00:00:00|2022-12-31 23:59:59|0|0|0|columns 0; glued 0; truncated 0; Data Error 0|
|2023|1|1,774,300,567|167,952|2023-01-01 00:00:01|2023-01-02 22:39:12|31,368,048|0|18,489,565|columns 212; glued 1; truncated 211; Data Error 211|

2021 contains only 2021-06-02 through year-end. The 2023 source begins one second late, ends its target-year coverage on 2023-01-02, then contains 2024/2025 rows and structural damage. The excluded older damaged 2023 file was not used. The 2022 authoritative redownload remains complete.

All years use `Timestamp_UTC`; ACST is derived as UTC+09:30. MB0/MB1/MB2 remain separate in W/m². No interpolation, repair, Excel rewrite, or concatenated raw copy was made. Config now accepts an explicit list of one or more annual/monthly files; ordering and overlap checks use actual UTC records.

## Five-minute quality

|Year|MB0 missing|MB1 missing|MB2 missing|Strict complete bins|Partial bins|Empty bins|
|---:|---:|---:|---:|---:|---:|---:|
|2021|781,177|781,579|781,759|58,711|2,633|43,776|
|2022|3,992|3,236|3,320|99,211|5,909|0|
|2023|59|58|42|464|96|104,560|

Right-closed `(t-5 min,t]` aggregation is fixed. Partial numeric missingness is retained through channel mean, valid fraction and valid mask; zero-timestamp bins interrupt windows.

## Frozen primary population

All seven success conditions use exactly **H12 + THREE_ARRAY_COMMON + mask-available + PRIMARY_DAYLIGHT_COMMON**. Strict 300/300 three-channel completeness is only a data-quality sensitivity population and cannot select the main result.

## Five-stage common origins

|Stage|Expected|Formal common|Strict 300/300|First|Last|Segments|Months|Seasons|
|---|---:|---:|---:|---|---|---:|---|---|
|BASE_TRAIN|105,037|58,122|54,067|2021-06-02T15:25|2021-12-31T22:55|13|6;7;8;9;10;11;12|spring;summer;winter|
|BASE_MODEL_VALIDATION|34,477|31,750|31,315|2022-01-01T05:55|2022-04-30T22:55|7|1;2;3;4|autumn;summer|
|RISK_FIT|35,341|32,509|31,676|2022-05-01T05:55|2022-08-31T22:55|6|5;6;7;8|autumn;winter|
|RISK_CALIBRATION|35,053|34,969|2,300|2022-09-01T05:55|2022-12-31T22:55|2|9;10;11;12|spring;summer|
|FINAL_TEST|105,037|604|0|2023-01-01T05:55|2023-01-03T08:10|1|1|summer|

## Implementation review

The READY path is complete and guarded rather than an unimplemented exception. It provides 14-channel causal construction; BASE_TRAIN-only median/scalers/target range; the frozen 4-layer 64-channel depthwise/pointwise TCN and 9-run matrix; AdamW loop and strict validation-RMSE checkpoint replacement; matched last-value Persistence; the fixed risk features and HistGradientBoostingRegressor; RISK_FIT-only fitting; same-scope `method=higher` calibration; stable-origin AURC; fixed-mask 7-day moving-block and day-cluster resampling; and programmatic seven-condition evaluation. This stage called none of the real fitting/training functions.

State fields are factual: raw Final-Test availability metadata was inspected; model predictions, errors, risk scores, coverage and AURC were not generated or accessed. NOT_RUN rows are execution status, not performance metrics.

## Tests and source protection

- Synthetic/fixture: 28/28 passed; 0 skipped.
- Real arrays: 7/7 passed; 0 skipped.
- Selected PV and irradiance source size/mtime_ns unchanged: **True**.
- Real optimizer/backward/epoch: **No**.
- Real risk-model fitting: **No**.
- Final-Test performance access: **No**.

Local `results/` contains only compact review artifacts and remains untracked; it is not a clean-worktree claim and will not be committed.

## Conclusion

Data remains **`C1_FORMAL_DATA_FAIL`**, so no GPU authorization follows. Implementation is **`C1_FORMAL_IMPLEMENTATION_READY_FOR_GPU_REVIEW`**, but that does not override missing annual data. The only defensible next action is to obtain officially exported 1-second 2021 and 2023 blocks whose explicit UTC union passes the frozen annual criteria; no alternative year, interpolation, model revision, or C1 v2/v3 is proposed.

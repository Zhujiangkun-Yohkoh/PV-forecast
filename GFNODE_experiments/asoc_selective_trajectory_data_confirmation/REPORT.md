# Scheme C1-S2 — Formal Data Confirmation and Executability Review

## Final decision

**`C1_FORMAL_DATA_FAIL`**

The five-stage protocol is now implementation-ready and unambiguous, but the required raw inputs are not ready. No authoritative full-calendar 2021 second-level MB0/MB1/MB2 file exists at the configured data location or elsewhere found under the project/Desktop download search. The local file labelled as 2023 second-level irradiance remains the previously damaged multi-year export: only 167,952 unique 2023 seconds are present, ending on 2023-01-02 22:39:12 UTC. Therefore `BASE_TRAIN` has zero irradiance-supported origins and `FINAL_TEST` has only 604 masked common origins in early January and zero strict all-channel-complete origins.

GPU training is **not authorized**. No substitute year, interpolation, repair, C1 v2/v3, or alternative model is proposed.

## Scope and isolation

This audit read only the three raw five-minute PV files, the configured 2021–2023 irradiance paths, C1-S0R, C1-S1/PR #6, and the verified 2022 redownload implementation/report. It did not load or generate a 2023 prediction, error, risk score, coverage, AURC, or accepted-set metric. It did not fit a risk model, instantiate an optimizer, call backward, write a checkpoint, train a neural network, or modify any source data.

All irradiance timestamps use the file's authoritative UTC field and are mapped to the timezone-free PV clock by fixed `UTC+09:30` ACST. The exported `Local` field is not used.

## Second-level irradiance validation

| Year | Path status | Size | Unique target-year seconds / expected | UTC extent within target year | Structural findings | Five-minute complete / partial / empty | Verdict |
|---|---|---:|---:|---|---|---:|---|
| 2021 | expected file absent | 0 | 0 / 31,536,000 | UNKNOWN | no second-level source to parse | 0 / 0 / 105,120 | FAIL |
| 2022 | present | 3,011,901,400 B | 31,536,000 / 31,536,000 | 2022-01-01 00:00:00–2022-12-31 23:59:59 | 0 duplicate, inverse, column, quote, glued, truncated, or Data Error records | 99,211 / 5,909 / 0 | PASS |
| 2023 | present but damaged | 1,774,300,504 B | 167,952 / 31,536,000 | 2023-01-01 00:00:01–2023-01-02 22:39:12 | 209 column anomalies; 2 glued lines; 208 truncated/Data Error lines; later parseable records jump through 2025-05-15 | 464 / 96 / 104,560 | FAIL |

“Complete” means all 300 expected seconds and all 300 finite values in each of MB0, MB1, and MB2 within a calendar five-minute bin. “Partial” keeps timestamp or channel-level incompleteness explicit. No bad line was silently skipped, no truncated value was guessed, and no timestamp/value was interpolated.

The generic file `.../高分辨率气象数据集/..._irradiancedata.csv` is a five-minute export beginning 2021-06-02, not a full-calendar second-level 2021 source. It is not used as a substitute.

The 2022 scan independently reproduces the prior authoritative validation: 31,536,000 physical data records, one-second dominant interval, no missing timestamp, and MB0/MB1/MB2 missing-value counts of 3,992/3,236/3,320. The annual structural bin counts use left-closed calendar bins. Causal model inputs continue to use the existing right-closed `(t-5 min, t]` aggregation.

## Five-minute PV validation

All three files contain the original `timestamp` and `Active_Power` fields, have no duplicate, inverse, off-grid, or malformed record before 2024, and are treated as ACST under the audited mapping.

| Array | Year | Present / 105,120 | Valid Active Power | Missing Active Power among present rows |
|---|---:|---:|---:|---:|
| Site 17 Sanyo | 2021 | 100,539 | 100,534 | 5 |
| Site 17 Sanyo | 2022 | 102,901 | 102,890 | 11 |
| Site 17 Sanyo | 2023 | 101,797 | 96,941 | 4,856 |
| Site 25 Hanwha | 2021 | 100,541 | 99,076 | 1,465 |
| Site 25 Hanwha | 2022 | 102,901 | 100,501 | 2,400 |
| Site 25 Hanwha | 2023 | 101,833 | 95,908 | 5,925 |
| Site 38 Q CELLS | 2021 | 100,539 | 99,074 | 1,465 |
| Site 38 Q CELLS | 2022 | 102,901 | 100,501 | 2,400 |
| Site 38 Q CELLS | 2023 | 101,812 | 95,981 | 5,831 |

## Formal five-stage window recomputation

The formal masked origin rule is fixed as follows:

- all 72 five-minute PV inputs, the origin, and all 12 future PV targets must exist and have finite `Active_Power`;
- the 72 causal irradiance bins use `(t-5 min, t]` and must contain source timestamps;
- MB0/MB1/MB2 remain separate; partial numerical missingness is retained through channel-specific valid fraction and valid mask, with fill values later fitted on `BASE_TRAIN` only;
- a bin with zero source timestamps breaks the continuous segment;
- no window crosses a stage, gap, or segment boundary.

For transparency, a stricter diagnostic also requires all 72 history bins to have exactly 300 timestamps and 300 finite values in every MB channel.

| Stage | Population | Mask-eligible origins | Strict complete origins | Lost vs calendar | First / last legal origin (ACST) | Segments |
|---|---|---:|---:|---:|---|---:|
| BASE_TRAIN (2021) | Site 17 / 25 / 38 / common | 0 / 0 / 0 / **0** | 0 / 0 / 0 / **0** | common 105,037 | NONE | 0 |
| BASE_MODEL_VALIDATION (2022 Jan–Apr) | Site 17 / 25 / 38 / common | 31,641 / 31,643 / 31,643 / **31,636** | 31,205 / 31,207 / 31,207 / **31,200** | common 2,841 | 2022-01-01 15:25 / 2022-04-30 22:55 | 7 |
| RISK_FIT (2022 May–Aug) | Site 17 / 25 / 38 / common | 35,004 / 32,593 / 32,593 / **32,509** | 34,171 / 31,760 / 31,760 / **31,676** | common 2,832 | 2022-05-01 05:55 / 2022-08-31 22:55 | 6 |
| RISK_CALIBRATION (2022 Sep–Dec) | Site 17 / 25 / 38 / common | 34,969 / 34,969 / 34,969 / **34,969** | 2,300 / 2,300 / 2,300 / **2,300** | common 84 | 2022-09-01 05:55 / 2022-12-31 22:55 | 2 |
| FINAL_TEST (2023) | Site 17 / 25 / 38 / common | 604 / 604 / 604 / **604** | 0 / 0 / 0 / **0** | common 104,433 | 2023-01-01 05:55 / 2023-01-03 08:10 | 1 |

The 2022 common origins cover all twelve months and all four Southern Hemisphere seasons across stages B–D. The 2021 and 2023 blockers prevent the formal five-stage study despite adequate five-minute PV coverage.

## Frozen formal protocol revision

### Endpoints and populations

The sole formal primary endpoint is **H12, origin-daylight, three-array common mask-eligible origins**. H3/H6 prefixes, full timeline, and array-specific all-valid origins are secondary. H3/H6/H12 always come from the same direct H12 trajectory.

### Nominal 80% operating point

For each array, seed, risk method, horizon, and scope, calibration uses only the same scope in `RISK_CALIBRATION`. For `n` finite calibration scores and `q=0.80`:

1. sort scores in ascending order;
2. save zero-based order index `ceil(q × (n−1))`;
3. set `threshold = numpy.quantile(scores, q, method="higher")`;
4. accept when `score <= threshold`;
5. save `n`, order index, threshold, accepted count, and realized calibration coverage.

Ties may make realized calibration coverage exceed the nominal value. Test scores never alter the threshold. This is an empirical operating rule, not conformal, distribution-free, or finite-sample coverage assurance.

### AURC

For each origin, the fixed loss is physical-unit H12 trajectory RMSE:

`l_i = sqrt(mean_{j=1..12}((y_ij − yhat_ij)^2))`.

For each of the 96 coverage points `c = 0.05, 0.06, …, 1.00`, sort the common origin set by increasing risk score, take `k(c)=max(1, ceil(cN))`, and compute `R(c)=mean(l_i)` over those `k(c)` origins. The normalized AURC is:

`AURC = trapezoid(R(c), c) / (0.95 × R(1.00))`.

Lower is better. `FULL_RISK_MODEL`, `RECENT_VARIATION`, and `MODEL_PERSISTENCE_DISAGREEMENT` use identical origins and the same grid. Accepted-RMSE claims are not substituted for AURC when realized coverages differ.

### Fixed model, features, and training plan

`config.json` records the complete pre-GPU specification:

- base model: `DEPTHWISE_TCN_TRAJECTORY`, channels 64, four depthwise/pointwise layers, kernel 5, dropout 0, direct H12 output;
- input: historical Active Power, four time encodings, separate MB0/MB1/MB2 five-minute means, valid fractions, and masks;
- deep optimizer: AdamW, learning rate 0.001, weight decay 1e-5, batch 256, 25 epochs maximum, patience 5, gradient clipping 1.0, no scheduler, no mixed precision, workers 0, target-scaled MSE;
- risk estimator: fixed `HistGradientBoostingRegressor` parameters from C1-S0R, fitted only on `RISK_FIT` to `log1p` normalized H12 trajectory loss;
- fixed causal risk features: six time/origin features; eight PV statistics over 12/36/72 points; five separate statistics for each MB channel over 12 points; six model-versus-Persistence/trajectory-shape features;
- high-error quantile 0.8 and high-change quantile 0.9 are configuration fields, not code constants;
- primary uncertainty: fixed-mask continuous seven-day moving-block bootstrap, 1,000 replicates; natural-day clustering is sensitivity only.

No risk feature may use a timestamp after the forecast origin.

### Origin-daylight reference

The official DKASC ratings are supported by panel rating × panel count: Sanyo 30×210 W=6.3 kW, Hanwha 22×265 W=5.83 kW, and Q CELLS 20×295 W=5.9 kW. This supports **DC nameplate array rating**, not a verified AC rating directly comparable to metered `Active_Power`. Official sources:

- [Site 17 Sanyo](https://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase)
- [Site 25 Hanwha](https://dkasolarcentre.com.au/source/alice-springs/dkasc-alice-springs-25-hanwha-q-cells-poly-si-fixed)
- [Site 38 Q CELLS](https://dkasolarcentre.com.au/source/alice-springs/dka-m19-b-phase)

Therefore the formal causal fallback is fixed array-specific `BASE_TRAIN` valid-positive `Active_Power` p99.9, without inspecting `FINAL_TEST`: 5.984334/5.565733/5.604867 kW for Sites 17/25/38. The corresponding 1% origin-daylight thresholds are **0.059843/0.055657/0.056049 kW**. These are Train-derived reference thresholds, not AC nameplate ratings.

### Preregistered primary-endpoint success rule

All conditions apply only to H12 origin-daylight common origins:

1. three-array macro Test coverage is in [0.75, 0.85], and no array is below 0.70;
2. `FULL_RISK_MODEL` macro AURC improves at least 5% over the best simple score;
3. AURC improvement direction holds for at least two of three arrays;
4. at least two of three arrays beat matched Last-value Persistence on identical accepted origins;
5. accepted RMSE is at least 10% below unselected RMSE;
6. seed-level three-array macro improvement direction holds for at least two of three seeds.

Any failure closes C1 without v2/v3. Macro results cannot hide an array-level failure.

## Ordinary tests

**14/14 tests passed** using the real audit arrays:

- source-backed status for all three irradiance years and fixed UTC→ACST conversion;
- no stage/segment crossing and exact three-array origin intersection;
- future-sentinel mutation leaves an actual origin's PV/MB foundation features element-identical;
- fictional Final-Test score mutation leaves the calibration threshold unchanged;
- `method="higher"` order statistic behavior;
- preprocessing fit rejects every stage except `BASE_TRAIN`;
- ordered, disjoint five-stage roles and input timestamps no later than origin;
- no training/gradient/checkpoint-write call in the audit script;
- raw file byte size and nanosecond mtime unchanged;
- no 2023 prediction/outcome access, risk fitting, or deep training.

`DATA_CONFIRMATION_SUMMARY.csv` contains 359 long-format data rows and was created and re-imported with the spreadsheet artifact library. The summary and its key counts can be regenerated from the raw files by the committed script.

## Required failure record

- Failed input: Alice Springs second-level MB0/MB1/MB2, calendar 2021; expected configured file is absent, so the entire `BASE_TRAIN` irradiance history is unavailable.
- Failed input: Alice Springs second-level MB0/MB1/MB2, calendar 2023; local file is structurally damaged and contains usable timestamps only through 2023-01-02 22:39:12 UTC before later-year material.
- Affected formal roles: `BASE_TRAIN` and `FINAL_TEST`.
- GPU authorization: **denied**; authorized formal deep runs this round: **0**.

No original PV/NWP/irradiance file was modified. Scheme A, master, C1-S0R/PR #4, PR #5, PR #6, and the NWP branches were not modified or merged.

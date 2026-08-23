# PV Dataset Lineage and Recoverability Audit

## Decision

**C. TIMESTAMPS_AVAILABLE_BUT_MISSINGNESS_LOST**

The best local source candidates retain explicit, parseable 5-minute timestamps, physical-looking unscaled measurements, original field names, and value-level NaNs. However, each spring file already contains every point on a perfectly regular grid from 2018-04-01 00:00 through 2018-08-31 23:55. No untouched DKASC export, download manifest, source-row presence mask, or acquisition log was found. Consequently, the current files cannot establish whether timestamps were absent upstream and later inserted, nor recover the original observation process.

This is a dataset-lineage decision, not a statement that the current clean forecasting protocol is unusable for ordinary regular-grid forecasting.

## Scope and method

The read-only scanner inspected `PVforecast16`, `PV_improve_v1`, all dataset paths referenced by accessible source/configuration text, and the file list of `origin/master`. It found **404 data/result files**. The inventory records table/array structure and content-derived lineage rather than classifying from filenames. Of these, 8 files meet the stated project-level RAW-candidate criteria (the four primary CSVs and their identical copies), while the remainder are derived results, window/prediction arrays, scaled artifacts, or files whose level cannot be supported from content.

The scanner does not deserialize pickle files, fabricate timestamps, regularize data, or write outside this audit directory. It records source file size and modification time before and after inspection and aborts if either changes. `origin/master` was inspected with `git ls-tree`; it contains code and compact result tables but not the primary PV CSVs.

## Best original-data candidates

| Array / period | Best local candidate | Shape | Timestamp range | Timestamp evidence | Value missingness | Assessment |
|---|---|---:|---|---|---|---|
| Sanyo, spring | `PVforecast16/17Sanyo.csv` | 44,064 × 9 | 2018-04-01 00:00–2018-08-31 23:55 | 44,063 intervals of exactly 5 min; 0 duplicates; 0 absent grid points | tilted global and diffuse radiation: 865 each; all other columns: 0 | Best project RAW candidate, but not proven to be an untouched DKASC export |
| Hanwha, spring | `PVforecast16/25Hanwha.csv` | 44,064 × 9 | same | same | Active_Power and Performance_Ratio: 858 each; tilted radiation columns: 865 each | Same limitation |
| Qcells, spring | `PVforecast16/38QCELLS.csv` | 44,064 × 9 | same | same | Active_Power and Performance_Ratio: 858 each; tilted radiation columns: 865 each | Same limitation |
| Hanwha, autumn | `PVforecast16/25Hanwha_Differentseason.csv` | 44,064 × 9 | 2018-09-01 00:00–2019-01-31 23:55 | 44,063 intervals of exactly 5 min; 0 duplicates; 0 absent grid points | Active_Power and Performance_Ratio: 36 each; tilted radiation columns: 44 each | Same limitation; seasonal extension only |

Each candidate has an exactly equal copy under `PVforecast16/GFNODE_experiments/`: equal shape, column order, timestamps, values, NaN positions, first record, and last record. This is a direct dataframe comparison, not a file hash. The copies are therefore duplicate locations of the same project-prepared upstream dataset, not independent raw sources.

All candidates expose these fields in this order: `timestamp`, `Active_Power`, `Performance_Ratio`, `Weather_Temperature_Celsius`, `Weather_Relative_Humidity`, `Global_Horizontal_Radiation`, `Diffuse_Horizontal_Radiation`, `Radiation_Global_Tilted`, `Radiation_Diffuse_Tilted`. The numeric ranges are not uniformly normalized. `Active_Power` is present in an unnormalized scale, but the file does **not** encode its unit; kW must not be inferred solely from magnitude. No candidate contains an original-row missing mask or Isolation Forest result.

The six meteorological columns are element-for-element equal across Sanyo, Hanwha, and Qcells, including NaN positions. This supports a common co-located weather stream in the project files. The project revision plan states that the three arrays are at Alice Springs and share weather (`PV_improve_v1/GFNODE_Revision_Plan.md:74`), while the data availability statement identifies DKASC as the public source (`PV_improve_v1/Final_Submission/Data_Availability_Statement.md:3`). Neither document supplies a download identifier, export parameters, field-level provenance, original file schema, or source-row manifest.

## Data lineage

1. **Documented upstream source:** DKASC public database, Alice Springs, according to the paper workspace. Exact export files and acquisition settings are absent locally.
2. **Best available project source layer:** the four 9-column CSVs above. They preserve real timestamp strings and some value-level NaNs, but already have complete 5-minute timestamp grids. Their exact transformation history before entering this repository is not documented.
3. **Legacy MANODE/GFNODE ingestion:** `main.py:89,1444-1446` and `GFNODE_experiments/gfnode_solo_benchmark.py:52,561-563` read the three named spring CSVs. The MANODE-related item found locally is a paper PDF; no separate original MANODE code/data package or download manifest was found. Thus its use of these files can be traced through the present project naming and ingestion code, but the untouched MANODE upstream dataset cannot be independently verified.
4. **Clean protocol:** `GFNODE_experiments/asoc_clean_decision/asoc_clean_decision.py:60-94` reads the configured CSV, parses/deduplicates timestamps, creates a full 5-minute index, and reindexes. Because the candidate spring CSVs already contain the entire grid, the current run adds no timestamp rows, but this cannot prove that an earlier preparation step did not do so. Configured paths are in `asoc_clean_decision/config.json:4-5`.
5. **Discrete viability:** `GFNODE_experiments/asoc_discrete_viability/config.json:3` points all three datasets to those same CSVs; `benchmark.py:82-83` delegates loading and preprocessing to `CleanDataProtocol`.
6. **Cross-technology work:** feasibility code imports the same `CleanDataProtocol` (`asoc_cross_technology_feasibility/cross_technology.py:8,18`), and the screen imports it at `asoc_cross_technology_screen/run_experiment.py:7,34`. These experiments therefore share the same best-available project source layer rather than an independent data download.
7. **Derived artifacts:** saved window tensors, predictions, labels, masks, metrics, and plotting tables are inventoried as WINDOWED, PREDICTION, SCALED, or UNKNOWN according to their content. They are not substitutes for the source timestamp stream.

## Recoverability and scientific consequences

What is retained:

- explicit timestamp strings and their spring/autumn date ranges;
- a regular 5-minute clock;
- original semantic column names;
- value-level NaNs in power/PR and tilted-radiation columns;
- unnormalized physical-looking numerical values;
- an exact mapping from every current CSV row to its current timestamp.

What is not recoverable from current local or GitHub assets:

- whether DKASC originally omitted any timestamps;
- which current rows, if any, were inserted before the files entered this project;
- the original acquisition/availability mask independently of measurement-value NaNs;
- sensor outage versus preprocessing-induced missingness;
- the exact raw export schema, units, download query, and array metadata;
- an untouched source file against which the current CSVs can be provenance-validated.

Therefore the current assets are insufficient to support claims about learning from the **original irregular observation process**, observation-controlled NODE dynamics driven by genuine asynchronous arrivals, robustness to naturally occurring timestamp dropout, or causal conclusions about sensor-availability patterns. A study may still use the observed value-level NaNs on the present regular grid, but must describe them narrowly and must not equate them with original timestamp missingness.

## Observation-controlled NODE suitability

The preserved value-level missing blocks can support a limited masked-input experiment on a regular clock. They do **not** provide adequate evidence for an observation-controlled NODE study whose scientific premise depends on true asynchronous observations or original missing-event timing. Such a study first requires a fresh authoritative DKASC export retaining source timestamps and metadata, saved alongside an explicit source-row presence mask. The new download should be compared by timestamp, schema, row count, and boundary records to the current candidates before any modeling.

## Ordinary audit tests

`python audit_data_lineage.py --self-test` passed:

- output is restricted to this audit directory; the scanner snapshots size and modification time for every source data file and verifies no change;
- all 404 inventory rows point to existing source files with source-reported byte sizes, and simulated/dummy paths are rejected;
- an unparseable timestamp series returns `UNKNOWN` for parse status and range; no time index is constructed as a fallback.

These are ordinary implementation checks, not a contract, hash chain, freeze, baseline, or gate.

## Recommendation

For ordinary point forecasting, retain the existing clean protocol but describe the four CSVs as the **best available project source files**, not provably untouched raw downloads. For any observation-controlled or asynchronous-time scientific claim, the single recommended action is to re-download the authoritative DKASC data with station/array metadata and preserve the exact source timestamps, units, query/export settings, and original-row missing mask before designing experiments. No data were downloaded or modified in this audit.

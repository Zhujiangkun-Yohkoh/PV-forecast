# Clean deterministic benchmark: unified evidence report

## Outcome

**Final determination: READY_FOR_MANUSCRIPT_WRITING.** All 36 expected neural benchmark runs are completed and map to real H144 prediction artifacts. Their labels, forecast-origin timestamps, and saved masks are elementwise identical within each dataset, and every horizon was recomputed from the same H144 prefix. No neural-network training, optimization, backward pass, checkpoint modification, or manuscript edit was performed.

## Dataset metadata and terminology

DKASC official array pages support describing the data as **three co-located PV technologies at the Alice Springs DKA Solar Centre**: Sanyo HIT hybrid silicon (Site 17), Hanwha Solar poly-Si (Site 25), and Q CELLS mono-Si (Site 38). Official array ratings are 6.3, 5.83, and 5.9 kW, respectively. The source power field is `Active_Power`, mapped by the DKASC glossary to 5-minute average AC power in kW. Official rated AC system capacities were not established (Sanyo, Hanwha, Qcells), and component-side array ratings are therefore not used as AC error denominators. Latitude and longitude remain `UNKNOWN` rather than inferred.

The final normalization is **range_nRMSE = RMSE / (Train maximum - Train minimum)**, fitted separately by dataset using Train only. Daylight is model-independent and defined as true target power greater than 1% of that dataset's Train maximum; this avoids using prediction outputs or Test-derived thresholds.

## Run and sample evidence

- Neural runs: 36/36 completed; models entering the main table are Discrete Candidate, iTransformer, PatchTST, and ModernTCN.
- Forecast task: one direct H144 prediction; H12/H48/H96/H144 are the 1/4/8/12-hour prefixes over the same origins.
- Last-value persistence uses only the exact forecast-origin power and the same H144 sample set.
- Daily seasonal persistence uses exact target timestamps lagged 288 five-minute steps, without interpolation; because validity can differ, it is a supplemental comparator with explicit counts.
- Persistence has no seed. Neural skill is computed against the identical deterministic last-value reference for each seed.

## Unified metrics

ModernTCN ranks first by mean RMSE in **18 of 24 dataset × horizon × scope combinations** (full timeline and daylight treated separately). Neural mean-RMSE skill contains **8 non-positive model-combinations** relative to last-value persistence; the long table retains all such outcomes rather than selectively reporting wins. Full-timeline and daylight rankings must be presented separately because nighttime zeros can materially change rankings.

`FINAL_METRICS_LONG.csv` contains per-seed values, sample mean, sample SD, deterministic persistence rows, RMSE/MAE skills, and RMSE ranks. Negative R² values are retained. The CSV is the sole numeric source for paper-wide result tables.

## Efficiency

Uniform GPU inference measurement completed for all four neural architectures using representative Hanwha seed-42 best-validation checkpoints on NVIDIA GeForce RTX 3060 Laptop GPU. All measurements use float32, `eval()`, inference mode, input shape [batch,72,15], 100 warmups, 500 batch-1 repetitions, and a common batch size of 256 for throughput. Loading, disk I/O, data loading, and host-device transfer are excluded. These are architecture-level measurements; historical heterogeneous latency values must not be mixed with them.

## Provenance and ordinary checks

The evidence builder verified: 36 real artifact mappings; H144 shapes; finite predictions/labels; within-dataset equality of labels, timestamps, and saved masks; equality with a freshly reconstructed clean protocol; Train-only normalization and daylight thresholds; causal persistence; exact 288-step daily lag; per-seed aggregation and ranks; inference-only timing; and absence of writes to source data/checkpoints/artifacts. Paths, seed, config, and source commit appear on every metrics row. No hashes or frozen registries were introduced.

## Remaining limitations

- Official rated AC capacities and exact latitude/longitude remain unknown; capacity-normalized RMSE is not supportable.
- The three arrays are co-located and share the project weather observations; results do not support cross-location, cross-climate, or regional generalization.
- Daily persistence is supplemental wherever exact 24-hour lag availability reduces the valid set.
- The 2022 Site-17 investigations are exploratory and are not mixed into this 2018 three-array benchmark main table.

## Manuscript-ready outputs

Next, generate only: (1) dataset/protocol table from `FINAL_DATASET_METADATA.csv` and split rows in `FINAL_SAMPLE_COUNTS.csv`; (2) multi-horizon mean±SD table from `FINAL_METRICS_LONG.csv`; (3) daylight/full ranking figure; (4) persistence skill figure; (5) parameter–RMSE trade-off and uniform latency table from `FINAL_EFFICIENCY.csv`; and (6) a leakage-free preprocessing/window schematic. No additional neural training is required for the planned application/benchmark manuscript.

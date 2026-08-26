# Stage B1 — Causal GFS minimal performance screen

## Verdict

**NWP_INFORMATION_FAIL**

- New GPU training: 9/9 completed.
- Numerical divergence/non-finite gradients: none.
- Sealed 2023 Test accessed: **no**.
- Models use identical Train/Validation origins, labels and masks; H12/H48/H96/H144 are prefixes of one H144 output.

## Data and operational protocol

- Site 17 Sanyo only; no cross-site or cross-climate claim.
- Train: 2021-03-23 00:00:00 to 2022-01-01 00:00:00 (exclusive end).
- Validation: 2022-01-01 00:00:00 to 2023-01-01 00:00:00 (exclusive end).
- `PREVIOUS_COMPLETED_CYCLE_6H`; each H144 NWP trajectory uses one cycle satisfying `cycle + 6h <= origin`.
- Official GFS `pgrb2.1p00` (1.0°), DSWRF and TCDC only, on the predeclared 3-hour lead grid f006/f009/.../f024. This B1 bandwidth-minimal product is explicitly distinct from the B0.1 0.25° hourly pilot.
- Train windows: 53,210 in 119 legal fragments; Validation windows: 59,424 in 176 fragments.
- Train NWP coverage: 53,210/53,210 (100.000%); Validation: 59,397/59,424 (99.955%).
- Unique official objects retained: 13,047; successful: 13,042; failed: 5; retained successful IDX/message bytes: 2,373,648,569. Retry/failure network overhead was not instrumented, so this is a measured lower bound rather than an exact wire-byte count.
- Fallback origins: Train 0; Validation 27.
- Compact NWP point artifacts: 536,985 bytes; prepared learning artifact: 14,477,756 bytes; preparation/download/extraction wall time 725.8 s.
- DSWRF is interval-average support; instantaneous TCDC is interpolated only within the selected cycle. Future measured GHI is metric-side only; only Train GHI forms the frozen reliability prior.

## Validation full-timeline RMSE, mean ± sample SD across seeds

| Model | H12 RMSE | H48 RMSE | H96 RMSE | H144 RMSE | Parameters |
|---|---:|---:|---:|---:|---:|
| HISTORY_ONLY | 10.704812 ± 5.664075 | 9.362516 ± 4.342525 | 7.143717 ± 3.050568 | 6.306359 ± 2.460753 | 683,856 |
| RAW_NWP | 15.047775 ± 5.131024 | 12.163703 ± 6.367384 | 9.616998 ± 5.208228 | 9.021008 ± 5.654234 | 683,937 |
| AGE_LEAD_RELIABILITY | 10.775605 ± 9.249075 | 9.102221 ± 7.237071 | 7.096861 ± 5.285892 | 6.877635 ± 4.808899 | 683,978 |

RAW_NWP relative to HISTORY_ONLY: {"12": -40.570194128921464, "48": -29.919175410180276, "96": -34.62176883120357, "144": -43.046219594572385} percent improvement by horizon.

AGE_LEAD_RELIABILITY relative to RAW_NWP: {"12": 28.390709006125253, "48": 25.168996977866918, "96": 26.205030143169516, "144": 23.759802987130172} percent improvement by horizon.

M2 parameter increase over RAW_NWP: 0.005995%.

## H144 scope metrics, mean ± sample SD

| Model | Scope | RMSE (kW) | MAE (kW) | Train-range nRMSE | R² |
|---|---|---:|---:|---:|---:|
| HISTORY_ONLY | regular_full_timeline | 6.306359 ± 2.460753 | 1.394438 ± 0.175124 | 1.042856 ± 0.406924 | -10.423733 ± 8.499997 |
| HISTORY_ONLY | daylight | 7.280525 ± 3.015807 | 1.986358 ± 0.196145 | 1.203949 ± 0.498711 | -16.487454 ± 13.296702 |
| HISTORY_ONLY | high_change_daylight | 10.014174 ± 4.485534 | 2.369615 ± 0.402722 | 1.656002 ± 0.741754 | -49.296573 ± 39.901199 |
| RAW_NWP | regular_full_timeline | 9.021008 ± 5.654234 | 1.540699 ± 0.658680 | 1.491766 ± 0.935017 | -25.779440 ± 31.225231 |
| RAW_NWP | daylight | 7.962803 ± 3.084689 | 1.588916 ± 0.334485 | 1.316775 ± 0.510102 | -19.649360 ± 14.747862 |
| RAW_NWP | high_change_daylight | 10.569876 ± 3.368504 | 2.094796 ± 0.423317 | 1.747896 ± 0.557035 | -51.769358 ± 29.884007 |
| AGE_LEAD_RELIABILITY | regular_full_timeline | 6.877635 ± 4.808899 | 1.376919 ± 0.439863 | 1.137325 ± 0.795227 | -15.355446 ± 20.722533 |
| AGE_LEAD_RELIABILITY | daylight | 6.323313 ± 4.179925 | 1.630657 ± 0.202124 | 1.045659 ± 0.691217 | -14.285649 ± 18.586456 |
| AGE_LEAD_RELIABILITY | high_change_daylight | 8.296442 ± 5.615160 | 1.971342 ± 0.502190 | 1.371947 ± 0.928555 | -38.747707 ± 49.006638 |

## H144 per-seed direction

| Seed | HISTORY_ONLY H144 RMSE | RAW_NWP H144 RMSE | RAW change | AGE_LEAD H144 RMSE | AGE_LEAD vs RAW |
|---:|---:|---:|---:|---:|---:|
| 42 | 8.938526 | 15.515882 | -73.584% | 12.428599 | 19.898% |
| 43 | 5.917139 | 5.196662 | 12.176% | 4.227050 | 18.658% |
| 44 | 4.063412 | 6.350482 | -56.284% | 3.977254 | 37.371% |

## Run and efficiency record

| Model | Seed | Best epoch | Stop | Epoch seconds | Peak GPU MB |
|---|---:|---:|---|---:|---:|
| AGE_LEAD_RELIABILITY | 42 | 1 | early_stopping | 11.10 | 106.4 |
| AGE_LEAD_RELIABILITY | 43 | 7 | early_stopping | 11.10 | 107.5 |
| AGE_LEAD_RELIABILITY | 44 | 9 | early_stopping | 11.31 | 105.3 |
| HISTORY_ONLY | 42 | 5 | early_stopping | 10.56 | 97.2 |
| HISTORY_ONLY | 43 | 3 | early_stopping | 10.57 | 99.8 |
| HISTORY_ONLY | 44 | 7 | early_stopping | 10.75 | 100.2 |
| RAW_NWP | 42 | 1 | early_stopping | 11.17 | 102.5 |
| RAW_NWP | 43 | 1 | early_stopping | 11.28 | 102.1 |
| RAW_NWP | 44 | 1 | early_stopping | 10.77 | 101.8 |

Device: `NVIDIA GeForce RTX 3060 Laptop GPU`. Per-seed inference latency and all scope metrics are retained in `metrics_per_seed.csv`; local checkpoints and Validation prediction arrays remain under `results/` and are not committed.

## Pre-registered interpretation

- RAW_NWP rule satisfied: **False**. H144 improvement=-43.046%; seed directions=[False, True, False]; improved horizons=0/4; worst change=-43.046%.
- AGE_LEAD_RELIABILITY rule satisfied: **True**. H144 improvement=23.760%; seed directions=[True, True, True]; improved horizons=4/4; worst change=23.760%; parameter increase=0.005995%.

Although M2 consistently reduces the very poor RAW_NWP errors, the pre-registered first-stage information criterion fails. M2 therefore cannot rescue this screen. The unusually large cross-year errors and seed variance limit the inference to this fixed implementation, 1.0° product, 3-hour forecast grid, and 2021→2022 split; they do not prove that GFS radiation/cloud forecasts are physically uninformative in every operational design. No additional variables, 2023 download, or v2/v3 model is justified under the registered stopping rule.

The decision is limited to Validation. No 2023 NWP was downloaded and no 2023 prediction, error, threshold or metric was produced.


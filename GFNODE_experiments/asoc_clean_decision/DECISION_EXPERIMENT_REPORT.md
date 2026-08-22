# Clean time-conditioned GFNODE decision experiment

## Fixed protocol

- Full 5-minute clock retained; missing source rows are reindexed and marked.
- KNN imputer, Isolation Forest, feature scaler, and target scaler are fit only on Train.
- Windows are built independently inside Train, Validation, and Test; lookback=72 and horizon=144.
- Validation MSE alone selects each checkpoint. Test is evaluated only after checkpoint selection.
- Predefined daylight rule: Global_Horizontal_Radiation >= 20.0.
- Parameters: GFNODE=98,738; Discrete=98,802; difference=0.065%.

## Split and sample summary

```json
{
  "Sanyo": {
    "dataset": "Sanyo",
    "raw_timestamp_range": [
      "2018-04-01 00:00:00",
      "2018-08-31 23:55:00"
    ],
    "duplicate_source_timestamps": 0,
    "feature_count_before_masks": 7,
    "model_input_feature_count": 15,
    "target_scaler_train_range": [
      0.0,
      6.05786705
    ],
    "target_range_for_nrmse": 6.05786705,
    "fit_log": [
      {
        "preprocessor": "KNNImputer",
        "operation": "fit",
        "split": "train"
      },
      {
        "preprocessor": "IsolationForest",
        "operation": "fit",
        "split": "train"
      },
      {
        "preprocessor": "feature_MinMaxScaler",
        "operation": "fit",
        "split": "train"
      },
      {
        "preprocessor": "target_MinMaxScaler",
        "operation": "fit",
        "split": "train"
      }
    ],
    "splits": {
      "train": {
        "timestamp_range": [
          "2018-04-01 00:00:00",
          "2018-07-15 23:55:00"
        ],
        "regular_rows": 30528,
        "source_timestamp_missing_rows": 0,
        "valid_windows": 20409,
        "valid_evaluation_targets": 2938896,
        "daylight_evaluation_targets": 1230308
      },
      "validation": {
        "timestamp_range": [
          "2018-07-16 00:00:00",
          "2018-08-07 23:55:00"
        ],
        "regular_rows": 6624,
        "source_timestamp_missing_rows": 0,
        "valid_windows": 3619,
        "valid_evaluation_targets": 521136,
        "daylight_evaluation_targets": 213896
      },
      "test": {
        "timestamp_range": [
          "2018-08-08 00:00:00",
          "2018-08-31 23:55:00"
        ],
        "regular_rows": 6912,
        "source_timestamp_missing_rows": 0,
        "valid_windows": 4177,
        "valid_evaluation_targets": 601488,
        "daylight_evaluation_targets": 263547
      }
    }
  },
  "Qcells": {
    "dataset": "Qcells",
    "raw_timestamp_range": [
      "2018-04-01 00:00:00",
      "2018-08-31 23:55:00"
    ],
    "duplicate_source_timestamps": 0,
    "feature_count_before_masks": 7,
    "model_input_feature_count": 15,
    "target_scaler_train_range": [
      0.0,
      6.036834717
    ],
    "target_range_for_nrmse": 6.036834717,
    "fit_log": [
      {
        "preprocessor": "KNNImputer",
        "operation": "fit",
        "split": "train"
      },
      {
        "preprocessor": "IsolationForest",
        "operation": "fit",
        "split": "train"
      },
      {
        "preprocessor": "feature_MinMaxScaler",
        "operation": "fit",
        "split": "train"
      },
      {
        "preprocessor": "target_MinMaxScaler",
        "operation": "fit",
        "split": "train"
      }
    ],
    "splits": {
      "train": {
        "timestamp_range": [
          "2018-04-01 00:00:00",
          "2018-07-15 23:55:00"
        ],
        "regular_rows": 30528,
        "source_timestamp_missing_rows": 0,
        "valid_windows": 12747,
        "valid_evaluation_targets": 1835568,
        "daylight_evaluation_targets": 731400
      },
      "validation": {
        "timestamp_range": [
          "2018-07-16 00:00:00",
          "2018-08-07 23:55:00"
        ],
        "regular_rows": 6624,
        "source_timestamp_missing_rows": 0,
        "valid_windows": 2999,
        "valid_evaluation_targets": 431856,
        "daylight_evaluation_targets": 171791
      },
      "test": {
        "timestamp_range": [
          "2018-08-08 00:00:00",
          "2018-08-31 23:55:00"
        ],
        "regular_rows": 6912,
        "source_timestamp_missing_rows": 0,
        "valid_windows": 3019,
        "valid_evaluation_targets": 434736,
        "daylight_evaluation_targets": 187271
      }
    }
  }
}
```

## Three-seed mean ± SD (regular and daylight metrics)

| Dataset | Model | Scope | H | RMSE mean±SD | MAE mean±SD | R² mean±SD | nRMSE mean±SD |
|---|---|---|---:|---|---|---|---|
| Qcells | Discrete | predefined_daylight | 12 | 0.16020 ± 0.02792 | 0.14684 ± 0.03060 | -15.71343 ± 5.96037 | 0.02654 ± 0.00462 |
| Qcells | Discrete | predefined_daylight | 48 | 1.55839 ± 0.01986 | 1.25131 ± 0.01870 | -1.00152 ± 0.05096 | 0.25815 ± 0.00329 |
| Qcells | Discrete | predefined_daylight | 96 | 1.70186 ± 0.00784 | 1.35012 ± 0.00890 | -0.11519 ± 0.01028 | 0.28191 ± 0.00130 |
| Qcells | Discrete | predefined_daylight | 144 | 1.32659 ± 0.02509 | 1.01614 ± 0.04450 | 0.26499 ± 0.02774 | 0.21975 ± 0.00416 |
| Qcells | Discrete | regular_full_timeline | 12 | 0.09095 ± 0.02086 | 0.07247 ± 0.01859 | -543.94958 ± 241.54179 | 0.01507 ± 0.00346 |
| Qcells | Discrete | regular_full_timeline | 48 | 0.51338 ± 0.00640 | 0.20096 ± 0.00764 | 0.27139 ± 0.01813 | 0.08504 ± 0.00106 |
| Qcells | Discrete | regular_full_timeline | 96 | 0.89359 ± 0.00451 | 0.43117 ± 0.01102 | 0.66384 ± 0.00339 | 0.14802 ± 0.00075 |
| Qcells | Discrete | regular_full_timeline | 144 | 0.88537 ± 0.01704 | 0.49243 ± 0.02583 | 0.79064 ± 0.00805 | 0.14666 ± 0.00282 |
| Qcells | GFNODE | predefined_daylight | 12 | 0.18681 ± 0.00534 | 0.16335 ± 0.00362 | -21.28968 ± 1.27905 | 0.03095 ± 0.00088 |
| Qcells | GFNODE | predefined_daylight | 48 | 1.52262 ± 0.11345 | 1.23063 ± 0.10723 | -0.91756 ± 0.28629 | 0.25222 ± 0.01879 |
| Qcells | GFNODE | predefined_daylight | 96 | 1.60278 ± 0.10862 | 1.29523 ± 0.08774 | 0.00786 ± 0.13621 | 0.26550 ± 0.01799 |
| Qcells | GFNODE | predefined_daylight | 144 | 1.22370 ± 0.05896 | 0.92328 ± 0.05087 | 0.37377 ± 0.06103 | 0.20271 ± 0.00977 |
| Qcells | GFNODE | regular_full_timeline | 12 | 0.13569 ± 0.03017 | 0.10116 ± 0.02250 | -1209.39421 ± 497.06979 | 0.02248 ± 0.00500 |
| Qcells | GFNODE | regular_full_timeline | 48 | 0.51440 ± 0.02487 | 0.22807 ± 0.01319 | 0.26741 ± 0.07140 | 0.08521 ± 0.00412 |
| Qcells | GFNODE | regular_full_timeline | 96 | 0.86114 ± 0.04311 | 0.45169 ± 0.01074 | 0.68730 ± 0.03171 | 0.14265 ± 0.00714 |
| Qcells | GFNODE | regular_full_timeline | 144 | 0.83072 ± 0.03225 | 0.47854 ± 0.02151 | 0.81555 ± 0.01439 | 0.13761 ± 0.00534 |
| Sanyo | Discrete | predefined_daylight | 12 | 0.29693 ± 0.06266 | 0.22525 ± 0.06562 | 0.96552 ± 0.01406 | 0.04902 ± 0.01034 |
| Sanyo | Discrete | predefined_daylight | 48 | 1.17973 ± 0.05390 | 0.80689 ± 0.08134 | 0.53180 ± 0.04305 | 0.19474 ± 0.00890 |
| Sanyo | Discrete | predefined_daylight | 96 | 1.51383 ± 0.01425 | 1.12396 ± 0.04658 | 0.18419 ± 0.01534 | 0.24990 ± 0.00235 |
| Sanyo | Discrete | predefined_daylight | 144 | 1.29335 ± 0.02334 | 0.96240 ± 0.01817 | 0.36862 ± 0.02275 | 0.21350 ± 0.00385 |
| Sanyo | Discrete | regular_full_timeline | 12 | 0.14615 ± 0.03867 | 0.08881 ± 0.03830 | 0.99076 ± 0.00485 | 0.02412 ± 0.00638 |
| Sanyo | Discrete | regular_full_timeline | 48 | 0.59237 ± 0.02846 | 0.24720 ± 0.03088 | 0.86107 ± 0.01344 | 0.09778 ± 0.00470 |
| Sanyo | Discrete | regular_full_timeline | 96 | 0.89278 ± 0.00989 | 0.42939 ± 0.02226 | 0.77276 ± 0.00502 | 0.14738 ± 0.00163 |
| Sanyo | Discrete | regular_full_timeline | 144 | 0.87194 ± 0.01251 | 0.46165 ± 0.00557 | 0.82093 ± 0.00514 | 0.14394 ± 0.00207 |
| Sanyo | GFNODE | predefined_daylight | 12 | 0.40916 ± 0.09717 | 0.33434 ± 0.09154 | 0.93403 ± 0.03051 | 0.06754 ± 0.01604 |
| Sanyo | GFNODE | predefined_daylight | 48 | 1.24898 ± 0.08504 | 0.83491 ± 0.08250 | 0.47433 ± 0.07089 | 0.20618 ± 0.01404 |
| Sanyo | GFNODE | predefined_daylight | 96 | 1.66255 ± 0.16057 | 1.23800 ± 0.14462 | 0.00996 ± 0.19471 | 0.27445 ± 0.02651 |
| Sanyo | GFNODE | predefined_daylight | 144 | 1.40108 ± 0.11774 | 1.04203 ± 0.09696 | 0.25573 ± 0.12591 | 0.23128 ± 0.01944 |
| Sanyo | GFNODE | regular_full_timeline | 12 | 0.22359 ± 0.04730 | 0.15302 ± 0.02769 | 0.97871 ± 0.00837 | 0.03691 ± 0.00781 |
| Sanyo | GFNODE | regular_full_timeline | 48 | 0.63916 ± 0.04680 | 0.29828 ± 0.04002 | 0.83793 ± 0.02339 | 0.10551 ± 0.00773 |
| Sanyo | GFNODE | regular_full_timeline | 96 | 0.98723 ± 0.09179 | 0.50785 ± 0.05754 | 0.72056 ± 0.05277 | 0.16297 ± 0.01515 |
| Sanyo | GFNODE | regular_full_timeline | 144 | 0.95409 ± 0.07913 | 0.53292 ± 0.05259 | 0.78465 ± 0.03579 | 0.15750 ± 0.01306 |

## Pre-specified route decision

- Qcells: H144 GFNODE-vs-Discrete relative RMSE improvement = 6.172%; seedwise wins = 3/3.
- Sanyo: H144 GFNODE-vs-Discrete relative RMSE improvement = -9.421%; seedwise wins = 0/3.

| Criterion | Result |
|---|---|
| both_datasets_H144_better | FAIL |
| at_least_one_H144_improves_3_percent | PASS |
| other_dataset_not_worse_than_1_percent | FAIL |
| long_horizon_not_more_than_1pp_weaker_than_H12 | PASS |
| not_single_seed_effect | FAIL |
| no_numerical_divergence | PASS |

**Final decision: FAIL**

A FAIL means do not tune ODE depth, width, step, activation, or time encoding again in this project stage. The permitted next step is to replace the ODE-centered route, not to create a GFNODE v2/v3.

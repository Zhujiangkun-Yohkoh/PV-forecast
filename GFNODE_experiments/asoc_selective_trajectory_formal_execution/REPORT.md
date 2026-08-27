# Scheme C1 final closeout

## Final status

**C1_ROUTE_CLOSED_DATA_UNAVAILABLE**

- Data readiness: `C1_FORMAL_DATA_FAIL`
- Termination reason: `REQUIRED_FULL_YEAR_1_SECOND_DATA_UNAVAILABLE_OR_UNCONFIRMED`
- Scientific method outcome: `NOT_EVALUATED`
- Implementation status: `C1_FORMAL_IMPLEMENTATION_NOT_VALIDATED_AND_NO_LONGER_REQUIRED`
- GPU training performed: no
- Completed runs: 0/9
- Future GPU execution authorized: no

Scheme C1 is closed because the preregistered complete-year one-second irradiance inputs cannot be obtained or confirmed from the current official source. This is a data-executability conclusion, not evidence that selective risk control or the base forecaster succeeds or fails.

## Read-only source facts

All time semantics use the authoritative UTC field. No Local field was substituted, and no interpolation, year substitution, manual repair, or raw-file rewrite was performed.

| Year | Selected source | Bytes | Target-year unique seconds | UTC range represented | Missing target-year seconds | Structural/cross-year findings |
|---|---|---:|---:|---|---:|---|
| 2021 | `C:\Users\Zhujiangkun-Yohkoh\Desktop\光伏项目_最新\PV_improve_v1\原始Dataset\高分辨率气象数据集\C1_fresh_downloads\2021\fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2021.csv` | 1,726,481,334 | 18,403,200 | 2021-06-02 00:00:00 to 2021-12-31 23:59:59 UTC | 13,132,800 | No structural anomaly detected, but January through 1 June are absent. |
| 2022 | `C:\Users\Zhujiangkun-Yohkoh\Desktop\光伏项目_最新\PV_improve_v1\GFNODE_experiments\asoc_multirate_redownload_validation\fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2022.csv` | 3,011,901,400 | 31,536,000 | 2022-01-01 00:00:00 to 2022-12-31 23:59:59 UTC | 0 | Complete verified reference year. |
| 2023 | `C:\Users\Zhujiangkun-Yohkoh\Desktop\光伏项目_最新\PV_improve_v1\原始Dataset\高分辨率气象数据集\C1_fresh_downloads\2023\fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2023.csv` | 1,774,300,567 | 167,952 | Target-year records from 2023-01-01 00:00:01 to 2023-01-02 22:39:12 UTC | 31,368,048 | 18,489,565 out-of-year records; 212 column anomalies, including 1 glued and 211 truncated/Data Error records; later content includes 2024/2025. |

The one-second channel audit recorded these missing-value counts and five-minute-bin states:

- 2021: MB0/MB1/MB2 missing values 781,177 / 781,579 / 781,759; 58,711 strict three-channel bins, 2,633 partially missing bins, 43,776 empty bins.
- 2022: MB0/MB1/MB2 missing values 3,992 / 3,236 / 3,320; 99,211 strict bins, 5,909 partially missing bins, 0 empty bins.
- 2023: MB0/MB1/MB2 missing values 59 / 58 / 42; 464 strict bins, 96 partially missing bins, 104,560 empty bins.

The official download page does not establish guaranteed complete-year one-second coverage. The user repeated the official downloads, but the resulting 2021 and 2023 files retained the same unusable coverage pattern. Consequently, completeness is unavailable or unconfirmed rather than safely repairable.

## Five-stage window evidence

The strict five-stage protocol produced the following three-array common legal origins before model execution:

| Stage | Expected calendar origins | Common legal origins | Strict 300/300 origins | First legal origin | Last legal origin | Coverage |
|---|---:|---:|---:|---|---|---|
| BASE_TRAIN | 105,037 | 58,122 | 54,067 | 2021-06-02 15:25 ACST | 2021-12-31 22:55 ACST | Months 6–12; spring, summer, winter |
| BASE_MODEL_VALIDATION | 34,477 | 31,750 | 31,315 | 2022-01-01 05:55 ACST | 2022-04-30 22:55 ACST | Months 1–4 |
| RISK_FIT | 35,341 | 32,509 | 31,676 | 2022-05-01 05:55 ACST | 2022-08-31 22:55 ACST | Months 5–8 |
| RISK_CALIBRATION | 35,053 | 34,969 | 2,300 | 2022-09-01 05:55 ACST | 2022-12-31 22:55 ACST | Months 9–12 |
| FINAL_TEST | 105,037 | 604 | **0** | 2023-01-01 05:55 ACST | 2023-01-03 08:10 ACST | January only; summer only |

The preregistered BASE_TRAIN and FINAL_TEST full-year requirements are therefore not met. In particular, FINAL_TEST has zero strict three-channel-complete origins and cannot support the preregistered annual confirmation.

## Unexecuted implementation limitations

The S4 implementation passed only local synthetic fixtures. It was never validated by a production training-and-evaluation execution, and it is no longer required because the data condition permanently blocks this project route. Known limitations are retained to prevent the code from being mistaken for GPU-ready software:

- The production Dataset yields `[B,72,14]`, while the current `Conv1d` path requires an explicit conversion to `[B,14,72]`.
- The random seed must be set before model initialization.
- The Final-Test payload is not yet truly delayed in materialization.
- Bootstrap computation is not connected to the formal execution chain.
- Formal metric persistence has not been completely validated.

These limitations are not repaired in this closeout and do not constitute evidence about scientific method performance.

## Execution and integrity statement

- No GPU, CPU, or synthetic training was executed in this closeout.
- No optimizer, backward pass, risk-model fit, checkpoint creation, Final-Test prediction, prediction-error access, coverage calculation, or AURC calculation occurred.
- Performance metrics are unavailable; all nine preregistered runs remain `NOT_RUN`.
- Selected PV and irradiance source byte sizes and `mtime_ns` remained unchanged during the closeout checks.
- No raw source was interpolated, repaired, combined with a substitute year, or rewritten.

## Closure interpretation

1. C1 closure is a data-executability conclusion.
2. It is not a failure result for the risk-control method.
3. It is not a failure result for the base trajectory forecaster.
4. It cannot be used as a method-performance comparison in a manuscript.
5. Re-downloading the same one-second source is not recommended.
6. If complete official annual data become available in the future, the work would require a newly proposed and preregistered study; this closed execution must not simply be resumed.
7. Current research resources return to the Scheme A manuscript submission.

No further C1 training, repair, or variant development is authorized.

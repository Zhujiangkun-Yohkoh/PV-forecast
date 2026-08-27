# Scheme C1 final archival closeout

## Terminal state

**C1_ROUTE_CLOSED_DATA_UNAVAILABLE**

- Data readiness: `C1_FORMAL_DATA_FAIL`
- Scientific method outcome: `NOT_EVALUATED`
- Implementation status: `C1_FORMAL_IMPLEMENTATION_NOT_VALIDATED_AND_NO_LONGER_REQUIRED`
- Future GPU execution authorized: `false`
- Completed runs: 0
- Expected runs: 9

The preregistered C1 execution is administratively closed under the currently obtainable official data snapshot. This is not a method-performance failure and is not a base-forecaster failure.

## Source-data evidence

The source data are incomplete or unconfirmed; therefore, the preregistered execution is unavailable and the scientific method remains unevaluated. No interpolation, repair, year substitution, or raw-file rewrite was performed.

| Year | Data file | Bytes | Target-year unique seconds | UTC coverage | Missing target-year seconds | Structural evidence |
|---|---|---:|---:|---|---:|---|
| 2021 | `fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2021.csv` | 1,726,481,334 | 18,403,200 | 2021-06-02 00:00:00 to 2021-12-31 23:59:59 | 13,132,800 | No recorded structural anomaly, but the first 152 days are absent. |
| 2022 | `fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2022.csv` | 3,011,901,400 | 31,536,000 | 2022-01-01 00:00:00 to 2022-12-31 23:59:59 | 0 | Complete verified reference year. |
| 2023 | `fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2023.csv` | 1,774,300,567 | 167,952 | Target-year records from 2023-01-01 00:00:01 to 2023-01-02 22:39:12 | 31,368,048 | 18,489,565 out-of-year records; 212 column anomalies, including 1 glued record and 211 truncated/Data Error records; later content includes 2024/2025. |

All three target years are non-leap years with 31,536,000 expected seconds. The saved audit arithmetic reconciles exactly:

- 2021: 18,403,200 + 13,132,800 = 31,536,000.
- 2023: 167,952 + 31,368,048 = 31,536,000.
- Five-minute bins: 2021 has 58,711 strict + 2,633 partial + 43,776 empty = 105,120; 2022 has 99,211 + 5,909 + 0 = 105,120; 2023 has 464 + 96 + 104,560 = 105,120.

The official public pages do not guarantee a complete annual one-second export. Repeated user downloads produced the same unusable temporal coverage. Re-downloading the same files is not recommended.

## Five-stage evidence

| Stage | Expected origins | Three-array common legal origins | Strict 300/300 common origins | Temporal coverage |
|---|---:|---:|---:|---|
| BASE_TRAIN | 105,037 | 58,122 | 54,067 | 2021-06-02 15:25 to 2021-12-31 22:55 ACST |
| BASE_MODEL_VALIDATION | 34,477 | 31,750 | 31,315 | 2022-01-01 05:55 to 2022-04-30 22:55 ACST |
| RISK_FIT | 35,341 | 32,509 | 31,676 | 2022-05-01 05:55 to 2022-08-31 22:55 ACST |
| RISK_CALIBRATION | 35,053 | 34,969 | 2,300 | 2022-09-01 05:55 to 2022-12-31 22:55 ACST |
| FINAL_TEST | 105,037 | 604 | **0** | 2023-01-01 05:55 to 2023-01-03 08:10 ACST |

FINAL_TEST strict three-channel common origins equal zero. BASE_TRAIN and FINAL_TEST also fail the preregistered complete-year coverage requirement.

## Unexecuted implementation limitations

The historical S4 code passed only local fixtures and was never validated through a real end-to-end execution. Known limitations are preserved as archival warnings:

- The Dataset produces `[B,72,14]`, while the `Conv1d` path expects `[B,14,72]` without the required explicit conversion.
- The seed is set after model initialization in the historical execution path.
- The Final-Test payload is materialized earlier than the intended delayed-access boundary.
- Bootstrap is not connected to the formal execution chain.
- Formal metric persistence was not validated.
- A bootstrap field named `mean` actually contains the median.
- Some historical tests are placeholders rather than production integration tests.

These defects will not be repaired because the route is closed. The archived pipeline is an unvalidated prototype and is not authorized for execution.

## Reproducible closure behavior

- `config.json` fixes `route_closed=true` and `route_status=C1_ROUTE_CLOSED_DATA_UNAVAILABLE`.
- Both public CLI modes return the same stable terminal-state JSON before audit, data preparation, model creation, loader creation, optimizer construction, risk fitting, or Final-Test access.
- The formal execution function independently refuses execution while the route is closed.
- Closeout tests are side-effect free and verify that the terminal JSON, report, and both status CSVs remain byte-for-byte unchanged.
- The nine array-seed rows are execution-status records only. No RMSE, MAE, coverage, AURC, or other performance value exists.

## Final interpretation

1. C1 is closed because the preregistered data execution is unavailable.
2. The risk-control method was not evaluated.
3. The base trajectory forecaster was not evaluated under this formal C1 protocol.
4. These closeout records cannot support a manuscript method-performance comparison.
5. If a new complete official dataset becomes available, it would require a new research proposal and a new preregistration; this closed execution must not be resumed.
6. No further C1 experiment, implementation work, or data download is authorized.
7. Research resources return to the Scheme A JRSE manuscript final refinement and author sign-off.

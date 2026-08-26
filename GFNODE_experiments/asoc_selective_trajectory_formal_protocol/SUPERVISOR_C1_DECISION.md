# Supervisor Decision — Scheme C1-S1

## Decision

**`AUTHORIZE_STRICT_C1_FORMAL_PROTOCOL`**

This authorizes the five-stage study design, not execution. The GPU decision is **`NOT_AUTHORIZED_UNTIL_DATA_AND_PROTOCOL_CONFIRMED`**.

## Why formal design is justified but publication is not

C1-S0R corrected a real calibration-scope error: daylight thresholds are now computed from the daylight subset of Risk-Calibration, while full-timeline thresholds use the full subset. Its Site 17/2022 screen shows substantial oracle headroom, strong past-only risk ranking, calibrated selective improvement, and matched-Persistence skill across three seeds. No reviewed paper in the existing C1 literature matrix completely occupies the exact combination of PV multi-horizon trajectories, past-only risk scoring, explicit abstention, calibrated risk control, and matched-coverage Persistence. This is enough to define one confirmatory study.

It is not yet publishable evidence. The base checkpoint used the full original Validation period; the risk estimator and threshold were then derived from halves of that same period, and the 2022 Test was inspected during method selection. The evidence is a feasibility screen, not a genuinely independent confirmation. The formal study therefore needs five chronologically disjoint roles: Base-Train, Base-Model-Validation, Risk-Fit, Risk-Calibration, and Final-Test.

## C1-S0R implementation review

| Item | Finding | Formal consequence |
|---|---|---|
| Scope-matched calibration | Correct after S0R: calibration membership is intersected with the matching full/daylight Validation mask before quantiles | Retain; thresholds remain scope-specific |
| `risk_fit_fraction` | Config says 0.5, but code uses `len(val_origins) // 2`; the config field does not control behavior | Formal implementation must honor it or remove it; calendar stages are authoritative |
| High-change quantile | The code calls `np.quantile(train_recent, 0.9)` directly; 0.9 is not read from config | Put the quantile in config and estimate it only in the authorized fit stage |
| Capacity/daylight consistency | Site 17 uses 6.3 kW and 0.063 kW, exactly 1%; consistent for Sanyo | Verify Site 25/38 references and use 0.0583/0.0590 kW only after unit confirmation |
| Base architecture | The class named `ModernTCN` is a compact depthwise-convolution TCN, not the official complete ModernTCN | Formal name: `DEPTHWISE_TCN_TRAJECTORY`; do not claim ModernTCN novelty or full implementation |
| Test isolation | Quantile API and array-memory checks are helpful but weak; they do not adversarially prove that future/Test values cannot influence features or threshold | Add future-sentinel and Final-Test score mutation tests before execution |
| Feature causality | Maximum tracked source timestamp ≤ origin is checked, but no future-sentinel mutation test exists | Require element-identical features after modifying all post-origin inputs |
| Accepted-RMSE comparison | At nominal 80%, Full, Recent-variation, and disagreement realized coverages differ materially; direct accepted-RMSE percentages are not coverage-matched | Use AURC as primary incremental metric; prohibit direct RMSE superiority if coverage differs by >2 points |
| Bootstrap | Natural-day clusters preserve within-day data but H12 windows overlap across midnight and adjacent days remain dependent | Use continuous 7-day moving blocks as primary; natural-day clusters only as sensitivity analysis |

## Final-Test qualification

The 2023 five-minute PV exports exist for Sites 17, 25, and 38, span both calendar endpoints, and retain explicit gaps/missing values. Their common valid timestamp set contains 91,296 points and approximately 69,875 conservative L72+H144 continuous windows, so the PV side is viable under segmented construction.

No Git-history evidence was found that 2023/2024 was used for a C1 risk score, C1 threshold, realized selective coverage, selective RMSE/MAE, risk–coverage curve, hyperparameter choice, or C1 decision. Thus 2023 qualifies as a **C1-method-unseen candidate Final Test**, not as globally untouched.

The blocker is irradiance. The local file labelled 2023 second-level irradiance covers only 560 five-minute bins in early January before a malformed transition and an approximately 446-day jump. It is not a Final-Test source. The validated complete second-level file exists only for 2022. The available 2024 fragments do not provide a cleaner unique fallback.

## Single authorized time plan

- 2021 calendar year: `BASE_TRAIN`
- 2022-01-01 to 2022-04-30: `BASE_MODEL_VALIDATION`
- 2022-05-01 to 2022-08-31: `RISK_FIT`
- 2022-09-01 to 2022-12-31: `RISK_CALIBRATION`
- 2023 calendar year: `FINAL_TEST`

All windows are constructed independently inside continuous valid segments. Nothing crosses a gap or stage boundary. All preprocessing is fitted on 2021; the base checkpoint uses only early 2022; risk fitting uses only middle 2022; thresholds use only late 2022; 2023 is opened once after implementation and criteria are fixed.

## Required user action and workload

The **single required action** is to download authoritative full-calendar UTC second-level Alice Springs irradiance exports with MB0/MB1/MB2 for **2021 and 2023**. Expected size is approximately 3.0 GB per year, about 6.0 GB total, based on the verified 2022 file. Existing five-minute PV files do not need replacement for this protocol; their gaps will remain explicit.

Once those two files pass read-only validation, the formal deep matrix is fixed at **9 runs**: three arrays × seeds 42/43/44. No new deep baseline, hyperparameter search, C1 v2/v3, dynamic threshold, probabilistic interval, or NODE is authorized.

## Recommendation

Preserve the formal protocol and Draft PR, obtain the two missing irradiance years, and run a final read-only data/protocol confirmation. Do not start GPU training before that confirmation. If the data pass, the next authorized action can be the single preregistered nine-run experiment. If the Final-Test criteria fail, close C1 without redesign; Scheme A remains the fallback.

No neural network or risk estimator was trained in S1. No candidate Final-Test C1 prediction error was accessed.


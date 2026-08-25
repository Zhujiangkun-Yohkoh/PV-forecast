# Clean deterministic PV benchmark/application manuscript blueprint

## Stage decision

**READY_AFTER_NONTRAINING_COMPLETION**

The project already contains a coherent, leakage-free core benchmark: four completed direct-H144 models, three co-located Alice Springs PV arrays, three fixed seeds, common split and checkpoint semantics, and 36 saved H144 prediction files. This is enough to support an application/benchmark paper, but not yet enough to start final prose without qualification. The remaining mandatory gaps are non-neural: authoritative metadata reconciliation, one unified metric recomputation, a persistence skill reference, consistent latency measurement, and reconstruction of every table and figure from clean artifacts.

No new neural-network training is required for the fast manuscript route. ModernTCN is an evaluated existing method, not the paper's invention.

## 1. Recommended journal route

### Target: Solar Energy Advances

The official scope explicitly includes solar-energy measurements, monitoring protocols, data analytics, artificial intelligence in solar systems, and forecasting. It is therefore the best match for a protocol-led solar forecasting study that does not claim a new neural architecture. The manuscript must still articulate a transferable knowledge gap: leakage-free evaluation changes what can be defended about model accuracy, and co-located arrays expose technology/horizon dependence under identical weather and time boundaries.

Official page: https://www.sciencedirect.com/journal/solar-energy-advances

Main risk: a manuscript framed as “we applied four known models at one site” remains desk-rejectable. The contribution must be the reproducible evaluation design, the paired co-located comparison, and the trajectory/information-boundary findings—not ModernTCN.

### Backup 1: Journal of Renewable and Sustainable Energy

The AIP journal is a broad renewable-energy engineering venue and its official platform publishes photovoltaic forecasting research. The fit is credible, but recent forecasting papers are methodologically competitive; the protocol and transferable application insights must be prominent.

Official page: https://pubs.aip.org/aip/jrse

### Backup 2: IET Renewable Power Generation

The official scope expressly covers photovoltaics, forecasting, validated modelling and renewable-power operation. It also says most papers are expected to have significant novelty of approach or application with general applicability, making it the highest desk-rejection risk of the three.

Official page: https://ietresearch.onlinelibrary.wiley.com/journal/17521424

The official IET page confirms Scopus and SCIE indexing. Solar Energy Advances' current JCR/SCIE status and Journal of Renewable and Sustainable Energy's current official indexing record were not conclusively accessible in this audit and must be checked manually before submission. No unverified impact factor or review duration should enter the cover letter.

## 2. Unique research question

> Under a timestamp-faithful, leakage-free evaluation protocol, how do established direct multi-step forecasting models compare across co-located PV arrays, and how do forecast horizon, array technology, and high-variation regimes shape their remaining trajectory errors?

This question is intentionally limited. It does not concern Neural ODEs, a new decoder, probability intervals, Ramp-event prediction, high-frequency feature innovation, or cross-climate generalization.

## 3. Recommended titles

### Preferred

**Leakage-Free Direct Multi-Step Forecasting Across Co-Located Photovoltaic Arrays: A Reproducible Benchmark and Trajectory Error Analysis**

### Alternatives

1. **Reassessing Multi-Horizon Photovoltaic Power Forecasting Under a Timestamp-Faithful Evaluation Protocol**
2. **Direct Multi-Step Forecasting for Co-Located Photovoltaic Technologies: Benchmarking, Error Growth, and Information Limits**

## 4. Article positioning

The paper should be presented as a reproducible solar forecasting benchmark and engineering evaluation. The primary study uses the 2018 co-located Sanyo, Hanwha and Q CELLS arrays, common weather features, a regular 5-minute timeline, a 72-step lookback and a single direct H144 forecast. It compares ModernTCN, iTransformer, PatchTST and the existing Discrete Candidate across seeds 42/43/44. A secondary Site 17/2022 case study explains lead-time error growth and trajectory smoothing, while the past-second irradiance experiment defines an information boundary. Claims remain limited to the tested site, arrays, periods, inputs and implementations.

## 5. Defensible contributions

1. A timestamp-faithful PV evaluation protocol that splits before preprocessing and windowing, fits KNN/Isolation Forest/scalers on Train only, constructs windows independently within splits, and isolates Test from model selection.
2. A paired direct-multi-step benchmark across three co-located PV technologies using common horizons, origins, labels, masks, checkpoint rules and three random seeds.
3. Horizon-, technology- and efficiency-aware evidence showing that rankings and errors depend on the array and forecast horizon, without presenting ModernTCN as an original algorithm.
4. A Validation-led trajectory-error and information-boundary analysis showing that high-change daylight windows dominate error, direct forecasts become over-smoothed with lead time, and past second-level irradiance dynamics do not provide stable incremental point-forecast value in the tested Site 17 setting.

## 6. Evidence sufficiency

### Data

The authoritative DKASC audit supports the identities and nominal capacities of Site 17 Sanyo (HIT hybrid silicon, 6.3 kW), Site 25 Hanwha Solar (poly-Si, 5.83 kW), and Site 38 Q CELLS (mono-Si, 5.9 kW). DKASC field definitions map `Active_Power` to 5-minute-average AC power in kW. The benchmark project-layer CSVs cover 2018-04-01 through 2018-08-31 on a complete 5-minute grid and contain timestamps, power, performance ratio, temperature, humidity and four irradiance variables.

However, the project-layer CSVs must not be described as untouched raw downloads. The lineage audit calls them best-available project source layers, and original missingness may already have been regularized. Formal writing therefore requires a one-page reconciliation between the newly downloaded authoritative DKASC files and the exact 2018 benchmark files, plus official coordinates, licensing and download-access dates.

### Models and repeats

The clean benchmark actually contains:

- ModernTCN;
- iTransformer;
- PatchTST;
- Discrete Candidate.

It does **not** contain completed clean DLinear or TimesNet runs. Each listed model has 3 datasets × 3 seeds = 9 completed runs. Each run contributes full-timeline and predefined-daylight RMSE, MAE, R² and nRMSE for H12/H48/H96/H144, all derived from one H144 output. All 36 prediction files exist locally. Existing parameter counts and training times are available; a common inference-latency rerun is still required but does not train a model.

The internal Discrete Candidate should be retained only as a declared comparator. It is not the proposed method, and its prior viability verdict was FAIL.

### Fairness

The code and prior tests establish common date splits, H144-prefix evaluation, Train-only fitting and Validation-only checkpoint selection. Before final tables, one compact script must revalidate all 36 files for elementwise-equal labels, timestamps and masks within each dataset/seed comparison and recompute every metric. This is ordinary analysis, not a new contract or gate.

### Current benchmark interpretation

Existing rankings show ModernTCN mean RMSE rank 1.25 and first place in 10 of 12 dataset–horizon combinations among the four tested models. This is a result about the tested implementations and protocol, not a state-of-the-art claim. The exact table values must be regenerated from saved predictions before writing.

## 7. Abstract structure

- **Background:** Reliable ultra-short-term PV forecasting requires evaluation procedures that preserve temporal causality and prevent preprocessing and model-selection leakage.
- **Problem:** Published accuracy comparisons can be overstated when data are globally preprocessed, windows overlap splits, or Test affects checkpoint selection; model behavior may also differ across co-located PV technologies.
- **Protocol/method:** Describe the regular 5-minute timeline, Train-only preprocessing, split-local windows, direct H144 forecasts, four established implementations, three arrays and three seeds.
- **Results type:** Give clean multi-horizon rankings and uncertainty, technology/horizon dependence, efficiency results, and the Validation-led smoothing finding. Insert numbers only after the unified recomputation.
- **Conclusion:** State what the benchmark establishes about defensible model comparison and remaining information limits.
- **Do not state:** novel architecture, GFNODE superiority, state of the art, cross-climate generalization, arbitrary temporal resolution, deployment readiness, or independent final confirmation from the repeatedly inspected 2022 Test set.

## 8. Section plan

### 1. Introduction

**Purpose:** establish why evaluation integrity matters for operational PV forecasting and why co-located technologies provide a controlled comparison.

**Evidence:** Stage 0 leakage findings as motivation; clean protocol and completed benchmark scope.

**Nontraining work:** update the literature on PV forecasting evaluation leakage, direct multi-step forecasting and co-located arrays; verify official DKASC citations.

**Old manuscript reuse:** general PV variability and grid-operation background may be rewritten after source verification.

**Delete:** all GFNODE novelty, performance and deployment claims; all old numerical claims.

### 2. Related Work

**Purpose:** distinguish model innovation papers from reproducible evaluation/application studies.

**Core content:** direct multi-step forecasting; iTransformer, PatchTST and ModernTCN; leakage in temporal ML; PV technology and horizon dependence; trajectory smoothing under squared loss.

**Nontraining work:** current literature search and accurate primary citations.

**Old manuscript reuse:** only generic cited background that remains correct; no old ranking narrative.

### 3. Data and Leakage-Free Evaluation Protocol

**Purpose:** make temporal causality and data lineage auditable.

**Core content:** DKASC metadata, project-layer lineage, exact split dates, 5-minute reindex, missing masks, Train-only KNN/Isolation Forest/scalers, split-local lookback 72/H144 windows, daylight rule and Test isolation.

**Evidence:** lineage and multirate audits, clean configs, benchmark implementation and tests.

**Nontraining work:** metadata reconciliation, split/sample table, function/line references, and official data-license statement.

**Old manuscript reuse:** only verified site background and variable definitions; old preprocessing narrative must be rewritten.

### 4. Forecasting Models and Experimental Setup

**Purpose:** describe a fair comparison of existing models without claiming ownership.

**Core content:** four completed models, direct H144 decoding, fixed optimizer and early stopping, seeds, metrics, parameter counting and hardware.

**Evidence:** clean config, run status, parameter-efficiency CSV, code.

**Nontraining work:** cite official implementations, reconcile parameter counts, standardize latency protocol, and compute persistence/seasonal-persistence skill.

**Delete:** GFNODE equations, ODE solver, arbitrary-resolution assertions and old baseline configurations.

### 5. Results

**Purpose:** answer which existing model performs best under the common protocol and how results change by array and horizon.

**Core content:** three-seed mean±SD for full/daylight RMSE, MAE, R² and nRMSE; rankings; persistence skill; efficiency.

**Evidence:** 36 H144 prediction artifacts and current metrics tables.

**Nontraining work:** unified recomputation and figures.

**Delete:** original Tables 9–13 and Figures 10–14 in full.

### 6. Trajectory Error and Information-Limit Analysis

**Purpose:** explain the remaining failure mode rather than proposing another model.

**Core content:** Site 17/2022 lead error, change-amplitude and TV collapse, high-change SSE concentration, representative Validation cases, and bounded negative HF_DYNAMICS result.

**Evidence:** deterministic opportunity and information screen artifacts.

**Nontraining work:** condense scenario rows and predefine example selection on Train/Validation.

**Boundary:** 2022 Test was repeatedly inspected and is exploratory, not a new independent confirmation set.

### 7. Discussion

**Purpose:** interpret engineering meaning and limitations.

**Core content:** ranking dependence, why historical volatility marks risk but not future cloud direction, conditional-mean smoothing, implications for forecast evaluation and future exogenous data.

**Optional negative evidence:** ALICD and joint sharing may appear briefly as bounded internal checks, not as main contributions or a catalogue of failures.

**Delete:** claims of continuous-time advantage, cross-technology generalization and operational deployment.

### 8. Conclusion

**Purpose:** summarize defensible benchmark findings and limits.

**Allowed:** protocol integrity, tested-model ranking, technology/horizon dependence, trajectory smoothing and data needs.

**Prohibited:** ModernTCN invention, universal superiority, cross-site/climate validity, or causal claims about PV technology.

## 9. Minimum table plan

| Table | Rows | Columns | Source | Current state | Training? | Core conclusion |
|---|---|---|---|---|---|---|
| T1 Dataset and protocol | 3 arrays plus 2022 case-study row | Site ID, technology, capacity, unit, period, variables, split counts, missingness | DKASC audit + clean summary | Needs metadata reconciliation | No | Data and time semantics are explicit |
| T2 Models and setup | 4 models + persistence | architecture citation, input/output, parameters, optimizer, epochs, selection | config/code/efficiency | Ready after reconciliation | No | Comparison is declared and reproducible |
| T3 Main benchmark | model × array rows | H12/H48/H96/H144 RMSE/MAE/R²/nRMSE mean±SD, full/daylight | 36 H144 files | Ready after recalculation | No | Rankings depend on model, array and horizon |
| T4 Efficiency | 4 trained models | parameters, training time, best epoch, standardized latency, rank | efficiency + new timing | Needs nontraining measurement | No | Accuracy–cost trade-offs are implementation-specific |
| T5 Trajectory error | Validation scenarios/leads | N, RMSE, bias, difference MAE, TV ratio, SSE share | trajectory metrics | Ready | No | High-change daylight smoothing dominates error |
| T6 Protocol verification | identified issue rows | invalid legacy behavior, clean replacement, evidence | Stage 0 + clean tests | Ready | No | The clean benchmark addresses specified leakage risks |

T6 is a protocol checklist, not a leaked-versus-clean performance table. No causal accuracy effect should be claimed without paired reruns.

## 10. Minimum figure plan

1. **Clean protocol flow:** raw/project-layer timestamps → split → Train-fit preprocessing → split-local windows → Validation selection → one-time Test evaluation.
2. **Timeline schematic:** exact date boundaries, lookback 72 and H144 target; no split crossing.
3. **Multi-horizon comparison:** array panels with seed uncertainty for RMSE/nRMSE.
4. **Lead-time error and smoothing:** Validation H1–H12 RMSE plus change-amplitude/TV ratios.
5. **Trajectory examples:** stable versus high-change examples selected by predeclared Train/Validation rules.
6. **Parameter–performance plot:** standardized inference cost and RMSE rank, without causal capacity claims.

Do not reuse the random ODE state plot, the continuous-integration illustration, original Figure 12, or any old result figure lacking a clean source chain.

## 11. Remaining work

### A. Mandatory and nontraining

1. Reconcile Sites 17/25/38 official metadata, coordinates, capacity, technology, unit, licensing and download dates with the exact 2018 benchmark files.
2. Write one compact analysis script that reloads all 36 H144 prediction artifacts, verifies labels/timestamps/masks, and regenerates all metrics, rankings and sample counts.
3. Compute predeclared persistence and, if time semantics permit, seasonal-persistence references on identical origins. These are analytic baselines, not neural training.
4. Measure inference latency under one documented hardware/software environment with warm-up, repeated runs and fixed batch sizes.
5. Rebuild every table and figure from clean artifacts; archive captions and generation commands.
6. Update primary-source citations for DKASC and all four model implementations.
7. Write a reproducibility statement covering code release, data access, artifact-derived tables and limitations of the project-layer CSVs.
8. Confirm the live target-journal Guide for Authors, indexing, article type, APC, abstract, highlights and graphical-abstract requirements immediately before formatting.

### B. Optional

- A small number of additional Validation-selected trajectory examples.
- Sensitivity visualizations using already saved predictions.
- Supplementary detail on failed HF_DYNAMICS or ALICD screens, kept subordinate.

### C. Requires training and is not authorized

- Additional neural baselines, seeds, tuning or architectures.
- Retraining on newly downloaded raw files.
- Cross-time, cross-location or cross-climate confirmation.
- Any revival of GFNODE, ALICD, Ramp, probability or shared-private methods.

These are excluded from the fast manuscript unless an editor or supervisor later explicitly changes scope.

## 12. Limitations to carry into writing

- The primary arrays are co-located, not independent locations.
- The clean benchmark period is 2018-04-01 to 2018-08-31; the 2022 Site 17 analysis is a separate case study.
- Forecast-origin inputs do not include future NWP, cloud motion, satellite imagery or sky images.
- ModernTCN is an established ICLR 2024 method.
- The 2022 Test period has been used repeatedly for exploration and cannot be presented as untouched final confirmation.
- The 2018 project-layer files may not preserve original download missingness even though their timestamp grids are complete.
- No claim of cross-climate, cross-site or deployment validity is supported.

## 13. Final action

Proceed to a nontraining evidence-consolidation stage, then begin the new manuscript. Do not return to model development. The immediate deliverable should be the unified recomputation/provenance script, authoritative metadata table, persistence reference and rebuilt clean figures; once those are complete, formal writing for Solar Energy Advances can start.


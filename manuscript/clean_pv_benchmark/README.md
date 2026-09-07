# Scheme A multisite manuscript

**Leakage-Aware Multi-Horizon Benchmarking of Compact Neural PV Forecasts Across Technologies and Sites** is a JRSE-oriented benchmark article for author review. It contains two evidence levels: three co-located Alice Springs arrays (2018, 17 channels, 36 runs), and separately fitted Yulara/NIST replications (2017, seven channels, 24 runs). These are compact project implementations, not official full architecture reproductions or zero-shot transfer.

## Rebuild without training

From this directory, use Python with NumPy, pandas and ReportLab, the Arial fonts used by the existing figure pipeline, and TeX Live 2025 / REVTeX 4.2:

```powershell
python -B build_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The figure builder reads frozen CSVs, never checkpoints or raw data, and performs no training. The main source has five figures and five tables; the Supplement has three figures and eighteen tables. Accessibility descriptions cover all 31 figures/tables in `FIGURE_ALT_TEXT.txt`.

## Evidence sources

- Alice: `GFNODE_experiments/scheme_A_submission_correction/corrected_metrics.csv`, `REPORT.md`, and `INDEPENDENT_EVIDENCE_AUDIT.json`.
- External: `GFNODE_experiments/scheme_A_multisite_extension/metrics_per_seed.csv`, `metrics_summary_mean_sd.csv`, `DATA_AUDIT_SUMMARY.csv`, and the M1-R/M2 protocol and audits.
- Paper comparison: `multisite_manuscript_comparison.csv` (400 program-generated rows) and `M3_COMPARISON_AUDIT.json`.
- Verification and production review: `M3_EVIDENCE_AUDIT.json` and `REVIEW.md`.

Alice's 12/9/2/1/0 primary wins, Daily's 22/24 wins against the post hoc envelope, Hanwha H12 exceptions, and Qcells H12 support are unchanged. The external prespecified Inverted-variate model wins against Last-value in 8/8 at each site and against Daily in Yulara 8/8 and NIST 7/8. The external post hoc envelope also has 8/8 and 7/8 Daily wins, with distinct selection status. The original and external groups are not pooled into a forty-comparison headline.

## Author and release boundary

This M3 revision is for author review, not submission. The four-author metadata block is preserved from the designated base; no optional ORCID is inferred. Final M3 wording, PDFs, cover letter and submission authorization remain author-owned. The Chinese `submission_package/AUTHOR_SIGNOFF_CHECKLIST.md` separates earlier user confirmations from approval of these revised files.

The data statements direct readers to official DKASC and NIST downloads under provider terms. A curated code release still needs an author-selected license and confirmed URL/scope. No raw observations, checkpoints, NPZ arrays, preprocessors, local path configurations or results are included in this manuscript change. No GitHub Release, visibility change, journal upload, OA selection, rebase, force push or merge is performed.

Draft PR: https://github.com/Zhujiangkun-Yohkoh/PV-forecast/pull/19
Base: `research/scheme-a-multisite-frozen-training`.

# Leakage-aware PV benchmark manuscript

This directory contains the JRSE-oriented Research Article **“Leakage-Aware Multi-Horizon Benchmarking of Compact Neural PV Forecasts Across Co-Located Technologies”** and its Supplementary Material. The work is a benchmark/application study, not a new-model paper. It evaluates four compact project implementations under a common causal protocol and does not claim full reproduction of iTransformer, PatchTST, or ModernTCN.

## Evidence and reproducibility boundary

The quantitative source of truth is:

`GFNODE_experiments/scheme_A_submission_correction/corrected_metrics.csv`

`build_figures.py` reads that long-format file directly and generates all five vector figures plus the main and supplementary quantitative tables. It does not read former GFNODE results. The independent verifier in the experiment directory reproduced 4,414/4,414 comparisons from saved artifacts without importing the production metric functions. The manuscript preserves the verified primary wins (12/9/2/1/0), the matched Daily-Persistence result (22/24), and the Qcells H12 support (6,463 origins; 77,556 full and 36,504 daylight target points).

No neural-network training is part of the manuscript build. Checkpoints, predictions, raw data, and local `results/` are deliberately excluded.

## Build

Requirements are TeX Live 2025 (or equivalent), `latexmk`, REVTeX 4.2, and Python with pandas, NumPy, and ReportLab.

```powershell
python build_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The checked build produces `main.pdf` and `supplementary.pdf`; all fonts are embedded. `FIGURE_ALT_TEXT.txt` provides 25–50-word descriptions for every figure and principal table. Both PDFs remain working documents until the authors confirm declarations and approve submission.

## JRSE format basis

Official requirements were checked on 2026-08-28 using the [AIP author instructions](https://publishing.aip.org/resources/researchers/author-instructions/), [JRSE scope](https://pubs.aip.org/aip/jrse/pages/about), [AIP ethics policies](https://publishing.aip.org/resources/researchers/policies-and-ethics/), and [JRSE charges](https://pubs.aip.org/aip/jrse/pages/charges). The manuscript uses the installed AIP REVTeX style, a single-paragraph abstract below 250 words, the required declarations/order, a separate Supplementary Material PDF, and alt text. The planned route is subscription/non-OA; optional Author Select is not authorized.

`INDEXING_STATUS_REQUIRES_AUTHOR_OR_LIBRARY_CONFIRMATION`: current SCIE/JCR status was not independently established from an accessible Clarivate institutional record.

## Files

- `main.tex`, `main.pdf`: main article source and compiled working PDF.
- `supplementary.tex`, `supplementary.pdf`: separate supplementary source and PDF.
- `references.bib`: cited literature, including the 2025–2026 direct competitors.
- `build_figures.py`, `figures/`: evidence-driven vector outputs.
- `main_result_tables.tex`, `supplementary_tables.tex`: generated LaTeX tables.
- `FIGURE_ALT_TEXT.txt`: figure/table accessibility descriptions.
- `REVIEW.md`: reviewer-style scientific and production audit.
- `submission_package/`: working cover letter and upload/metadata checklists.
- `PUBLIC_RELEASE_MANIFEST.md`: proposed scope for a future dedicated public repository.

## Author action still required

Before upload, all authors must confirm the author list, CRediT roles, Funding Grant Nos. 62271151 and W2421092, conflict-of-interest and AI-use wording, ethics applicability, code/data release wording, and final manuscript. Three optional ORCIDs remain unconfirmed. The corresponding authors must also reconfirm the subscription route, current indexing, and any conditional production charges.

## Public release status

`PUBLIC_RELEASE_REQUIRES_ACTION`. The existing multi-branch repository should remain private. A dedicated release still requires license selection, a reviewed dependency specification, path-independent public configuration, provider-compliant data download instructions, and an author-approved release scope. No release or visibility change is performed here.

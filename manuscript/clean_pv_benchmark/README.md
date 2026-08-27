# Clean deterministic PV benchmark manuscript

This directory contains a complete LaTeX draft targeted to the **Journal of Renewable and Sustainable Energy (JRSE, AIP Publishing)**. It is a leakage-aware benchmark/application article, not a revision of the former GFNODE manuscript and not a new-model paper.

## Journal and publication-route verification

Verified on 2026-08-27 from official AIP/JRSE sources; manuscript evidence corrected and independently verified on 2026-08-27:

- [JRSE aims and scope](https://pubs.aip.org/aip/jrse/pages/about): the journal covers interdisciplinary renewable-energy physical science and engineering, including solar photovoltaics, energy meteorology, distributed generation, utility power, and system integration. This supports the present PV forecasting application/evaluation scope.
- [AIP author instructions](https://publishing.aip.org/resources/researchers/author-instructions/): Word or LaTeX submissions are accepted; initial submission uses a single compiled PDF (plus a separate PDF only if supplementary material exists); the abstract is one paragraph of at most 250 words; conflict-of-interest, CRediT author-contribution, and data-availability statements are required. Figures and tables require alt text at revision. No JRSE-specific Research Article word/page limit is stated on this page.
- [Official AIP LaTeX template on Overleaf](https://www.overleaf.com/latex/templates/template-for-submission-to-aip-journals/wdmsvzfjgvyj): the template covers JRSE and uses REVTeX. This repository uses the installed `revtex4-2` AIP style and does not redistribute publisher class/style files.
- [JRSE publication charges](https://pubs.aip.org/aip/jrse/pages/charges): the official page states that there are no page charges. Author Select open access is optional and listed at USD 3,800. The intended route is subscription/non-OA, with no Author Select purchase.
- [AIP license information](https://publishing.aip.org/resources/researchers/rights-and-permissions/licensing/): JRSE is listed among AIP subscription journals, with Author Select as an optional open-access route.

The JRSE page reports a 2025 Journal Impact Factor of 2.4 and Q4 in both *Energy & Fuels* and *Green & Sustainable Science & Technology* (Clarivate data displayed by the publisher in 2026). Direct institutional access to the Clarivate Master Journal List/JCR record was unavailable during this task. Therefore:

- **INDEXING_STATUS_REQUIRES_AUTHOR_OR_LIBRARY_CONFIRMATION.**
- **JCR quartile requires institutional verification.**
- Before submission, the corresponding author should reconfirm that no mandatory color, overlength, or other unavoidable production charge applies to the selected subscription route. The official JRSE charges page currently states no page charges and only optional OA fees.

## Evidence boundary

The original manuscript draft used Stage-2 evidence under:

`GFNODE_experiments/clean_deterministic_manuscript_stage2_evidence/`

Submission-critical review identified three issues in that evidence: historical Active Power was absent from all learned inputs despite being available to Last-value Persistence, Validation batch means were weighted equally, and prefix metrics were restricted to complete-H144 origins. The corrected quantitative source is now:

`GFNODE_experiments/scheme_A_submission_correction/corrected_metrics.csv`

`build_figures.py` reads that long-format file directly. The previous Stage-2 CSVs are retained as audit history but are not authoritative for manuscript results. The paper does not use former GFNODE Tables 9--13, Figures 10--14, or any C1/NWP artifact.

## Build

Requirements: TeX Live 2025 (or equivalent), `latexmk`, REVTeX 4.2, and the bundled Python environment with pandas, NumPy, and ReportLab.

```powershell
python build_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked build produces `main.pdf` with no missing citations, no undefined references, and no overfull boxes. The vector figures are regenerated without neural-network training.

## Files

- `main.tex`: complete English manuscript.
- `references.bib`: verified references including MANODE and other Alice Springs/PV forecasting studies.
- `build_figures.py`: evidence-driven generator for four vector PDF figures.
- `figures/`: final vector figures.
- `main.pdf`: compiled manuscript.
- `REVIEW.md`: independent reviewer-style audit and remaining submission actions.
- `submission_package/`: cover-letter draft, author-owned metadata checklist, upload manifest, and a copy of the final compiled manuscript PDF.

## Human confirmation required

The following are intentionally not guessed:

1. Final CRediT author-contribution statement and author approval/order.
2. Funding agency names and grant numbers.
3. Institutional confirmation of current SCIE indexing and JCR quartile.
4. Corresponding-author confirmation of subscription-route production charges at the point of submission.
5. Final confirmation that the repository/data-availability wording matches the release policy.
6. Conflict of Interest, CRediT, funding, ethics applicability, ORCID, exclusive-submission, and all-author-approval declarations. They are intentionally not guessed in the manuscript or cover letter.

The submission correction trained 36 runs in the isolated Scheme-A worktree. It did not modify the old GFNODE manuscript, raw data, prior artifacts, master worktree, Scheme C1, or NWP branches. New checkpoints and predictions remain local and untracked.

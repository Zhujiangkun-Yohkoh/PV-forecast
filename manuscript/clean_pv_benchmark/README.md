# Clean deterministic PV benchmark manuscript

This directory contains a complete LaTeX draft targeted to the **Journal of Renewable and Sustainable Energy (JRSE, AIP Publishing)**. It is a leakage-aware benchmark/application article, not a revision of the former GFNODE manuscript and not a new-model paper.

## Journal and publication-route verification

Verified on 2026-08-26 from official AIP sources:

- [JRSE aims and scope](https://pubs.aip.org/aip/jrse/pages/about): the journal covers interdisciplinary renewable-energy physical science and engineering, including solar photovoltaics, energy meteorology, distributed generation, utility power, and system integration. This supports the present PV forecasting application/evaluation scope.
- [AIP author instructions](https://publishing.aip.org/resources/researchers/author-instructions/): Word or LaTeX submissions are accepted; the initial submission may be a single compiled PDF; the abstract limit is 250 words; conflict-of-interest, author-contribution, and data-availability statements are required.
- [Official AIP LaTeX template on Overleaf](https://www.overleaf.com/latex/templates/template-for-submission-to-aip-journals/wdmsvzfjgvyj): the template covers JRSE and uses REVTeX. This repository uses the installed `revtex4-2` AIP style and does not redistribute publisher class/style files.
- [JRSE publication charges](https://pubs.aip.org/aip/jrse/pages/charges): the official page states that there are no page charges. Author Select open access is optional and listed at USD 3,800. The intended route is subscription/non-OA, with no Author Select purchase.
- [AIP license information](https://publishing.aip.org/resources/researchers/rights-and-permissions/licensing/): JRSE is listed among AIP subscription journals, with Author Select as an optional open-access route.

The JRSE page reports a 2025 Journal Impact Factor of 2.4 and Q4 in both *Energy & Fuels* and *Green & Sustainable Science & Technology* (Clarivate data displayed by the publisher in 2026). Direct institutional access to the Clarivate Master Journal List/JCR record was unavailable during this task. Therefore:

- **SCIE status requires institutional Master Journal List verification.**
- **JCR quartile requires institutional verification.**
- Before submission, the corresponding author should reconfirm that no mandatory color, overlength, or other unavoidable production charge applies to the selected subscription route. The official JRSE charges page currently states no page charges and only optional OA fees.

## Evidence boundary

All quantitative results and metadata come from the committed Stage-2 evidence under:

`GFNODE_experiments/clean_deterministic_manuscript_stage2_evidence/`

`build_figures.py` reads `FINAL_METRICS_LONG.csv` and `FINAL_EFFICIENCY.csv` directly from that location. No duplicate evidence CSV is stored here. If Stage-1 and Stage-2 differ, Stage-2 is authoritative. The manuscript does not use the former GFNODE Tables 9--13, Figures 10--14, or any C1/NWP artifacts.

## Build

Requirements: TeX Live 2025 (or equivalent), `latexmk`, REVTeX 4.2, and the bundled Python environment with pandas, NumPy, and ReportLab.

```powershell
python build_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The checked build produces `main.pdf` with no missing citations, no undefined references, and no overfull boxes. The vector figures are regenerated without neural-network training.

## Files

- `main.tex`: complete English manuscript.
- `references.bib`: 35 references, including current time-series methods and PV forecasting studies.
- `build_figures.py`: evidence-driven generator for four vector PDF figures.
- `figures/`: final vector figures.
- `main.pdf`: compiled manuscript.
- `REVIEW.md`: independent reviewer-style audit and remaining submission actions.

## Human confirmation required

The following are intentionally not guessed:

1. Final CRediT author-contribution statement and author approval/order.
2. Funding agency names and grant numbers.
3. Institutional confirmation of current SCIE indexing and JCR quartile.
4. Corresponding-author confirmation of subscription-route production charges at the point of submission.
5. Final confirmation that the repository/data-availability wording matches the release policy.

No neural-network training was performed, and no old manuscript, raw data, checkpoint, prediction artifact, master worktree, or Scheme C1 file was modified.

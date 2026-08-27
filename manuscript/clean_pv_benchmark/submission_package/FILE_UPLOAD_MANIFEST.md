# JRSE file upload manifest

Checked against official AIP/JRSE instructions on **2026-08-27**.

## Initial submission

| File | Status | Purpose |
|---|---|---|
| `main.pdf` | Prepared; author approval required | Single compiled manuscript PDF with embedded figures. |
| `COVER_LETTER_DRAFT.md` | Draft only | Convert or paste only after the corresponding author confirms all declarations. A JRSE-specific mandatory cover-letter rule was not found. |

No supplementary-material PDF, Highlights file, or graphical abstract is included because none is required by the official instructions reviewed and no supplementary analysis is being submitted.

## Source bundle when requested by the submission/production system

Do not create a duplicate metric source. Package the repository versions of:

- `main.tex`
- `references.bib`
- `figures/fig1_leakage_free_protocol.pdf`
- `figures/fig2_multihorizon_nrmse.pdf`
- `figures/fig3_rank_heatmap.pdf`
- `figures/fig4_accuracy_efficiency.pdf`

The project uses the installed REVTeX 4.2 AIP style and does not redistribute publisher class/style files. AIP’s initial-submission instruction permits a single compiled PDF; source/figure upload should follow the live submission system if it asks for them.

## Revision-stage accessibility item

AIP requires alt text for figures and tables and asks for a separate TXT or DOCX upon revision. It is not included yet because the authors must review the scientific descriptions and the current package is for initial submission preparation.

## Explicit exclusions

The upload set must not contain raw PV data, checkpoints, NPZ predictions, local `results/`, caches, LaTeX auxiliary files, Git credentials, or unconfirmed author/funding/declaration metadata.

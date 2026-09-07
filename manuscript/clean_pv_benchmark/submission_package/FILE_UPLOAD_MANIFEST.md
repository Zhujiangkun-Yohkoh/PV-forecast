# M3 file manifest — for author review

| File | Current content | Approval status |
|---|---|---|
| `main.pdf` | 13-page article; 205-word abstract; five figures and five tables | Final M3 all-author approval pending |
| `supplementary.pdf` | 29-page Supplement; three figures and eighteen tables | Final M3 all-author approval pending |
| `FIGURE_ALT_TEXT.txt` | All eight figures and 23 tables | Supply through the journal's accessibility workflow |
| `COVER_LETTER_DRAFT.md` | Two-level study, external primary results and conditional ranking | Corresponding-author approval/signature pending |
| `SUBMISSION_METADATA_CHECKLIST.md` | Updated title, scope, metadata and production record | Internal preparation file |
| `AUTHOR_AND_DECLARATIONS_DRAFT.md` | Preserved metadata and declaration status | Internal preparation file |
| `AUTHOR_SIGNOFF_CHECKLIST.md` | Chinese record of earlier confirmations and final M3 approvals | No automatic signoff |

The package PDF copies must be byte-identical to the corresponding PDFs in the manuscript directory. The exact version is the final commit/tree of Draft PR #19; this manifest does not assert upload authorization.

## Source files when requested

`main.tex`, `supplementary.tex`, `references.bib`, `main_result_tables.tex`, `supplementary_tables.tex`, `multisite_main_tables.tex`, `multisite_quality_tables.tex`, and `multisite_supplementary_tables.tex`.

Vector figure files and display roles:

- Figure 1: `figures/fig1_leakage_free_protocol.pdf`
- Figure 2: `figures/fig2_persistence_reversal.pdf`
- Figure 3: `figures/fig3_multisite_last.pdf`
- Figure 4: `figures/fig4_multisite_daily.pdf`
- Figure 5: `figures/fig5_multisite_ranks.pdf`
- Figure S1: `figures/figS1_rank_heatmap.pdf`
- Figure S2: `figures/fig3_horizon_technology.pdf`
- Figure S3: `figures/fig4_accuracy_efficiency.pdf`

The program-generated paper comparison CSV and audit JSONs are supporting repository evidence. Complete original and external metric CSVs remain in their experiment directories without changes.

## Exclusions and remaining author decisions

No raw or full derived data, checkpoints, NPZ arrays, preprocessors, results, caches, local absolute path configurations or temporary renders are committed or uploaded. No raw DKASC/NIST data redistribution is claimed. Final code URL, license, release scope, final declarations and submission approval remain author-owned. No GitHub Release, visibility change, OA choice or journal submission is performed.

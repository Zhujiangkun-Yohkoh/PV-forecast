# Current closeout figure and PDF QA

Actual final render: main 12 pages, Supplement 47 pages. All 59 pages were viewed as rendered contact sheets; affected figures also inspected at their embedded scale. No clipped labels, overlapping captions, missing references or overfull boxes were observed. All PDF fonts are embedded; scientific figure objects remain vector (zero raster images in both PDFs).

Main has 5 figures and 5 tables; Supplement has 17 figures and 18 historical tables. Ordinary main figure text is at least 8.96 pt. The smallest extracted Supplement glyph is a logarithmic superscript (5.98 pt), not ordinary tick or legend text. Historical long numerical tables remain dense electronic reference material; their CSVs provide accessible numeric lookup.

| Current figure | Change and data | Actual size review / remaining limitation |
|---|---|---|
| Main 3 / fig9_alice_support | New accepted Hanwha/Qcells Inverted three-seed contrasts on Original/S1/S2; accepted_summary.csv | Four panels, 1 h/12 h, full targets; 48 h time intervals, original reference support named. No Sanyo incomplete mean. Readable, no clipping. |
| Main 5 / fig8_fixed_period | New four-cell complete/pointwise 2017/2018 contrasts; four_cell_intervals.csv | Six panels, separately scaled sites, all unfavorable effects shown. No pooling of dates or seeds. Readable, no clipping. |
| Other 20 groups | Retained data and vector assets from preceding revision; individual data/caption/alt preserved | Viewed on final PDF; no redraw where data unchanged. Supplement retains one-figure pages and historical table detail; journal-specific placement remains editorial work. |

The numbered CSV map distinguishes main 1–5 and Supplement S1–S17. Each of 22 groups has PDF, SVG, 320 dpi PNG, source CSV, caption and alt text. FIGURE_OVERVIEW.png is a navigation contact sheet, not an embedded paper figure. No neural training, tolerance change or favorable-result filtering was used for visual revision.

The PDFs were rebuilt after removing the forced main page break and shortening repeated historical audit prose. Main dropped from 14 to 12 pages through float/text flow, without font reduction. Final logs and font inventories are in scheme_A_support_closeout/results/PDF_QA.json. Raster previews stay local and are not represented as scientific evidence.

# Scheme A Multisite M3 manuscript review

Date: 2026-09-07. Decision: **SCHEME_A_MULTISITE_MANUSCRIPT_READY_FOR_AUTHOR_REVIEW**.

This is a manuscript-review decision, not final author signoff or submission authorization. The existing branch and Draft PR #19 are retained, based on the frozen M2 branch. The corrected author-provided Alice artifact location resolved the earlier path blocker without changing any experiment rule or result.

## Evidence and preservation

- Original Alice: exactly 36 complete groups, each with `best_validation.pt`, `completed.json`, and `test_H144.npz`; identity, shapes, and strict 17-channel checkpoint loading verified. All 108 files have unchanged size and mtime_ns.
- M1-R tests: 43 passed; M2 ordinary tests: 17 passed; M2 artifact tests: 10 passed. Failed, errors, and skipped are zero.
- Independent external artifact comparisons: 10,432/10,432; maximum absolute discrepancy 4.547473508864641e-13. All 24 external checkpoints reproduce saved predictions within the frozen numerical tolerance.
- The 96 support groups and 24-run matrix are unchanged. An additional 1,360 paper-comparison checks pass, with zero failures or skips, covering the generated 400-row comparison CSV, envelopes, matched counts, and external seed means/sample SDs.
- A total of 605 protected files retain their size and mtime_ns: 497 external raw/result/frozen-evidence files plus the 108 original Alice artifacts. Full before/after inventories and absolute paths remain in ignored local configuration. Original Alice raw files were not accessed; no master worktree was entered.
- No training, checkpoint update, raw-data edit, prediction-array edit, or frozen metric CSV change occurred. The original Alice independent audit of 4,414 comparisons is retained as historical frozen evidence, not misrepresented as a newly executed M3 test.

## Scientific assessment

The manuscript now separates two evidence levels: the 2018 Alice co-located technology comparison with 17 inputs, and independently fitted 2017 external replications with seven common inputs at Yulara and NIST Ground. It describes 36 plus 24 historical training runs without performing any additional training.

Alice's original primary wins remain 12/9/2/1/0. Daily Persistence remains better than the favorable post hoc neural envelope in 22/24 comparisons; the two exceptions remain Hanwha H12 full/daylight. Qcells H12 retains 6,463 origins, 77,556 full points, and 36,504 daylight points. The descriptive focal Inverted-variate model has 24/24 Last-value wins and 0/24 Daily wins at Alice; it is not retroactively called prespecified there.

The external prespecified Inverted-variate model has 16/16 Last-value wins and 15/16 Daily wins. The programmatically recomputed post hoc envelope has 8/8 Daily wins at Yulara and 7/8 at NIST. NIST H144/full remains the Daily win even after favorable neural-member selection. The primary model's Daily skill there is -6.47% with sample SD 1.65 percentage points.

Yulara retains Inverted-variate at neural rank one. NIST mean ranks are 1.75 recurrent, 2.00 TCN, 2.25 Inverted-variate, and 4.00 joint-patch; recurrent leads full scope and TCN leads the longer daylight horizons. The narrative retains these outcomes and Yulara's smaller long-horizon Daily skills. Site, horizon, scope, and historical information condition the interpretation. No pooled cross-site kW ranking or undifferentiated forty-comparison win total is reported.

Matched target points resolve sample-support differences, not historical-information differences. Daily's 24-hour lag and the neural six-hour history remain different information strategies. Three-seed SD measures training randomness, not temporal sampling uncertainty. Counts of dependent comparisons are descriptive, not significance tests. The neural envelope is consistently a post hoc descriptive upper bound, not a deployable or prespecified model.

## Methods and limitations

The fixed NIST EST/LST coordinate, complete five-minute bins, meter power target, direct Wm2 field, Yulara provider-local coordinate and conservative five-minute availability, off-grid exclusions, Train-only preprocessing, split-local windows, and horizon-specific Test eligibility are documented. Alice's nonnegative target-validity rule and the external retention of finite negative targets are distinguished.

Missing NIST 2017 operation logs and partial Yulara metadata remain nonblocking documentation limitations. They are not interpreted as absence of anomalies. The limitations also state three geographic sites, co-located Alice arrays, October-December external evaluation, different years and input regimes, separate fitting rather than zero-shot transfer, compact project implementations, prior Test data-quality inspection, and the scope of seed SD. Possible explanations for ranking differences are identified as hypotheses, not proven mechanisms.

## Production and language audit

Title: **Leakage-Aware Multi-Horizon Benchmarking of Compact Neural PV Forecasts Across Technologies and Sites**.

- Abstract: 205 whitespace-delimited words. TeXcount main text: 4,735 words including frontmatter and declarations; bibliography/captions are outside this text count.
- Main: 13 pages, five figures, five tables. Supplement: 29 pages, three figures, eighteen tables. Full metric CSVs remain repository material rather than thousands of PDF rows.
- Both PDFs compile. All 37 bibliography entries are cited. Missing citations, undefined references, duplicate labels, and overfull boxes: zero.
- All fonts are embedded (31 main and 20 Supplement font resources); recursive PDF resource inspection finds zero raster images. Figures remain vector PDFs.
- All 42 pages were rendered and visually inspected; final changed pages were rechecked. Tables, legends, signs, and captions are readable without overlap or clipping. Three nonfatal REVTeX main float-placement fallback warnings remain; all corresponding floats are present and correctly placed in the inspected output.
- Signed heatmap values and a neutral zero support grayscale reading. The accuracy/latency figure uses zero-origin axes and a Last-value reference; complete ratio ranges remain visible in the logarithmic Alice evidence figure.
- Alt text accompanies all eight figures and 23 tables in `FIGURE_ALT_TEXT.txt`; the package copy matches. Manuscript/package PDF copies are byte-identical.
- Scientific English was revised around the evidence rather than the work history. Unsupported novelty, universal superiority, pure architectural causality, and significance claims are absent. Replication, transfer, validation, site, facility, array, horizon, and scope are distinguished.

## Author review and delivery boundary

The author block is unchanged from the frozen base, including order, affiliations, corresponding emails, and the supplied Zhu ORCID, and is consistent across both PDFs. No other ORCID was inferred. Earlier user confirmations of funding, contributions, conflict, ethics, and submission intent are preserved in the Chinese signoff checklist; final M3 wording and the complete revised documents still require author review.

Authors must confirm the final main/Supplement, declaration wording and any required AI disclosure, public code URL and license/release details, cover letter, metadata, related-work disclosure, and submission authorization. No final signoff, OA selection, license choice, release, visibility change, or submission has been made. Source data availability points to the providers and does not claim redistribution.

`M3_EVIDENCE_AUDIT.json` records the final evidence and PDF checks. `M3_COMPARISON_AUDIT.json` and `multisite_manuscript_comparison.csv` record programmatic cross-setting comparisons. The existing Draft PR #19 is the delivery vehicle; commit and tree identifiers are reported in the PR and final delivery message.

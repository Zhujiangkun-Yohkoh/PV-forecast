# Independent reviewer-style final-polish review

## Recommendation

**Ready for four-author signoff, but not for upload until the declarations are confirmed.** The article now presents an independent benchmark/application study rather than an algorithm paper or an internal correction narrative. Its contribution is the controlled estimand: causal Train-only preprocessing, regular-grid history with explicit missingness, split-local windows, elementwise-matched evaluation, two persistence lags, four horizons, two scopes, and unified efficiency measurement across three co-located PV technologies.

No neural-network training was executed during this polish. Frozen evidence is unchanged: the independent verifier passes 4,414/4,414 comparisons (maximum absolute difference 8.51e-12; maximum relative difference 3.23e-11); primary lowest-RMSE wins remain 12/9/2/1/0 for the inverted-variate, depthwise TCN, joint-patch, discrete recurrent, and Last-value implementations; matched Daily Persistence remains lower than the post hoc best-of-four neural envelope in 22/24 comparisons, with the two envelope wins limited to Hanwha H12 full/daylight; Qcells H12 remains 6,463 origins, 77,556 full points, and 36,504 daylight points (47.1%).

## Scientific assessment

### Strengths

1. The manuscript distinguishes individual model rankings from a **post hoc best-of-four neural envelope**. It explicitly states that the envelope is calculated after Test results and is neither a prespecified choice nor a deployable model.
2. The Last-value and Daily Persistence questions are separated. Their supports are matched elementwise within each comparison, so the 22/24 reversal is not a missing-lag selection artifact.
3. The missing-data description now matches the implementation: inputs remain on a regular five-minute coordinate, missing observations are retained through masks and Train-only imputation, and target eligibility is horizon-specific.
4. The co-located design controls much of the weather/site context while avoiding causal attribution of target differences to module technology alone.
5. The four forecasters are described as compact project implementations inspired by, but not equivalent to, official iTransformer, PatchTST, or ModernTCN releases.
6. Figure 2 now shows all 24 ratios on a shared logarithmic scale without truncation; Figures 1, 3, and 4 clarify causal support, scope dependence, and the neural-only efficiency frontier.

### Literature and data provenance

The comparison includes the 2025 review and SolNet, plus 2026 work by Markovics et al., Dhingra et al., Cross-Unet, and Mansour et al. It does not allege evaluation errors in those studies; it states that information sets, splits, lags, and eligible samples differ. The DKASC citation now points to the official Alice Springs download service and records the access date. The bibliography contains only entries cited in the main text.

### Interpretation and limitations

The strongest defensible conclusion is about evaluation practice, not a universal architecture advantage. At least one compact neural implementation is lower than Last-value Persistence in every primary comparison, but Daily Persistence is lower than the favorable post hoc envelope in 22/24 matched comparisons. The two Hanwha H12 exceptions are narrow observations, and the 24 combinations are dependent summaries rather than independent trials.

The estimand remains limited to one co-located facility, April--August 2018, history-only inputs, four compact implementations, and three seeds. No future NWP, probability intervals, cross-climate validation, or official full architecture reproduction is evaluated. Test did not affect preprocessing, training, or checkpoint selection, but it was repeatedly inspected during development and is not an untouched external confirmation set.

## Production and accessibility audit

- Main and Supplementary PDFs compile independently; all fonts are embedded.
- The main article is 12 pages with four main figures and four main tables; the supplement is 18 pages with S-numbered tables and figure.
- Figure 2 covers the complete observed ratio range; the previous lower-axis truncation is removed.
- Table I is set in `footnotesize` and is readable at 100% PDF scale.
- Main and supplement display the same four authors, order, affiliations, corresponding authors, emails, and confirmed Jiangkun Zhu ORCID.
- No missing citations, undefined references, duplicate labels, or visible content overflow remains. Ordinary underfull warnings do not impair the rendered pages.
- Both PDFs were rendered page by page and inspected for boundaries, labels, legends, whitespace, and table readability.

## Author-owned confirmations

The visible working-manuscript marker is intentional. Funding, CRediT, conflict of interest, ethics applicability, AI-use wording and exact tool/version, Data Availability, Code Availability, exclusive-submission status at upload, OA route, and final all-author approval remain pending. `submission_package/AUTHOR_SIGNOFF_CHECKLIST.md` identifies the responsible confirmer and date field for each item.

## Public release readiness

**PUBLIC_RELEASE_REQUIRES_ACTION.** The existing multi-branch repository is not a curated Scheme A reproducibility release. A dedicated release still needs an author-selected license, reviewed dependency specification, path-independent configuration, DKASC provider-compliant download instructions, exclusion of submission-only metadata, and a final secret/path scan. No release or visibility change was made.

## Final decision

Subject to the explicit author-owned confirmations, the polished scientific and production files support **SCHEME_A_JRSE_FINAL_POLISH_READY_FOR_AUTHOR_SIGNOFF**. This decision does not authorize journal submission, a GitHub release, or completion of any unconfirmed declaration.

## Visual-accessibility correction before signoff

The signoff files were rechecked after a limited accessibility correction. Figure 1 now states matching only within each array and no longer implies a three-array origin intersection. Figure 2 places its 24/24 and 22/24 annotations inside the corresponding panels and retains the complete observed ratio range. Figure 4 has four labeled vertical ticks. Supplementary Figure S1 shows all five complete model names without clipping. Main-text alt descriptions now map correctly to Tables I--IV, and the generated Supplementary Material places concise alt text below Tables S1--S10 and Figure S1.

Both PDFs were rebuilt and every page was rendered for inspection. The main article remains 12 pages; the inline supplementary alt text increases the Supplementary Material to 19 pages. All fonts are embedded. Ordinary tests passed 16/16, artifact tests passed 9/9, and independent evidence tests passed 15/15, including 4,414/4,414 numerical comparisons. No model training or protected-artifact modification occurred. The updated files support **SCHEME_A_JRSE_AUTHOR_SIGNOFF_FILES_READY**, subject to the unchanged author-confirmation requirements above.

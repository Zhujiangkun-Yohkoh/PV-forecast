# Independent reviewer-style review after submission-critical correction

## Overall assessment

The corrected manuscript is a defensible leakage-aware benchmark/application study, but it is not yet a submission-ready final package. The correction materially changes its inputs, samples, rankings, and persistence conclusions. Thirty-six corrected GPU runs and 15 protocol tests support the new results. The manuscript now describes the executed compact implementations rather than presenting them as full official iTransformer, PatchTST, or ModernTCN reproductions.

**Completion estimate: 78%.** Scientific consistency is restored, figures and tables are regenerated from corrected evidence, and the main conclusions are appropriately narrow. Before submission, the authors must confirm CRediT, funding, author metadata, current journal/indexing/charge details, and decide whether a manuscript whose strongest supplementary result is Daily Persistence beating the best neural implementation in 22/24 combinations is positioned persuasively enough for JRSE.

## Correction audit

1. **Input fairness:** passed. The previous 15-channel input had seven raw fields, seven missing masks, and one Isolation Forest marker; it had no cyclical time features and no past Active Power. The corrected 17-channel input adds only causal historical Active Power and its missing mask. Future power remains target-only.
2. **Train-only fitting:** passed. Active Power and all other imputation/scaling state, plus Isolation Forest, are fitted on Train only.
3. **Validation aggregation:** passed. Validation selection uses global target-weighted MSE (total SSE divided by valid-target count), not an equal average of batch means.
4. **Test isolation:** correctly narrowed. Test was excluded from preprocessing fitting, training, and checkpoint selection. The manuscript no longer calls it untouched or evaluated once at project level.
5. **Sample matching:** passed. Within every array/horizon, neural models and Last-value Persistence have identical origins, labels, masks, forecast-origin counts, and valid-target counts.
6. **Horizon design:** corrected. Primary metrics use horizon-specific valid origins; complete-H144-prefix evaluation is explicitly secondary sensitivity analysis.
7. **Qcells H12:** corrected. The old 3,019-versus-2,996 origin mismatch and 42-point daylight subset are removed. The primary set has 6,463 origins, 77,556 targets, and 36,504 daylight targets.
8. **Model naming:** corrected. Descriptive implementation names are used, and cited methods are identified only as inspiration.
9. **Reproducibility:** passed at ordinary implementation level. All 36 runs completed and 15/15 tests passed with no non-finite result.

## Scientific interpretation

The corrected primary ranking, including Last-value Persistence, is: Inverted-variate Transformer 12 wins (mean rank 1.875), Depthwise convolutional TCN 9 (2.167), Joint-patch Transformer 2 (2.458), Discrete recurrent decoder 1 (3.667), and Last-value Persistence 0 (4.833). Arithmetic mean RMSE skill relative to Last-value Persistence is positive for all learned implementations, but this ratio-based summary is not interchangeable with macro mean range-nRMSE.

The more important qualification is Daily Persistence. Its valid sample set differs slightly because a 24-hour lag is required, so it is correctly excluded from the primary rank. On its own eligible set it beats the best neural result in 22/24 array--horizon--scope combinations. This is not a footnote: it changes the practical message from “deep models beat persistence” to “learned trajectories beat Last-value Persistence but rarely beat daily recurrence in this short, seasonal archive.”

## Reviewer checklist

1. **Does the paper imply algorithmic novelty?** No. It explicitly frames the models as compact implementations inspired by established families.
2. **Is the application contribution potentially sufficient?** Yes, if the editor values fairness corrections, controlled co-location, horizon-specific sample accounting, and persistence-centered evidence. Limited architectural fidelity reduces suitability as an architecture benchmark.
3. **Are unfavorable results visible?** Yes. Daily Persistence dominance and the two Qcells H12 neural failures versus Last-value Persistence are explicit.
4. **Do manuscript numbers match corrected evidence?** Yes for the main tables, ranks, skill summaries, sample counts, and efficiency values checked after regeneration.
5. **Are full/daylight definitions clear?** Yes. Daylight uses true target power above 1% of the Train maximum and is evaluation-only.
6. **Is Daily Persistence mixed into the primary ranking?** No; its different eligible counts are stated.
7. **Is unknown AC capacity handled correctly?** Yes. Train-range nRMSE is used and DC nameplate capacity is not mixed with AC output.
8. **Are cross-site/deployment claims controlled?** Yes. The study is explicitly one co-located facility and one historical period.
9. **Is the old 18/24 claim removed?** Yes. It is mentioned only as withdrawn correction history.
10. **Are citations and Alice Springs precedents adequate?** MANODE 2024 and existing Alice Springs/DKASC work are included; a final publisher metadata check remains advisable.
11. **Are figures reproducible and fonts embedded?** Yes. The script reads corrected evidence and embeds Arial in vector PDFs. Figure 4 marker area is mathematically proportional to parameter count.
12. **Has the PDF been compiled and visually inspected?** Yes after correction; see build notes below.

## Main strengths

- The study openly corrects a target-history fairness flaw instead of preserving favorable legacy conclusions.
- Horizon-specific eligibility and exact model--Persistence sample matching make the primary comparisons interpretable.
- Co-location reduces weather/site confounding without being mislabeled as cross-site validation.
- The paper separates absolute normalized error from the arithmetic mean of ratio-based skill.
- Daily Persistence is allowed to challenge the entire learned-model narrative.
- Code-level names prevent overstating fidelity to published architectures.

## Most likely rejection risks

1. **Limited novelty and architecture fidelity.** The contribution is evaluation methodology and empirical correction, while the models are compact project implementations rather than official baselines.
2. **Daily Persistence dominance.** An editor may ask why learned models are necessary when daily recurrence wins 22/24 supplementary comparisons. The answer must be scientific---different eligible samples and a short seasonal archive---not rhetorical concealment.
3. **External validity.** One site, April--August 2018, three arrays, and three seeds do not establish multi-climate or multi-year generality.
4. **Test reuse during development.** Although Test never influenced fitting, training, or checkpoint selection, the project has inspected it repeatedly. The paper must not market it as pristine external confirmation.

## Submission judgment

**GO for continued JRSE benchmark/application preparation; NO-GO for immediate submission without author confirmation and editorial positioning review.** No additional model development is recommended. Any extra scientific work should be an independently held-out temporal confirmation, not another architecture or hyperparameter cycle; that work is outside the present authorization.

## Human confirmation required

- Author order, affiliations, corresponding authors, and CRediT roles.
- Funding agencies and grant numbers.
- Current institutional confirmation of SCIE/JCR status and subscription-route unavoidable charges.
- Agreement that Daily Persistence is presented as a central limitation rather than minimized.
- Final release wording for data, code, and the untracked corrected artifacts.

## Next Codex prompt

> In the isolated `manuscript/clean-pv-benchmark-latex` worktree, prepare the final JRSE submission package without training or changing any result. Read the corrected `GFNODE_experiments/scheme_A_submission_correction/REPORT.md`, `manuscript/clean_pv_benchmark/main.tex`, `README.md`, and `REVIEW.md`. Obtain from me only the verified CRediT roles, funding text, author approval, and institutional journal/indexing/charge confirmation. Insert that information; run a full DOI/author metadata check; ensure Daily Persistence remains prominent and sample-set caveats remain exact; rebuild all figures and `main.pdf`; verify no undefined references, missing citations, overfull boxes, or out-of-page content; render and inspect every page; prepare the cover letter and required declarations; then commit only the manuscript directory to the same branch and update Draft PR #5. Do not train, modify evidence, touch master/C1/NWP/old manuscript files, or merge the PR.

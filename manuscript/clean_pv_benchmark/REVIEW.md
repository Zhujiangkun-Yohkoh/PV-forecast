# Independent reviewer-style final evidence review

## Recommendation

**Evidence ready for preparation of a submission package; manuscript not ready for upload until author-owned metadata is confirmed.** The numerical comparisons, code paths, and manuscript statements are now mutually consistent. No new model training is warranted.

An independent NumPy/Pandas verifier that does not import the production metric or mask functions passed 4,414/4,414 comparisons. Its maximum absolute and relative differences were 8.51e-12 and 3.23e-11. The 22/24 Daily-Persistence result, both Hanwha H12 exceptions, the Primary ranks, and the Qcells H12 counts were all reproduced from source arrays and raw target timestamps.

## Evidence verification

1. **Run completeness:** 36/36 corrected runs contain `completed.json`, H144 predictions, and Validation-best checkpoints.
2. **Artifact identity:** all labels, forecast origins, target starts, and target-valid masks agree elementwise with the current protocol.
3. **Checkpoint provenance:** all checkpoints load as 17-input models, match their model/array/seed metadata, and reproduce saved predictions under no-gradient inference.
4. **Stale protection:** completed predictions cannot be silently reused when shape, labels, masks, origins, target starts, input dimension, identity, H144 settings, or split dates differ.
5. **Training boundary:** models were trained and selected on complete-H144 Train/Validation windows. Horizon-specific eligibility is used only for primary Test prefixes.
6. **Test boundary:** Test was excluded from preprocessing fitting, training, and checkpoint selection, but it has been inspected repeatedly during project development and is not presented as pristine external confirmation.
7. **Implementation scope:** the four neural models are compact project implementations inspired by established families, not official full reproductions and not new architectures.

## Daily Persistence correction

The earlier 22/24 statement compared Daily Persistence and neural metrics derived with different valid masks. That implementation was unsuitable for a win/loss claim even though both calculations were individually reproducible.

The final analysis, `supplementary_daily_matched`, constructs the actual Daily-Persistence point mask and applies it identically to Daily Persistence, Last-value Persistence, and every neural prediction. Forecast origins, labels, target points, and daylight masks therefore match within every comparison. The fair result remains **22/24 for Daily Persistence and 2/24 for the best neural implementation**, with the two neural wins limited to Hanwha H12 full timeline and daylight.

Because the count happens to remain unchanged, the abstract and conclusion retain 22/24 but now state the matched design. All “own eligible sample set” wording has been removed. The result is stronger methodologically and remains unfavorable to a general neural-superiority narrative.

## Primary evidence

- Inverted-variate Transformer: 12/24 primary RMSE wins, mean rank 1.875, arithmetic mean skill versus Last-value Persistence 0.549.
- Depthwise convolutional TCN: 9/24 wins, mean rank 2.167, skill 0.515.
- Joint-patch Transformer: 2/24 wins, mean rank 2.458, skill 0.493.
- Discrete recurrent decoder: 1/24 win, mean rank 3.667, skill 0.399.
- Last-value Persistence: 0/24 wins, mean rank 4.833.

These ratio-based skills are not described as arithmetic mean absolute or normalized errors. Qcells H12 remains corrected at 6,463 common origins and 36,504 daylight target points.

## Manuscript and PDF audit

- The title and benchmark/application positioning remain appropriate.
- Abstract, contributions, Persistence methods, results, discussion, limitations, and conclusion use the matched Daily-Persistence evidence.
- Missing timestamp rows, explicit masks, Train-only imputation, complete-H144 training/selection, and Test-only horizon eligibility are explicit.
- The efficiency table now uses readable `table*` layout and run-metadata values rather than compressed prior rounding.
- Figure 4 is enlarged; its circular marker area is mathematically proportional to parameter count.
- All four figures are regenerated from `corrected_metrics.csv` with embedded TrueType fonts.
- The complete PDF was compiled, rendered, and inspected page by page. Font embedding and LaTeX diagnostics are recorded in the final task report.

## Main strengths

- Exact artifact-to-protocol validation and reproducible checkpoint inference.
- Same-mask fairness for both Last-value and Daily Persistence comparisons.
- Transparent correction of conclusions rather than selective retention of favorable numbers.
- Clear distinction between project implementations and published architecture identities.
- Negative persistence evidence remains visible throughout the paper.

## Remaining rejection risks

1. **Limited methodological novelty:** the paper contributes protocol discipline and empirical evidence, not a new forecasting architecture.
2. **Daily recurrence dominates:** Daily Persistence wins 22/24 matched comparisons, raising a legitimate question about the practical value of the neural models for this short seasonal archive.
3. **External validity:** one co-located site, April--August 2018, three arrays, and three seeds cannot support cross-climate or broad technology-generalization claims.
4. **No pristine external confirmation:** the Test period was excluded from learning but repeatedly inspected during development.

## Author and release status

- Confirmed metadata now records four authors in this order: Jiangkun Zhu, Mengling Yang, Zhicong Chen, and Lijun Wu. Zhicong Chen and Lijun Wu are corresponding authors; Jiangkun Zhu's ORCID is recorded.
- CRediT roles and Funding Grant Nos. 62271151 and W2421092 remain drafts inherited from an earlier manuscript and require final confirmation by all authors.
- Institutional confirmation of current SCIE/JCR status and subscription-route charges.
- Final Data/Code Availability language, exact AIP-compliant AI-tool disclosure, ethics applicability, optional ORCIDs for the other authors, and explicit approval by all authors.
- Exclusive submission is user-confirmed; traditional/subscription publication is planned. Suggested reviewers were not provided and opposed reviewers are none.
- The existing multi-branch repository should remain private; a dedicated Scheme A public release is recommended after revision and license selection.

The current PDF is explicitly a `WORKING MANUSCRIPT FOR REVISION`. With that boundary, the organizational decision is **SUBMISSION_FILES_ORGANIZED_READY_FOR_MANUSCRIPT_REVISION**: the numerical evidence, author metadata, declaration drafts, and submission checklist are organized, but scientific revision and final author approval remain outstanding and no journal submission has been made.

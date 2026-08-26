# Independent reviewer-style audit

## Overall assessment

This draft is now a coherent benchmark/application manuscript rather than a renamed GFNODE paper. Its contribution is a controlled evaluation protocol and an evidence-backed negative result: ModernTCN is the most consistent **neural** model, but none of the neural models has positive mean RMSE skill over last-value persistence across all 24 array–horizon–scope combinations. The paper does not imply that ModernTCN is original, does not revive ODE claims, and does not claim cross-site or deployment generality.

Estimated completion: **85% of a submission-ready scientific manuscript**. The technical narrative, evidence tables, four figures, references, and compiled PDF are complete. Human author/funding approval and institutional journal verification remain.

## Checklist review

1. **Does the paper still imply algorithmic novelty?**
   No. The title, abstract, contribution list, methods, discussion, and conclusion consistently describe an empirical and methodological benchmark. ModernTCN, iTransformer, PatchTST, and the other methods are cited as existing architectures.

2. **Is the contribution sufficient for a benchmark/application article?**
   Potentially yes for JRSE if the editor values reproducibility-oriented application evidence. The strongest contribution is the combination of Train-only preprocessing, split-local windows, Validation-only checkpoint selection, a same-origin persistence challenge, controlled co-location, multiple horizons/scopes, seed variability, and unified GPU efficiency. The paper avoids claiming that any one component is new.

3. **Are unfavorable persistence results fully reported?**
   Yes. Persistence is included in the main table and Figure 2. The Qcells H12 reversal is shown, quantified, and discussed. The negative mean skills (ModernTCN −0.066; discrete candidate −0.444; iTransformer −0.798; PatchTST −1.073) are reported rather than hidden. The text explains why the small persistence denominator makes the macro skill sensitive without dismissing the result.

4. **Do key numbers match Stage-2 evidence?**
   Checked against `FINAL_METRICS_LONG.csv`, `FINAL_EFFICIENCY.csv`, `FINAL_DATASET_METADATA.csv`, and `FINAL_SAMPLE_COUNTS.csv`. The 36/36 run count, window counts, RMSE/range-nRMSE values, neural win counts and ranks, parameters, latency, throughput, and memory values match Stage 2. Stage 2 supersedes all earlier summaries.

5. **Are full/daylight definitions clear?**
   Yes. Daylight is target power above 1% of the array’s Train maximum. It is explicitly evaluation-only and is not described as a causal input, routing feature, or deployable detector.

6. **Is Daily Persistence incorrectly mixed into the main ranking?**
   No. It is described as supplementary because the 24 h lag changes sample availability. Last-value persistence alone is the same-sample primary reference.

7. **Is unknown AC capacity handled correctly?**
   Yes. The manuscript separates official DC nameplate ratings from measured AC Active Power and uses Train-range nRMSE. It does not call this capacity-normalized RMSE.

8. **Are there overextended cross-site or deployment claims?**
   No. The systems are repeatedly described as co-located. Limitations state that the evidence is not cross-climate, cross-region, or deployment validation. GPU timing is expressly hardware-specific.

9. **Is the 18/24 result correctly scoped?**
   Yes. It is consistently “first among neural models” and never represented as 18 wins against persistence.

10. **Are references real and relevant?**
    The bibliography contains 35 entries spanning PV forecasting, multi-horizon time-series architectures, leakage/evaluation, imputation/anomaly tools, and official DKASC material. DOI/official links are included where available. Before formal submission, the authors should run the publisher’s reference check and confirm full author lists for entries intentionally abbreviated with `and others`.

11. **Are figures reproducible?**
    Yes. `build_figures.py` reads the Stage-2 evidence directly, asserts the 3,696-row evidence table and expected model/horizon sets, and generates four vector PDFs. No key result is hand-entered into a figure.

12. **Does LaTeX compile and pass visual inspection?**
    Yes. `latexmk` produced a 12-page PDF with resolved references/citations and no overfull boxes. All pages were rendered to PNG and inspected as a contact sheet. Tables remain inside page bounds; plot labels and legends are legible at the present two-column scale. A benign REVTeX float-placement warning appears even though all four figures and all four tables are present in the rendered PDF; it does not indicate missing content.

13. **Is JRSE in scope?**
    Yes at the subject level: the official scope includes solar photovoltaics, energy meteorology, distributed generation, utility power, and system integration. The reproducibility/benchmark emphasis should be connected to renewable-energy engineering consequences in the cover letter.

14. **Did Stage 1 and Stage 2 conflict?**
    Stage 1 was a planning document; Stage 2 is the final unified evidence. Where provisional Stage-1 descriptions could differ (especially daylight definition, normalized metric wording, or actual model list), the manuscript follows Stage 2. No favorable Stage-1 number was selected over a Stage-2 value.

## Main strengths

- The temporal protocol is unusually explicit and internally consistent.
- Co-location reduces climate/site confounding without being misrepresented as cross-site generalization.
- Persistence is a genuine same-sample challenge rather than a decorative baseline.
- The negative result is useful, nuanced, and not hidden behind average neural ranks.
- Results have artifact-level provenance, three-seed summaries, and common efficiency measurements.

## Three most likely rejection risks

1. **Perceived limited novelty.** A reviewer may view the work as a comparison of existing models on one public dataset. The response must emphasize evaluation rigor, controlled technology comparison, persistence reversal, and reproducible evidence rather than overstate architecture novelty.
2. **Limited external validity.** One site, a short April–August 2018 interval, and three arrays may be judged insufficient for broad forecasting conclusions. The manuscript already narrows its claims, but an editor may still prefer multi-site or multi-year evidence.
3. **Persistence result and data regime.** The extremely low Qcells H12 persistence error may prompt questions about target dynamics, masks, or sample composition. The artifact consistency checks and exact daylight/full definitions should be highlighted, and sample counts should be readily available during review.

## Human actions before direct submission

- Confirm author order, affiliations, corresponding authors, and CRediT roles with all authors.
- Insert verified funding and grant information.
- Verify current SCIE/JCR status through institutional Clarivate access.
- Reconfirm JRSE subscription-route charges and decide whether all figures should remain color in the online version.
- Run a final bibliographic metadata check and replace abbreviated author lists if required by the submission system.
- Prepare cover letter, highlights only if requested by AIP, and any data/code-access statements required by the submission portal.

## Recommendation

Proceed to formal submission-material preparation after the human confirmations above. No new neural-network training is required to complete this JRSE-targeted draft. Additional training should not be started merely to make the benchmark appear algorithmically novel.

## Complete prompt for the next Codex round

> In the isolated `manuscript/clean-pv-benchmark-latex` worktree, finalize the JRSE submission package without changing any numerical result or training a model. Read `manuscript/clean_pv_benchmark/README.md`, `REVIEW.md`, `main.tex`, and the official current AIP/JRSE author instructions. Ask me only for the missing verified author CRediT roles, funding/grant text, and institutional SCIE/JCR confirmation. Then: (1) insert only the confirmed information; (2) perform a full DOI/author metadata check of every cited reference and correct bibliographic metadata without changing the scientific claims; (3) prepare the cover letter and any JRSE-required submission declarations in the same manuscript directory; (4) rebuild figures and `main.pdf`; (5) verify no undefined references, missing citations, overfull boxes, or out-of-page tables; (6) render and visually inspect every PDF page; (7) update `REVIEW.md`; and (8) commit only the manuscript directory to the same branch and update the existing Draft PR. Do not train models, alter Stage-2 evidence, modify master/C1/NWP/old manuscript files, or merge the PR.

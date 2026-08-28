# Independent reviewer-style scientific review

## Recommendation

**Ready for author approval after declaration confirmation; not yet ready for journal upload.** The revision now reads as an independent benchmark/application article rather than a project audit or an algorithm paper. Its defensible contribution is the controlled estimand: causal Train-only preprocessing, split-local windows, elementwise-matched evaluation, two persistence lags, four horizons, two scopes, and unified efficiency measurement across three co-located PV technologies.

No neural-network training was executed during this revision. The existing evidence remains unchanged: the independent verifier passes 4,414/4,414 comparisons (maximum absolute difference 8.51e-12, maximum relative difference 3.23e-11); primary lowest-RMSE wins are 12, 9, 2, 1, and 0 for the inverted-variate, depthwise TCN, joint-patch, discrete recurrent, and Last-value implementations; matched Daily Persistence wins 22/24, with neural wins only for Hanwha H12 full/daylight; Qcells H12 contains 6,463 origins, 77,556 full points, and 36,504 daylight points (47.1%).

## Scientific assessment

### Strengths

1. The research questions distinguish a weak short-lag reference from a stronger 24-hour seasonal reference instead of presenting one persistence definition as universally sufficient.
2. Models and references share origins, labels, target points, and scope masks within each comparison. This makes the 22/24 reversal interpretable as a baseline effect rather than a sample-selection artifact.
3. The co-located design controls much of the site/weather context while appropriately avoiding causal attribution of differences to module technology alone.
4. The architecture-identity table makes clear that the four forecasters are compact project implementations inspired by, but not equivalent to, full iTransformer, PatchTST, or ModernTCN releases.
5. Negative evidence is central rather than hidden: neural forecasts beat Last-value Persistence, while Daily Persistence usually remains better.
6. Main and supplementary figures/tables are generated from the authoritative long-format evidence, with complete per-seed/support results moved out of the main text.

### Literature position

The revised comparison covers SolNet (geographic/transfer breadth), the 2025 review (standardization, generalization, and uncertainty), Markovics et al. (multi-site NWP benchmark and tuning), Dhingra et al. (missing-data and probabilistic evidence), Cross-Unet (history plus future weather), Mansour et al. (high-resolution Yulara seasonal/stress benchmark), and Alice Springs precedents. The paper does not claim those studies are flawed; it states that their information sets, splits, lags, and eligible samples differ, so errors are not directly comparable. The remaining niche is narrow but coherent: elementwise-matched history-only evaluation across co-located technologies with weak/strong persistence and hardware-specific efficiency.

### Interpretation and limitations

The Last-value and Daily-Persistence conclusions are compatible. The former measures improvement over local continuity; the latter tests whether a history-only learner beats daily recurrence. The two Hanwha H12 exceptions are genuine but do not support a general neural advantage. The 24 array–horizon–scope combinations are correlated summaries, not 24 independent experiments, and no unsupported p-values are introduced.

The estimand is limited to one Alice Springs facility, April–August 2018, history-only inputs, compact implementations, and three random seeds. No future NWP, probability intervals, cross-climate validation, or official architecture reproduction is evaluated. Test was excluded from fitting/training/checkpoint selection but was repeatedly audited during development; it is not an untouched external confirmation set.

## Production audit

- The main article and Supplementary Material compile independently.
- The abstract is one paragraph and below 250 words.
- Four main figures and four main tables provide the core story; the rank heat map and full result tables are supplementary.
- Vector figures use embedded fonts, color-blind-friendly encodings, consistent scope/baseline semantics, and programmatic evidence sources.
- Alt text is supplied separately for each figure and main table.
- No missing citations, undefined references, or content-overflow defects remain; ordinary underfull warnings do not impair layout.
- Both PDFs were rendered page by page and inspected for labels, legends, tables, and whitespace.

## Most likely rejection risks

1. **Incremental methodological novelty.** The contribution is evaluation discipline and controlled evidence, not a new forecaster.
2. **Limited external validity.** One co-located facility and a short seasonal interval restrict generalization.
3. **Weak deployment value of the neural models.** Daily Persistence wins 22/24 matched comparisons, so operational benefit is not established.
4. **Implementation scope.** Reviewers may request full official model reproductions or broader baselines, which this compact-mechanism study intentionally does not provide.

## Author confirmation and release readiness

Funding, CRediT, conflict-of-interest, AI-use wording, ethics applicability, code/data release wording, and final approval remain `FINAL_AUTHOR_CONFIRMATION_REQUIRED`. These drafts must not be treated as approved statements. Current indexing and optional/conditional fee details also require author or library confirmation.

Public release status is **PUBLIC_RELEASE_REQUIRES_ACTION**. GitHub reported the existing multi-branch repository as public on 2026-08-28; no visibility change was made in this revision. That repository is not a curated Scheme A release. A dedicated repository is preferable, but license selection, dependency/environment documentation, path-independent inputs, DKASC redistribution/download instructions, and a final secret/path scan remain outstanding.

## Decision

Subject to the author-owned confirmations above, the scientific and production revision supports **SCHEME_A_JRSE_REVISION_READY_FOR_AUTHOR_APPROVAL**. It does not authorize journal submission, public release, or completion of any unconfirmed declaration.

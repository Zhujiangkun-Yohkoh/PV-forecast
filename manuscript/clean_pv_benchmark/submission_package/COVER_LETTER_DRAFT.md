# Cover letter draft — author confirmation required

**Target:** Journal of Renewable and Sustainable Energy (JRSE), Research Article
**Manuscript:** “Leakage-Aware Multi-Horizon Benchmarking of Deterministic PV Forecasts Across Co-Located Technologies”

Dear Editor,

Please consider the enclosed manuscript for publication as a Research Article in the *Journal of Renewable and Sustainable Energy*. The study presents a leakage-aware, sample-matched application benchmark of deterministic photovoltaic power forecasts across three co-located arrays at the DKA Solar Centre in Alice Springs.

The manuscript does not propose a new neural architecture. Its contribution is a controlled and auditable evaluation: all preprocessing is fitted on Train only, windows are confined to temporal splits, checkpoints are selected on Validation, and neural forecasts are compared with causal persistence references on identical target masks. Four compact project implementations produce a common 12-hour trajectory whose 1-, 4-, 8-, and 12-hour prefixes are evaluated under full-timeline and daylight scopes.

The results are deliberately reported without selective favorable framing. Learned models outperform Last-value Persistence in the primary matched comparisons, but exact-lag Daily Persistence outperforms the best neural implementation in 22 of 24 separately matched comparisons. This negative but operationally useful finding defines where model complexity does and does not add value under a strict protocol. The paper therefore fits JRSE’s scope in solar photovoltaics, energy meteorology, and renewable-energy engineering through its evidence on reliable forecast evaluation and technology-conditioned performance.

The submission package includes the compiled manuscript. Source data remain available from the DKA Solar Centre under the provider’s terms; the evaluation code and aggregate evidence provenance are described in the manuscript and repository.

Before upload, the corresponding author must personally confirm and add the following statements; this draft does not assert them:

- [AUTHOR CONFIRMATION REQUIRED: all authors approved the manuscript and submission]
- [AUTHOR CONFIRMATION REQUIRED: the work is not under consideration elsewhere]
- [AUTHOR CONFIRMATION REQUIRED: Conflict of Interest disclosure]
- [AUTHOR CONFIRMATION REQUIRED: related manuscripts/preprints and how they differ]
- [AUTHOR CONFIRMATION REQUIRED: corresponding-author name, affiliation, email, and signature]

Sincerely,

[AUTHOR CONFIRMATION REQUIRED: corresponding author]

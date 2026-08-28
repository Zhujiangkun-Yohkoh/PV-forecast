# Cover letter draft - final author confirmation required

**Target:** Journal of Renewable and Sustainable Energy (JRSE), Research Article  
**Manuscript:** “Leakage-Aware Multi-Horizon Benchmarking of Compact Neural PV Forecasts Across Co-Located Technologies”
**Corresponding authors:** Zhicong Chen (`zhicong.chen@fzu.edu.cn`) and Lijun Wu (`lijun.wu@fzu.edu.cn`)

Dear Editor,

Please consider the enclosed manuscript for publication as a Research Article in the *Journal of Renewable and Sustainable Energy*. The study presents a leakage-aware, sample-matched application benchmark of deterministic photovoltaic power forecasts across three co-located arrays at the DKA Solar Centre in Alice Springs.

The manuscript does not propose a new neural architecture. Its contribution is a controlled and auditable evaluation: all preprocessing is fitted on Train only, windows are confined to temporal splits, checkpoints are selected on Validation, and neural forecasts are compared with causal persistence references on identical target masks. Four compact project implementations produce a common 12-hour trajectory whose 1-, 4-, 8-, and 12-hour prefixes are evaluated under full-timeline and daylight scopes. A separate Supplementary Material file reports complete per-seed results, support counts, model configurations, and sensitivity analyses.

The results are deliberately reported without selective favorable framing. Learned models outperform Last-value Persistence in the primary matched comparisons, but exact-lag Daily Persistence outperforms the best neural implementation in 22 of 24 separately matched comparisons. This negative but operationally useful finding defines where model complexity does and does not add value under a strict protocol. The paper therefore fits JRSE's scope in solar photovoltaics, energy meteorology, and renewable-energy engineering through its evidence on reliable forecast evaluation and technology-conditioned performance.

The authors confirm that this manuscript is not simultaneously submitted to, and is not currently under consideration by, another journal. Source data remain available from the DKA Solar Centre under the provider's terms; the evaluation code and aggregate evidence provenance are described in the manuscript and repository.

Authors, in order: Jiangkun Zhu, Mengling Yang, Zhicong Chen, and Lijun Wu.

Before upload, the corresponding authors must personally confirm:

- `FINAL_AUTHOR_CONFIRMATION_REQUIRED`: all four authors approve the revised manuscript and submission;
- `FINAL_AUTHOR_CONFIRMATION_REQUIRED`: the Funding, Conflict of Interest, CRediT, AI-use, Data Availability, and Code Availability wording is accurate for this manuscript;
- `FINAL_AUTHOR_CONFIRMATION_REQUIRED`: related manuscripts or preprints are fully disclosed;
- `FINAL_AUTHOR_CONFIRMATION_REQUIRED`: one or both corresponding authors sign the final letter.

No editor name, suggested reviewer, or opposed reviewer has been invented.

Sincerely,

Zhicong Chen and Lijun Wu  
`FINAL_CORRESPONDING_AUTHOR_SIGNATURE_REQUIRED`


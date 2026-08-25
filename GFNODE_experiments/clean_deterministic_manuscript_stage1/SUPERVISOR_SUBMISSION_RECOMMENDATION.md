# Supervisor submission recommendation

## Decision

**READY_AFTER_NONTRAINING_COMPLETION**

The old GFNODE manuscript should not be revised again. The project does, however, contain enough reliable material for a different paper: a clean deterministic PV forecasting benchmark/application study. The fastest defensible route is to stop GPU work, complete a short list of metric, metadata and figure tasks, and then write a new manuscript for **Solar Energy Advances**.

## Why the GFNODE manuscript cannot continue

The Stage 0 audit found global preprocessing before split, overlapping temporal windows, Test involvement in model selection, and loss of the physical 5-minute axis after night removal. The implemented ODE did not use time in its vector field and did not support the continuous-time claims made in the manuscript. Original Tables 9–13 and Figures 10–14 do not have a clean, reproducible source chain.

The corrected time-conditioned GFNODE then failed its predeclared real experiment: it improved Qcells at H144 but worsened Sanyo by 9.42% and lost all three Sanyo seeds. Later attempts—discrete candidate, shared-private arrays, high-frequency dynamics, probability intervals, joint Ramp events, gradient conflict handling, frozen hazard heads and ALICD—did not produce a viable new forecasting method. These results must not be repackaged as GFNODE support.

## What the new paper actually contributes

The new paper will not claim an original neural network. ModernTCN, iTransformer and PatchTST are existing methods. The contribution is instead:

1. a timestamp-faithful, leakage-free evaluation protocol with Train-only preprocessing, split-local windows and Validation-only selection;
2. a paired direct-H144 comparison across three co-located PV technologies, common horizons and three seeds;
3. an engineering analysis of horizon/technology dependence, uncertainty and implementation cost;
4. a Validation-led diagnosis of high-change trajectory smoothing and the bounded value of past high-frequency irradiance for point forecasting.

The clean benchmark contains 36 completed model–dataset–seed runs and 36 saved H144 prediction files. Four models are genuinely complete: ModernTCN, iTransformer, PatchTST and the Discrete Candidate. DLinear and TimesNet must not be listed as completed clean baselines. ModernTCN currently has mean RMSE rank 1.25 and wins 10/12 tested array–horizon combinations, but this must be worded as “best among the four tested implementations,” never “state of the art.”

## Journal recommendation

### Target: Solar Energy Advances

Its official scope explicitly includes solar-energy measurements, monitoring protocols, data analytics, artificial intelligence in solar systems and forecasting. This is the closest fit to a rigorous protocol/application paper and does not require us to invent a new architecture.

Main risk: a local four-model application can still be desk-rejected. The manuscript must emphasize transferable evaluation methodology, paired co-located comparison and error/information-boundary findings.

Official page: https://www.sciencedirect.com/journal/solar-energy-advances

### Backup 1: Journal of Renewable and Sustainable Energy

This AIP journal accepts broad renewable-energy engineering research and currently publishes photovoltaic forecasting work. It is a credible application venue, although recent forecasting papers raise the methodological-competition bar.

Official page: https://pubs.aip.org/aip/jrse

### Backup 2: IET Renewable Power Generation

Its official scope explicitly includes photovoltaics, forecasting and validated modelling. It is SCIE- and Scopus-indexed according to the official page, but its stated expectation of significant novelty/general applicability makes it the highest-risk option.

Official page: https://ietresearch.onlinelibrary.wiley.com/journal/17521424

Applied Soft Computing is not recommended for the current evidence. A future ASOC submission would require a new research project and genuinely new predictive information, not another decoder or loss adjustment.

## Is the evidence sufficient?

Yes for a benchmark/application manuscript after a bounded nontraining completion step. The core results, seeds, predictions, labels, metrics, model configurations and protocol tests exist. The following gaps prevent immediate final prose:

- official Site 17/25/38 metadata, coordinates, units, licensing and the relationship between authoritative downloads and exact 2018 benchmark files must be reconciled;
- all 36 prediction artifacts must be reloaded once to regenerate the final metrics, rankings, evaluated-target counts and fairness checks;
- a persistence reference should be calculated on the same origins;
- latency must be measured under one documented environment;
- all tables and figures must be rebuilt from clean artifacts.

None of these tasks requires neural-network training. The 2022 Test period has already been repeatedly inspected, so it should remain an exploratory secondary check rather than a supposedly untouched confirmation set.

## Required work before writing

1. Complete authoritative metadata and data-lineage reconciliation.
2. Produce one reproducible nontraining analysis script for metrics, ranks and artifact equality.
3. Add persistence/seasonal-persistence skill and standardized inference timing.
4. Rebuild the six proposed tables and six minimal figures from clean evidence.
5. Update model, DKASC and evaluation-protocol citations.
6. Recheck the live Solar Energy Advances author guide and indexing/APC details before formatting.

No GPU training is recommended. New neural baselines, additional seeds, hyperparameter searches and new architectures would delay the fastest viable submission and are not needed to establish the proposed contribution.

## Unique recommended action

Approve the clean deterministic benchmark/application manuscript route with **Solar Energy Advances** as the target. Authorize only the nontraining completion tasks above, then start a wholly new manuscript. Do not edit the old GFNODE manuscript and do not resume model iteration.


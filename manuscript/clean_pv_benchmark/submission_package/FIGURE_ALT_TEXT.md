# Current figure captions and alt text

## 1 — fig1_design

Every future target at origin plus j times five minutes uses its exact minus-24-hour historical counterpart for Daily or Ridge B. The six-hour history contains 72 samples from minus 355 minutes through origin; the twelve-hour target sequence begins at plus five minutes.

Alt: Three non-pooled information intervals show all previous-day values available by origin. Exact endpoints distinguish sample count from elapsed span.

## 2 — fig6_ridge_intervals

Expanded-grid Ridge B minus each reference over cumulative 12 h windows. Negative differences favor B. Bars are 95% paired 48 h origin-block intervals, 2000 draws; Inverted contrasts average three seed-specific effects, not predictions. Identical complete-H144 and Daily-finite target support is used within each system/scope. Scales differ between panels. Original-grid intervals remain in the supplementary evidence.

Alt: All five systems are shown. Yulara expanded B has intervals crossing zero against Daily and Inverted; Qcells deterioration relative to original B is preserved in the grid sensitivity data.

## 3 — fig7_qcells_selection

Qcells Test: 6463 one-hour origins become 2996 complete-twelve-hour origins. Exclusive attribution assigns 3350 removals to future negative labels and 117 to the tail boundary. Active one-hour pairs decrease from 36504 to 42, involving 25 unique active timestamps and 25 origins.

Alt: The frozen negative-label rule and continuous-window requirement select a strongly different hourly and power distribution; complete-window support is not inherently representative.

## 4 — fig5_decomposition_case

Left: additive active/low-power contributions to full MSE on NIST twelve-hour Daily-matched targets; neural contributions are averaged over seeds. Active and low-power Inverted-minus-Daily MSE contributions are -420.64 and +655.38 kW²; net +234.74 kW². Right: seed42 at 2017-10-15 10:00 fixed EST, chosen by the calendar rule, not a rolling reconstruction.

Alt: Stacked squared-error contributions show active error savings offset by low-power excess. A fixed October origin compares observed, neural and Daily trajectories without claiming a weather event.

## 5 — fig2_alice_references

Post hoc minimum three-seed mean neural RMSE relative to both references on the same Daily-valid targets. F: full; A: power-active. The envelope is descriptive, not deployable.

Alt: Twenty-four Alice comparisons on shared logarithmic axes: both references use the Daily intersection. Ratio one is the comparison boundary; unfavorable ratios remain visible.

## 6 — fig3_external_skills

Inverted-variate mean seed skill with 95% paired temporal-block intervals: 48 h blocks, 2000 resamples. These are conditional time-sample intervals, not seed SD. Each reference has its own matched targets.

Alt: Full and power-active effects include intervals crossing zero at long windows; small training-seed SD does not replace temporal uncertainty.

## 7 — fig8_fixed_period

Same-site next-year evaluation, origins fixed to 1 April–30 June 2018; cumulative 12 h windows. All methods use the same finite target/Last-value/Daily intersection in each panel. Original 2017 transformations, weights, and both saved Ridge grids are unchanged. Points and 95% paired 48 h block intervals condition on frozen predictions. Negative differences favor the named method over Daily. Panel scales differ.

Alt: The next-year period changes comparisons: original and expanded Ridge grids remain distinct and all adverse effects are shown. Neither seasons nor geographic sites are treated as exchangeable repeats.

## S1 — figS11_yulara_diagnostic

Original-grid Yulara prediction ranges (thin), 1st–99th percentiles (thick) and medians (markers), over saved candidate trajectories. The dotted line is the Train target maximum, not clipping. Missing temperature at origin occurs in 4/69333 Train origins (0.0058%). These diagnostics support a missing-pattern extrapolation explanation, not an identified unique cause.

Alt: Original A and B adverse ranges remain visible. The different denominators for candidate prediction distributions and headline matched complete windows are explicit.

## S2 — figS12_alpha_sensitivity

Uniform expanded alpha grid 10^-4 through 10^8, selected solely by full-H144 Validation MSE for all five systems and both information sets. Original and expanded Test errors use identical Daily-matched full targets; all ten comparisons are shown.

Alt: Separate scales retain the unfavorable original Yulara errors and every sensitivity result. The added grid is post hoc, not a silent replacement of the original fit.

## S3 — figS10_qcells_support

Historical Qcells support. Unequal-width power bins are categorical points, not a continuous density. Each power series is normalized by its target-pair count in the displayed bins. Origin-hour series use all origins in the named support. Repeated target times remain overlapping forecast pairs.

Alt: Three split rows compare one-hour, complete-twelve-hour and removed support without interpreting unequal bin widths as density.

## S4 — figS2_common_origins

Historical horizon-specific versus common complete-twelve-hour origin support, full target range. Bars show seed SD. Separate reference and neural panels retain full error ranges and readable details. Common-origin selection does not isolate window length from the selected population.

Alt: All internal support abbreviations are expanded. Qcells common origins select a different power distribution; this is not the new S1/S2 restored-origin error comparison.

## S5 — figS4_fixed_leads

Lead-specific error at each five-minute lead on common complete-H144 origins and Daily-valid targets. These are not cumulative-prefix metrics. Neural curves average per-seed RMSE.

Alt: Five system panels show fixed-lead curves on one origin set, distinguishing them from cumulative windows.

## S6 — fig4_neural_gaps

Relative RMSE gaps from the best mean neural implementation in each cumulative window and power scope. Zero identifies the lowest error; small gaps are shown as small gaps rather than enlarged rank differences.

Alt: Four model curves show NIST full and power-active winners differing across windows, while Yulara favors Inverted-variate. Zero-valued points have visible lower margins; gap = 100 times (RMSE/best neural RMSE minus one).

## S7 — figS1_alice_ranks

Alice neural ranks by cumulative window and target-power scope. F: full; A: power-active; four discrete rank levels.

Alt: Three facets retain co-located arrays separately and show discrete numerical ranks for four models.

## S8 — figS3_latency

Historical 17-channel inference timings on RTX 3060 Laptop GPU, excluding loading/preprocessing. Point summaries do not establish a stable difference between 0.535 and 0.558 ms; no new timing experiment.

Alt: A logarithmic latency axis displays all four models and their parameter counts without compressing sub-millisecond values into zero.

## S9 — figS5_block_intervals

Paired temporal-block percentile intervals (2000 resamples; 24/48/72 h). SSE/count is aggregated before each seed RMSE, then seed skills are averaged. These intervals are conditional on observed dates and fitted models, not prediction intervals.

Alt: Temporal intervals expose uncertainty around small long-window Daily improvements. All three prechosen block sizes are shown, including intervals crossing zero.

## S10 — figS6_monthly

Monthly errors assign each full twelve-hour trajectory to its forecast-origin month on common H144 Daily-valid support. Bars show seed SD. O denotes origins and N scored origin--lead target pairs, printed for each month.

Alt: October, November and December are compared without assigning months by target time or treating them as independent climate replications.

## S11 — figS7_cases_fixed

Fixed calendar seed42 trajectories. Fixed cases use the first eligible origin on or after each month’s 15th at 10:00; extremes use original Daily SSE differences and are not representative frequencies.

Alt: Readable site labels distinguish provider-local Yulara and fixed-EST NIST. Each twelve-hour curve comes from one origin; no weather events are inferred.

## S12 — figS7_cases_extremes

Post hoc diagnostic extremes seed42 trajectories. Fixed cases use the first eligible origin on or after each month’s 15th at 10:00; extremes use original Daily SSE differences and are not representative frequencies.

Alt: Readable site labels distinguish provider-local Yulara and fixed-EST NIST. Each twelve-hour curve comes from one origin; no weather events are inferred.

## S13 — figS8_ridge

ORIGINAL GRID. Recent-history Ridge and Ridge plus available previous-day trajectory, with original Validation-selected alphas. Daily-matched full targets use the historical horizon-specific support. Ridge is deterministic; neural bars are seed SD, not temporal intervals. Expanded-grid results are separately labeled.

Alt: All original-grid outcomes, including large Yulara errors, are retained without presenting them as the final expanded linear reference.

## S14 — figS9_learning_budget

Historical NIST TCN learning records. All three selected checkpoints occur at the fixed maximum budget of 25 epochs. These curves document the budget boundary, not convergence or a new fit.

Alt: Three historical training and validation curves expose continued improvement at the budget boundary; no extended training was performed.
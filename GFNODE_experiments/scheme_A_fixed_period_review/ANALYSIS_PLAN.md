# Frozen-prediction support and fixed-period review

Recorded on 2026-09-09, before computing any 2018 external-period errors. Source commit: ad59ed234de457f251ea16bfb51521f976c8fd83. No neural training, processor fitting, alpha search, or replacement of historical evidence is permitted.

## A. Expanded Ridge uncertainty
Use the saved original and expanded grids, all five systems, complete H144 origins on a common Daily-valid point intersection. Full, power-active (true target > 1% of original Train maximum), and complementary low-power targets. Differences are first-named minus reference RMSE in kW; skill is 1 minus first/reference RMSE. Blocks follow forecast-origin time, anchored at the original Test split start. Primary 48 hours; sensitivity 24 and 72 hours; 2,000 paired bootstrap draws with seed 20260908. Retain empty blocks; report draws with zero total support. Aggregate SSE and count before RMSE; average neural seed effects, never average predictions. The same draws are used for every method and seed.

## B. Alice scoring support
Read only all 36 checkpoints and original Train-fitted transformations. First reproduce predictions at saved Test origins. Preserve input transformations, including finite negative numerical inputs, and the frozen nonnegative-origin Last-value and nonnegative Daily reference construction.

Within the original Test dates, candidate origins require 72 regular-grid historical rows and at least one following target within Test. No future sign controls prediction generation. Targets outside Test remain missing; rows are never deleted to concatenate time. At each prefix compare historical complete-nonnegative-window support, S1 (individual finite nonnegative targets), and S2 (individual finite raw targets). For each sensitivity identify origins absent from historical support separately. Main all-method comparisons also require the original valid Last-value and Daily point intersection; give pre-intersection support separately. Evaluate H12 and H144, all three power scopes. The change concerns scoring support, not retraining or corrected physical interpretation of undocumented negative values.

Report actual neural versus Ridge Train/Validation origin rules separately. The neural rule does not require a valid origin label; Ridge does. Historical retention is for traceability, not proof that negative-value exclusion is justified.

## C. Same-site next-year fixed period
For Yulara and NIST, forecast origins are fixed from 2018-04-01 00:00 through 2018-06-30 23:55 in their respective frozen coordinates. Obtain raw data from 2018-03-30 through 2018-07-01 inclusive as buffers: the last target is 2018-07-01 11:55. Targets can extend beyond June; this is an origin-defined period. NIST retains fixed UTC−05:00 and strict [T−5 min,T) aggregation. Yulara retains provider-local coordinates and raw timestamp +5 min availability. No DST or inferred UTC is introduced.

Use original 2017 neural processors, Inverted seeds 42/43/44, and saved original/expanded Ridge coefficients and their saved processors. Replay original 2017 origins first. Do not fit any object on 2018, change thresholds, replace dates, or select methods from new scores. H144 full and active are primary; low-power explanatory, H12/H48/H96 secondary. Use the paired block scheme above, anchored at April 1. Audit coverage, identity, units, equipment-event evidence, and missing-pattern exposure independently of score. An unavailable NIST source is a documented partial result, not permission to substitute another period.

## Interpretation and delivery
All additions are post hoc analyses of a developed project. Time intervals condition on frozen predictions and do not cover retraining uncertainty or remove every overlapping-trajectory dependence. No interval crossing zero is called equivalence. Preserve prior outputs, report actual versus historical verification, revise affected manuscript material, then commit and create full, lite, and figure archives with manifests, CRC, extraction, and applicable light verification.

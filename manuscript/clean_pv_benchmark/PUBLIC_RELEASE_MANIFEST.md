# Dedicated public release manifest

## Recommendation

`PUBLIC_RELEASE_REQUIRES_ACTION` — `DEDICATED_PUBLIC_RELEASE_RECOMMENDED`

GitHub's repository API reported the existing `PV-forecast` repository as **public** on 2026-08-28. This revision did not alter visibility. Repository visibility applies to every branch, commit, open Draft PR, and historical object, not only Scheme A. The repository contains unrelated C1 and NWP research branches, multiple Draft PRs, historical absolute workstation paths in experiment reports/configurations, and manuscript/submission materials. It therefore must not be represented as a curated Scheme A reproducibility release; the owner should separately review the already-public exposure.

After author approval of the manuscript and release scope, create a separate repository provisionally named `leakage-aware-pv-benchmark`. This task does not create that repository and does not rewrite the existing repository's history.

## Files proposed for the dedicated repository

- Scheme A benchmark execution and evaluation code.
- `config.json` after replacing local path assumptions with documented command-line/environment inputs.
- Ordinary protocol tests that can run without unpublished results.
- `independent_verify_evidence.py`, with a documented artifact-dependent mode.
- `corrected_metrics.csv` and the evidence report, subject to author approval.
- A new public README with environment setup, DKASC official download instructions, split/window definitions, commands, limitations, and manuscript citation.
- A dependency/environment specification generated and reviewed for the release.
- Data and artifact schemas sufficient to reproduce evaluation when users obtain data and produce their own runs.

`LICENSE_SELECTION_REQUIRED_BEFORE_PUBLIC_RELEASE`: no software or documentation license is selected by this task.

## Exclusions

- Raw DKASC or NWP data and copies of provider files.
- Local `results/`, checkpoints, NPZ prediction arrays, and caches.
- Absolute workstation paths, usernames, or local directory layouts.
- Manuscript drafts, cover letters, author emails/ORCIDs, submission checklists, and submission-system information.
- Unrelated Scheme C1, NWP, GFNODE, ramp, probabilistic, or failed-model experiments.
- Downloaded third-party papers or any file whose redistribution permission has not been confirmed.

## Audit findings supporting this recommendation

- Remote refs: 11 named branches plus `origin/HEAD`; no tags were present at audit time.
- Open pull requests: PRs #1 through #10, all Draft, spanning NWP, Scheme C1, and Scheme A.
- Git objects larger than 20 MiB: none.
- Tracked PDF history contains only Scheme A manuscript/figure PDFs, but those submission materials are not part of the intended code release.
- No high-confidence API-key, token, password, GitHub credential, or private-key pattern was found by the all-ref filename/content scan.
- No tracked checkpoint, NPZ, GRIB, NetCDF, or raw-data copy was found in Git history; tracked CSVs are aggregate inventories/metrics, not raw arrays.
- The all-ref deletion scan found no deleted path in repository history, so it revealed no hidden deleted credential, data, checkpoint, or restricted-PDF candidate.
- Absolute local paths occur throughout historical experiment reports/configurations and in the pre-sanitized independent-audit commit. The current audit JSON removes its two display paths, but historical commits remain immutable without a prohibited history rewrite.
- There is no repository-root `.gitignore`; several experiment-local ignore files cover `results/`, `*.pt`, `*.npz`, and caches. A dedicated release should use a reviewed root ignore policy.
- Ordinary Scheme A tests do not require local `results/` but do require user-obtained source CSVs. The 36-run artifact-integrity tests and regenerated independent evidence audit require unpublished local checkpoints/NPZ artifacts and must be documented as maintainer-only verification unless release artifacts are separately approved.

## Preconditions for public release

1. Complete the next scientific manuscript revision and freeze the public code/evidence scope.
2. Select a license with all rights holders' approval.
3. Replace local path conventions with explicit documented inputs.
4. Add a root `.gitignore` and dependency specification in the new repository.
5. Verify DKASC redistribution terms and publish download instructions rather than data copies.
6. Remove submission-only and private author metadata from the release tree.
7. Run secret/path scans on the new repository before changing its visibility.

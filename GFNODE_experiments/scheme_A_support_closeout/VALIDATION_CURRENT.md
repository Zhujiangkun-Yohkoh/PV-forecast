# Actual closeout verification

Date: 2026-09-09. Commands run from repository root with documented runtime and ignored .local/closeout_paths.json; heavy paths are not committed.

- analysis_saved.py: 36 aligned original/replay arrays and 216 metric-impact rows; original tolerance retained; accepted/diagnostic methods separated.
- batch_probe.py: five strict checkpoint and batch-configuration forward probes; no fit or weight write.
- period_forward.py: six external neural and eight Ridge original replay checks, then new 2017 candidates with frozen state. Initial mixed-batch Yulara43 diagnostic remains in the plan/report. Canonical partition recalculates original batches and new origins separately; it never splices archived predictions.
- reduce_periods.py: four cells from new 2017/existing 2018 arrays.
- audit_points.py: 1836 metric rows / 114 support groups, failed=0, skipped=0.
- independent_verify.py / verify_light.py: 6120 effect/interval rows / 342 groups, independent block-multiplicity implementation, failed=0, skipped=0. These checks are not independent experiments.
- protect.py: 932 protected files unchanged in SHA-256, size and mtime_ns.
- latexmk -pdf -interaction=nonstopmode -halt-on-error: both PDFs compiled in manuscript/clean_pv_benchmark to .local/tex-closeout-main and tex-closeout-supp. An initial wrong-cwd invocation stopped before compilation; correct-cwd rerun succeeded.
- pdf_qa.py: main 12 / Supplement 47 pages rendered and viewed; embedded fonts, no rasterized figures, undefined references, duplicate labels or overfull boxes. Current figures have six assets each.

Historical M1-R/M2 full suites and original training were not rerun. No neural training, preprocessing refit, alpha search or submission. Historical Alice neural processors were not separately saved; tiny metric effects do not prove exact historical input identity. Training-support selection remains.

Archive CRC, full fresh extraction, manifest SHA/size and extracted lightweight arithmetic are executed after commit and recorded in package sidecars. Archive verification is not neural replay or full training.

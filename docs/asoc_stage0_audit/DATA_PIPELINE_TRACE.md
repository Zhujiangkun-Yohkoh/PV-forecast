# Data pipeline trace

Authoritative path is PVforecast16/GFNODE_experiments/gfnode_solo_benchmark.py. Executable flow is CSV raw data → pd.to_datetime timestamp → PR fraction conversion → full-file radiation thresholds → centered 5-row night filter → PR filter → global KNN fit/transform → global Isolation Forest fit/predict → global feature/target MinMax fit/transform → row-index sliding windows → chronological split → model. Evidence: gfnode_solo_benchmark.py:36-83,605-621; duplicated in fair_sota_comparison.py:69-131,1055-1075 and seasonal_robustness_experiment.py:55-87,167-187.

1. Split before preprocessing? NO — BLOCKER. All preprocessing/scaling precedes windows/split (gfnode_solo_benchmark.py:51-83,609-615).
2. MinMax fit range? Entire post-IF series including future test rows — BLOCKER (:70-75).
3. KNN fit range? Entire daytime/PR series including future test rows — BLOCKER (:64-68).
4. Isolation Forest fit range? Entire imputed series including future test rows — BLOCKER (:44-49,69).
5. Validation? Benchmark/fair/seasonal: NO, only train/test (:611-621; fair_sota_comparison.py:1063-1075; seasonal_robustness_experiment.py:175-187). Optuna has 70/15/15 array slices (optuna_optimization_v2.py:180-212,273-301).
6. Optuna metric? Separate middle-15% validation loss (optuna_optimization_v2.py:180-253), but supplied arrays were already globally preprocessed/scaled.
7. Shared time points? YES — BLOCKER. Windowing before split shares 71+H original row positions.
8. Purge gap? NO — BLOCKER (gfnode_solo_benchmark.py:609-621).
9. Shuffle? Only training DataLoader minibatch shuffle, not pre-split time order (:617-621).
10. Adjacent filtered rows always 5 min? NO — BLOCKER; exact observed deltas are in TIMESTAMP_AND_SPLIT_AUDIT.csv.
11. Windows cross sunset/sunrise or dates? YES. create_sequences uses only contiguous rows, no timestamp check (:78-83).
12. Does H equal real H×5 minutes? NO — BLOCKER. horizon*5/60 is only a label (:605-607); gaps remain.

Fit scopes: radiation thresholds full raw CSV (:36-42,55-63); KNN all retained rows (:64-68); IF all imputed rows (:44-49,69); both MinMax all post-IF rows (:70-75). The paper reports an 80/20 split (PV_improve_v1/Final_Submission/Paper_md版本/4_Experiments_and_Results_Analysis.md:28,121,186) but does not disclose these fit scopes, test selection, overlap, irregular time, or no purge.


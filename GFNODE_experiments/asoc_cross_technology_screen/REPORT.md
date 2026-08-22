# Cross-technology synchronized ModernTCN screen

Verdict: **FAIL**. Six new runs completed (Joint Early-Fusion and Shared–Private, seeds 42/43/44). Nine Independent ModernTCN runs were reused after checking the fixed date splits, no historical Active_Power input, H144-prefix evaluation, and elementwise labels at each technology's complete-H144 test starts.

The synchronized training set preserves the complete regular 5-minute timeline. Shared weather is stored once (six configured weather variables); each technology receives only its Performance_Ratio, missing marker, and Isolation-Forest marker. All preprocessing remains train-only. Targets retain independent validity masks; masked MSE is calculated per technology and averaged equally. Validation selects checkpoints; Test is evaluation-only.

## Result

Regular-full-timeline three-seed mean nRMSE: Shared–Private versus Independent was worse in all 12 technology-horizon comparisons. Its macro nRMSE change was -16.331% (an increase in error), with mean changes Sanyo -16.567%, Hanwha -14.686%, and Qcells -17.924%. H144 improved in 0/3 technologies. Versus Joint Early-Fusion it was worse by 1.962% macro nRMSE and non-inferior in only 4/12 combinations.

Parameter counts: Joint Early-Fusion 1,496,832; Shared–Private 55,257,000; one Independent ModernTCN 682,896; three independent models 2,048,688. Shared–Private therefore also violates the required total-parameter condition.

All six status entries are completed. `metrics.csv` contains per-seed RMSE, MAE, R2, nRMSE and both regular/daylight scopes for all three models; all horizons are prefixes of one H144 prediction. Local artifacts contain the predictions, labels, timestamps, masks and retained validation checkpoints. Completed-run `last.pt` files were reclaimed during the disk-space recovery; this does not affect evaluation or checkpoint selection.

## Screening conditions

1. Independent macro nRMSE improvement >=2%: failed (-16.331%).
2. Improvement in >=8/12 combinations, no technology average worsening >1%, and H144 improvement >=2/3: failed (0/12, all three worsen, 0/3 H144).
3. Better than Early-Fusion macro nRMSE and non-inferior >=7/12: failed (-1.962%, 4/12).
4. Fewer parameters than three independent models: failed (55,257,000 > 2,048,688).

No further Shared–Private variants should be developed. The single next recommendation is to retain independent per-technology forecasting as the research control and redirect the paper to a scientifically distinct, evidence-backed question rather than cross-technology shared/private capacity expansion.

Metadata: the project records the arrays as Alice Springs co-located PV data; authoritative component technology, capacity, Active_Power unit, and coordinates remain UNKNOWN in the available DKASC/project evidence and are not inferred from filenames.

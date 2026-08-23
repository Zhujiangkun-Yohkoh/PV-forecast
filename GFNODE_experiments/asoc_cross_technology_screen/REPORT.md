# Cross-technology synchronized ModernTCN screen

Verdict: **FAIL_CURRENT_IMPLEMENTATION / NO_GO_FOR_SUBMISSION**. Six new runs completed (Joint Early-Fusion and Shared–Private, seeds 42/43/44). Nine Independent ModernTCN runs were reused after checking the fixed date splits, no historical Active_Power input, H144-prefix evaluation, and elementwise labels at each technology's complete-H144 test starts.

The synchronized training set preserves the complete regular 5-minute timeline. Shared weather is stored once (six configured weather variables); each technology receives only its Performance_Ratio, missing marker, and Isolation-Forest marker. All preprocessing remains train-only. Targets retain independent validity masks; masked MSE is calculated per technology and averaged equally. Validation selects checkpoints; Test is evaluation-only.

## Result

Regular-full-timeline three-seed mean nRMSE: Shared–Private versus Independent was worse in all 12 technology-horizon comparisons. Its macro nRMSE change was -16.331% (an increase in error), with mean changes Sanyo -16.567%, Hanwha -14.686%, and Qcells -17.924%. H144 improved in 0/3 technologies. Versus Joint Early-Fusion it was worse by 1.962% macro nRMSE and non-inferior in only 4/12 combinations.

Parameter counts: Joint Early-Fusion 1,496,832; Shared–Private 55,257,000; one Independent ModernTCN 682,896; three independent models 2,048,688. Shared–Private therefore also violates the required total-parameter condition. The inflation is principally caused by the three fusion projections. Each is `Linear(5184, 3456)`, where `(48 shared + 24 private) x 72 = 5184` inputs are projected to `48 x 72 = 3456` outputs. One projection has 17,919,360 parameters including bias; the three together have 53,758,080 of the model's 55,257,000 parameters.

All six status entries are completed. `metrics.csv` contains per-seed RMSE, MAE, R2, nRMSE and both regular/daylight scopes for all three models; all horizons are prefixes of one H144 prediction. Local artifacts contain the predictions, labels, timestamps, masks and retained validation checkpoints. Completed-run `last.pt` files were reclaimed during the disk-space recovery; this does not affect evaluation or checkpoint selection.

## Screening conditions

1. Independent macro nRMSE improvement >=2%: failed (-16.331%).
2. Improvement in >=8/12 combinations, no technology average worsening >1%, and H144 improvement >=2/3: failed (0/12, all three worsen, 0/3 H144).
3. Better than Early-Fusion macro nRMSE and non-inferior >=7/12: failed (-1.962%, 4/12).
4. Fewer parameters than three independent models: failed (55,257,000 > 2,048,688).

## Interpretation boundary

The evidence is sufficient to reject the **current** Joint Early-Fusion and Shared–Private implementations: both lose to Independent ModernTCN in all 12 regular-full-timeline technology-horizon combinations, and Shared–Private also fails the Early-Fusion and parameter criteria.

It is not sufficient to claim that the general hypothesis of cross-technology shared forecasting has been disproved. The candidate models and Independent ModernTCN were not strictly matched in backbone implementation, channel width, layer depth, parameterization, or effective training budget. Those design differences prevent a universal causal conclusion about sharing itself.

A matched rerun is nevertheless not recommended for this project. The two implemented joint models trail Independent ModernTCN in 12/12 combinations, while the purpose of this screen is to find the fastest viable route to a final paper. Additional matching or Shared–Private v2/v3 development would not be a proportionate next investment.

## Verification status

**Implementation smoke tests passed; end-to-end protocol provenance is only partially verified.** The tests cover output shape, head/shared/private gradient routing, finite masked loss, the absence of cross-technology input averaging at the declared interfaces, and confirmation that the training function has no Test-loader argument.

The following items were not fully programmatically verified in this screen: every saved run's split-boundary provenance; the complete Train-only fit-call history for every preprocessor; exact common-weather non-duplication at the realized batch level; per-technology inverse-scaler correctness across all saved artifacts; exact prediction/label/timestamp/mask identity and metric recomputation for every reused Independent run; and checkpoint-selection provenance across all runs. These gaps are recorded rather than expanded into new contracts, gates, or infrastructure for a terminated implementation route.

The single next recommendation is to retain independent per-technology forecasting as the research control and redirect the paper to a scientifically distinct, evidence-backed question.

Metadata: the project records the arrays as Alice Springs co-located PV data; authoritative component technology, capacity, Active_Power unit, and coordinates remain UNKNOWN in the available DKASC/project evidence and are not inferred from filenames.

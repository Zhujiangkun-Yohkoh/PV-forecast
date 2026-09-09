from pathlib import Path
import json
H=Path('GFNODE_experiments/scheme_A_support_closeout');root=Path('.')
for name in ['PROJECT_REPORT_CN.md','REVIEW_RESPONSE_MATRIX.md','MANUSCRIPT_CHANGELOG.md','JOURNAL_STRATEGY_CN.md']:
 p=root/name;old=p.read_text(encoding='utf8')
 if name=='PROJECT_REPORT_CN.md':new=(H/'REPORT_CN.md').read_text(encoding='utf8')
 elif name=='REVIEW_RESPONSE_MATRIX.md':new='''# 当前收尾响应矩阵（基线89e494f）

|问题|实际修改|证据|状态|剩余限制|
|---|---|---|---|---|
|R1 布尔回放失败|36组分布、W/kW、216指标影响和误差界限；5代表GPU批次|Alice_prediction_differences、Alice_metric_impact、batch_probe、ALICE_REPLAY_IMPACT_REPORT|完成诊断，历史数值状态部分未恢复|原16/20状态不变，不能宣称恢复历史处理器|
|R2 整场址阻塞|按方法接受；Hanwha/Qcells三个Inv种子正式评分，Sanyo单seed；S1/S2/新增按各前缀|Alice_accepted_metrics、Alice_diagnostic_metrics、各site块CSV|完成约定核心，Sanyo三seed均值合理不生成|训练支持仍不变；诊断模型不入正式排名|
|R3 支持/时期混杂|两站2017/2018×complete/pointwise四格；统一缓冲，原归档另表|four_cell_summary/intervals/MSE、boundary_audit、6原神经回放|完成|不是纯季节因果；2017推理分组明确记录|
|独立复算|不调用生产interval的块计数矩阵法；动态回放状态|independent_verify、POINT_AUDIT、INDEPENDENT_AUDIT|完成|检查行数不是独立实验数；不证明重新训练结果|
|R4文稿/图件|5主图，支持与四格前置，3张历史图移补充；弱化decisive|main、Supplement、closeout_figures|完成后依PDF QA定稿|作者声明与期刊格式待最终流程|
|PR描述|仅生成本地草稿，不写外部PR描述|PR_DESCRIPTION_DRAFT.md|按要求完成|不绕过此前自动审批拒绝|

结论B：新增核心收窄至经回放验证的方法，可进入投稿准备，不把全部次要模型逐点通过作为条件。
'''
 elif name=='MANUSCRIPT_CHANGELOG.md':new='''# 当前收尾修改（基线89e494f）

- 将数值容差与科学指标影响分开；不再将整个Alice场址停止评分。
- 新增Hanwha/Qcells三seed主模型S1/S2与新增起报结果；Sanyo不补假三seed均值。
- 外部2017/2018统一候选缓冲，形成complete/pointwise四格，保留原归档数字。
- 摘要重写为181词；主文只保留5幅决定性图，将3幅未改动历史图移入补充。两幅核心图重绘，其余数据不变。
- “The decisive diagnostic”改为证据支持的诊断表达。补充材料记录批次与处理器限制，主文不写环境排错过程。
- 原所有预测、参数、checkpoint、Ridge网格和旧失败记录不覆盖。没有训练、参数搜索或新增起报月份。
'''
 else:new='''# 当前投稿判断（收尾R1–R4）

选择状态B：收窄新增核心模型范围后，可进入投稿准备。已验证的Hanwha/Qcells支持敏感性与两外部系统四格比较形成完整证据链；Sanyo未恢复的两seed、TCN/Recurrent新支持只保留诊断身份。原支持指标的极小回放变化没有改变已比较排序，但不据此认定历史输入状态相同。

按作者高分区优先偏好，Renewable Energy仍为有条件冲刺首选；Solar Energy是主题最直接的下一顺位；JRSE为备选。该顺序是适配性判断，不是已核实CAS分区排序。CAS正式版本年份、大类与小类仍须机构权威查询；本次不重复核实或猜测。此前官方scope/费用查询日期保留在历史段，不能当作本次重新查询。

当前无需为维持论文主张重训：论文讨论冻结模型评价，不声称纠正训练选择偏移后的最优模型。若未来研究训练选择，可单独安排Qcells Inverted三seed固定条件的目标mask训练对照；不是本轮或投稿前无限加模型/月/alpha。作者仍需确认期刊、声明、公开范围与费用选项。未实际投稿。
'''
 p.write_text(new+'\n\n---\n## 89e494f及以前的历史记录（下文“本轮”不是当前收尾）\n\n'+old,encoding='utf8')
(root/'START_HERE_REVIEW.md').write_text('''# 当前审核入口：状态B，可进入收窄后的投稿准备

先读PROJECT_REPORT_CN.md及GFNODE_experiments/scheme_A_support_closeout/ALICE_REPLAY_IMPACT_REPORT.md，再读最新main.pdf与supplementary.pdf。核心新图在closeout_figures。原/扩展网格和各时期/支持分别标明。

旧容差仍16通过/20未通过，但已量化指标影响；新增接受比较按方法推进。Sanyo没有完整三seed Inverted新增支持均值。未训练、未选alpha、未改变月份。PR描述只在本地草稿。

完整包保存全部已知本地数据和重型证据；轻量包仅可独立复算块/CSV，不替代原始输入前向。各包清单/校验记录给出本次提交，旧包不递归进入。旧历史报告不覆盖当前状态。
''',encoding='utf8')
(root/'PR_DESCRIPTION_DRAFT.md').write_text('''# Local-only PR #19 description draft

Not posted. This file is not authorization to write the PR description externally.

The closeout quantifies all 36 Alice replay differences without changing the original tolerance: 16 pass and 20 remain outside it. On checked identical targets the maximum RMSE change is 0.000567 W, with no historical-support rank or Daily-effect sign changes. Five representative GPU batch probes document remaining numerical state limits rather than claiming model failure or recovered historical preprocessors.

New-support comparisons now proceed per verified method. Hanwha and Qcells include complete Inverted three-seed contrasts; Sanyo retains only its verified single seed and deterministic references, without a fabricated mean. S1/S2 restore scoring points while leaving training, inputs and Daily/Last-value rules frozen. Unaccepted methods remain separate diagnostics.

A two-site, two-period, two-support comparison uses unified candidate boundaries and buffers. The NIST Inverted/Daily point-estimate reversal and Yulara original/expanded Ridge change persist under either scoring rule. Original archived boundaries and scores remain separately traceable. Six original external checkpoint replays pass; supplemented candidates use explicit original/new batch partitions with newly computed forwards, not spliced old predictions.

Independent saved-point and block-count bootstrap implementations verify current masks, effects and intervals. The paper centers reference information, scoring population and period dependence, with numerical-state detail in the Supplement. Readiness is B: a narrower but complete scientific core can enter journal preparation. No neural training, refit, alpha search, new origin months, submission, release or merge. Keep PR in draft. Commit and archive checks are in the delivered Git manifest.
''',encoding='utf8')
# Preserve original entry and its hardcoded historical snapshot, use dynamic new entry.
p=root/'ENVIRONMENT_AND_REPRODUCTION.md';p.write_text('''# Current closeout entries

From project root: `python GFNODE_experiments/scheme_A_support_closeout/verify_light.py` independently recomputes block-weight intervals without importing production interval functions, and reports observed replay status without expecting a fixed failure count. `audit_points.py --paths LOCAL.json` needs the full saved prediction evidence. `reduce_periods.py --paths LOCAL.json` only reduces existing arrays. `period_forward.py` refuses an existing output file; use an explicitly fresh destination for new inference, never archive evidence. No processor fitting or model training is invoked.

Current figure entry: `python manuscript/clean_pv_benchmark/build_figures.py`; current folder closeout_figures. Only two affected plots are regenerated; twenty unchanged figures are retained with their source scripts. The full handoff includes earlier builders and their CSV inputs. Core scientific state is B, not a claim of complete historical neural-state recovery. The original fixed_period_review/verify_light.py remains a frozen historical 20-failure snapshot.

## Earlier entries (historical)

'''+p.read_text(encoding='utf8'),encoding='utf8')
print('Root reports and local-only PR draft updated')

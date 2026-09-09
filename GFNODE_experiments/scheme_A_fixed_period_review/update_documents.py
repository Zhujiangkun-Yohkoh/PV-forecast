# One-time authoring record; do not rerun over the manually finalized manuscript.
"""Generate version-explicit manuscript additions and reviewer reports from CSVs."""
from pathlib import Path
import json,re
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PAPER=ROOT/'manuscript/clean_pv_benchmark';R=HERE/'results'
def text(p):return p.read_text(encoding='utf8')
def write(p,s):p.write_text(s,encoding='utf8')
def md(df):
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'
A=pd.read_csv(R/'expanded_paired_intervals.csv');C=pd.read_csv(R/'period_table_summary.csv');I=pd.read_csv(R/'fixed_2018_primary_intervals.csv');D=pd.read_csv(R/'period_table_decomposition.csv');rep=pd.read_csv(R/'Alice_replay_summary.csv');neural=rep[~rep['check'].str.startswith(('Original','Expanded'))]
y=A[(A.site=='YULARA_COMBINED')&(A.block_hours==48)&A.reference.isin(['Daily','Inverted mean'])]
summary=C[(C.horizon==144)&(C.scope=='full')].pivot(index='site',columns='method',values='RMSE').reset_index()
support=pd.concat([pd.read_csv(R/(s+'_support_only.csv')) for s in ['Sanyo','Hanwha','Qcells']]);train=pd.concat([pd.read_csv(R/(s+'_training_support.csv')) for s in ['Sanyo','Hanwha','Qcells']])
report='''# Scheme A 固定范围补强报告（2026-09-09）

起点 ad59ed234de457f251ea16bfb51521f976c8fd83；当前分支 manuscript/clean-pv-benchmark-multisite-revision。本报告取代此前“本轮”状态；历史报告保留在各原目录。

## 当前判定
A 与 C 的分析已完成。B 完成原规则来源核查、真实训练支持、36 个 checkpoint 加载和前向、S1/S2 支持组成，但 **20/36 神经前向未满足历史 rtol=atol=2e-5 容差**；16/36 满足。12/12 Alice 原/扩展 Ridge 系数回放通过。新增 Alice 神经数组保留为未接受的诊断输出，没有据此发布正式 S1/S2 全模型分数。因此本次科研补强为部分完成，不能报告已具备最终投稿条件。

没有神经训练、alpha 搜索或重新拟合处理器。没有覆盖旧原始数据、预测、系数或 checkpoint。前向程序以运行时补丁禁止 KNN、IF、MinMaxScaler 的 fit、Tensor.backward 和 torch.save。

## A：扩展网格的时间不确定性
五系统 × 三功率范围 × 三块长 × 八配对 = 360 行区间；这些是相关描述性统计条目，不是360个独立实验。逐元素核对原/扩展 origin、target_start、标签、有效掩码、Daily 和 Last-value。目标时间为 origin+5…720min。共同完整 H144、Daily-finite 支持，48h主方案，24/72h敏感性，2000次、seed20260908。SSE与计数先合计再开平方；种子效应先分别计算，再平均，不集成预测。零参考RMSE时skill未定义。

Yulara 的关键结果：
'''+md(y[['scope','reference','rmse','reference_rmse','effect_kW','ci_low_kW','ci_high_kW']])+'''
full 中与 Daily、Inverted 的区间都跨零；不能称显著占优、等效或无效。低功率损失单独保留。Qcells 扩展 B 从0.236520升至0.259459kW；NIST B原/扩展预测相同，零差和退化区间未删除。各行提供非空块数、有效重采样次数及零支持次数。原网格区间未覆盖。

## B：负标签支持与尚未解除的前向复现问题
提供方资料确认AC功率含义，但没有找到“所有有限负值为无效”的定义。`_valid_power` 的 finite & >=0 是项目实现规则；历史保留不等于已证明合理，也不能把负数或近零值直接解释为正常净功率。

真实Train/Validation支持：
'''+md(train[train.horizon==144][['site','split','method_support','origins','points','active_fraction']])+'''
神经完整H144训练不要求origin功率有效；Ridge另要求有效origin，不能混称。各小时分布在独立CSV中。

Daily匹配后的支持变化（不是已经接受的新模型评分）：
'''+md(support[support.horizon==144][['site','support','origins','points','unique_targets','active_points']])+'''
候选起报只要求72个历史网格位置和Test内至少一个未来点。S1逐点排除负值；S2保留有限原值；超出Test的目标不评分。Last-value和Daily仍沿用原非负规则，未与标签规则混改。负标签不是唯一可能的支持限制；表中是Daily及origin参考交集之后的数量。未发布恢复起报后的正式神经排名，不能断言比较改变或不变。

### 实际定位经过
新增入口初版把“未来标签超出Test”同时作用于过去一天的Ridge输入，已在任何评分前修正：未来标签不可评分，不表示其-24h输入不可用。修正后原/扩展Ridge回放通过。

随后发现Alice部分神经回放误差。Sanyo TCN42相同原起报、原窗口构造与相同批次下最大差0.000247717kW，125/6589个起报至少一点评价容差未通过，差的中位数为0。重新构造的历史窗口与使用已保存Ridge处理器的窗口逐元素相同，但原Alice神经处理器没有独立历史文件，因此这不是与历史神经输入数组的直接对照。CPU/禁用TF32的最大差约0.006973kW，未消除差异；改变推理批量或矩阵TF32也未解决。不能把这些观察证明为唯一后端原因。全部36个记录见 Alice_replay_summary.csv；没有静默放宽容差或替换旧预测。

最小下一步是恢复原Alice推理环境/确切处理状态并通过36个原起报复现，再评分已经生成的新起报。当前不建议先重训，因为尚未确定复现差异来源。若之后接受S1并发现训练选择影响，需要另立Qcells Inverted三seed、原17通道/原日期/原预算、逐目标mask损失的最小定向训练设计；本次未执行，也不据未接受的评分主张其必要性。

## C：同场址次年固定时段
计划先于2018误差计算写入。两场址origin均固定2018-04-01 00:00至06-30 23:55，各26208个候选；最后目标为07-01 11:55。NIST官方年档案提取03-30至07-01共94日作缓冲，原ZIP CRC通过。Yulara用已有同一combined-system文件。2017处理器、6个Inverted checkpoint、两套Ridge系数和阈值全部冻结；6个外部神经原2017完整起报回放及8个Ridge回放通过。

2018是逐目标有限值与有效origin/Last-value/Daily的共同交集；原2017主表使用完整H144合法支持。时期和支持定义均标明，不能把二者差异全归因为季节。数据集此前用于覆盖审计，本次固定期误差按本计划生成，不包装成整个项目从未查看数据的确认性验证。

12h/full，kW：
'''+md(summary)+'''
主要48h配对区间：
'''+md(I[(I.horizon==144)&(I.block_hours==48)&(I.scope=='full')][['site','method','reference','effect_kW','ci_low_kW','ci_high_kW']])+'''
Yulara原B在新时期低于Daily，扩展B高于Daily；Inverted也高于Daily，2017中神经对Daily均值占优的模式未延续。不能用已知新分数重新选择alpha或把原网格改成新的预注册主模型。NIST中Inverted低于Daily，原/扩展B仍相同且低于Daily，B−Inverted区间跨零。模型、网格与时期的排序不稳定是本次新证据，不是实验失败。

NIST 12h的Inverted−Daily加权MSE贡献（先每seed分解后平均）：
'''+md(D[(D.site=='NIST_GROUND')&(D.horizon==144)&D.method.str.startswith('Inverted')].groupby('scope',as_index=False).agg(delta_MSE=('delta_MSE','mean'),points=('points','first'),full_points=('full_points','first')))+'''
active节省仍与low-power额外误差方向相反，但本次active节省更大，净差转为负；不能把2017的full反转当作不变机制。未据此推断夜间、积雪或维护。完整运行日志和Yulara详细字段元数据仍不完整，不能声称设备跨年完全未变。

## D：论文、图件与投稿
摘要和主文改为历史信息、目标支持和固定次年时期的证据链。主表清楚并列原/扩展网格；主Fig.2改为扩展区间；增加固定期效应区间图，保留所有负结果。S1明确4/69333=0.0058%，S3用不等宽分类点，S13注明Original grid。Alice实际恢复起报误差图尚不能完成，原因是复现未通过；不以支持分布图冒充误差验证。

投稿判断：可以进行导师科学审阅与格式准备，暂不建议提交期刊。当前价值是参考信息、目标组成和时间段改变收益判断的可复核证据。Solar Energy主题匹配最直接；Renewable Energy保留冲刺候选，但在Alice复现/支持问题解决前不推荐实际首投。CAS权威年份、大类、小类仍待机构核实；候选顺序不是已核实分区排名。无需再换模型、月份或扩大alpha。

## 验证与复现边界
本次：A保存数组对齐、360行块统计；36个Alice权重加载/前向（16通过历史数值容差，20未通过）；6个外部神经原2017回放；8个外部Ridge回放；2018新预测及独立逐点指标/SSE核对；CSV块重放；原795个文件前后保护；论文编译和逐页QA记录另见结果JSON与FIGURE_QA。
历史：M1-R 43、M2 17/10及10432比较未整套重跑，不能把历史通过数当作本次验证。未进行完整从原始数据重训复现。包内轻量核验仅证明CSV/块统计与清单一致，不证明原Alice神经前向已修复。

命令入口（在项目根目录，路径配置不提交）：
```
python GFNODE_experiments/scheme_A_fixed_period_review/expanded_intervals.py --paths LOCAL.json
python GFNODE_experiments/scheme_A_fixed_period_review/frozen_evaluation.py --paths LOCAL.json --sites Sanyo Hanwha Qcells
python GFNODE_experiments/scheme_A_fixed_period_review/frozen_evaluation.py --paths LOCAL.json --sites YULARA_COMBINED NIST_GROUND
python GFNODE_experiments/scheme_A_fixed_period_review/verify_and_summarize.py --paths LOCAL.json
python GFNODE_experiments/scheme_A_fixed_period_review/verify_light.py
```
完整包保留全部本地关键证据，包括未接受的Alice新增数组并明确标记；轻量包不含重型原数据/权重/全量NPZ，携带清单和必要案例；图件包含当前21组源文件。打包CRC、试解压和实际轻量执行结果以对应sidecar为准。作者声明、公开范围、费用选择与投稿均未代为确认。
'''
write(HERE/'REPORT_CN.md',report)
rootold=text(ROOT/'PROJECT_REPORT_CN.md')
if '# 历史项目报告（截至 ad59ed2）' not in rootold:write(ROOT/'PROJECT_REPORT_CN.md',report+'\n\n---\n# 历史项目报告（截至 ad59ed2）\n\n'+rootold)
matrix='''# 本次固定范围审核响应（2026-09-09）

|问题|原位置|措施与实际执行|证据|状态|剩余限制|
|---|---|---|---|---|---|
|A 扩展网格区间|diagnostic results|复用20组原/扩展预测，精确对齐，360行配对；空块和零差保留|fixed_period_review/results/expanded_*|完成|条件时间区间，不含重训|
|B1 负值依据|_valid_power|官方资料及代码复核，撤回“合理性已确认”暗示|DATA_SOURCE_REVIEW.md|部分完成|无全负值无效的提供方依据|
|B2 训练支持|Alice/Ridge窗口代码|三个阵列Train/Val实际起报及小时/功率组成|*_training_support.csv; *_training_hours.csv|完成|保留神经与Ridge不同origin规则|
|B3–B5 恢复起报评分|原36checkpoint|36加载/前向；S1/S2支持已算；20未过历史数值容差，不发布正式新评分|Alice_replay_summary.csv; *_support_only.csv|部分完成|需修复原前向复现；Fig.3实际误差补充未完成|
|C 固定次年|2017外部权重|先记录计划，下载94缓冲日，6个神经+8个Ridge原预测复现后评价2018|*_2017_REPLAY.json; *_2018_*|完成|运行日志不完整；非全年/新场址泛化|
|D 论文与图件|main/Supplement|扩展区间、原/扩展主表、新时期结果和对应反例|fixed_period_figures; manuscript changelog|部分完成|不把B未接受评分写成完成|
|投稿判断|JOURNAL_STRATEGY_CN|主题适配与CAS待核实分开|官方URL及查询日期|完成|先科学复现修复，再最终定刊|

以下旧矩阵仅作历史记录；其中“合理保留”不应解释为负值规则已获提供方证明。

'''
write(ROOT/'REVIEW_RESPONSE_MATRIX.md',matrix+text(ROOT/'REVIEW_RESPONSE_MATRIX.md'))
title_old='Reference Information and Target-Power Regimes in Multi-Window Photovoltaic Forecasting';title_new='Reference History and Target Support in Multi-Window Photovoltaic Forecasting'
s=text(PAPER/'main.tex').replace(title_old,title_new).replace('diagnostic_figures/','fixed_period_figures/')
abstract='''Forecasting gains depend on the reference history, the scored power range, and the evaluation period. We examine these dependencies in five photovoltaic systems at three geographic sites, using 60 frozen neural runs and two linear information sets: six-hour recent history, with or without an available previous-day trajectory. Forecasts are assessed over cumulative one- to twelve-hour windows. A post hoc expanded regularization grid brings Yulara daily-augmented Ridge to 15.84 kW twelve-hour error, close to Daily at 16.07 kW and Inverted-variate at 15.81 kW; paired temporal intervals cross zero for both contrasts. Fixed next-year evaluation at the two external sites changes these comparisons without refitting. At Yulara, original-grid Ridge, Daily, expanded-grid Ridge and Inverted-variate have full-target errors of 8.02, 9.56, 10.84 and 13.86 kW, respectively. At NIST, Inverted-variate improves on Daily in that period, reversing the earlier full-target ordering. Separately, complete-window conditioning in Qcells strongly changes target-power composition; the finite-negative exclusion rule lacks confirmed provider justification. These results show why information gains, target selection and time-period effects must be distinguished. They support conditional comparisons of frozen forecasters, rather than a universally preferred model or regularization grid.'''
s=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',s,flags=re.S)
s=s.replace('Figure~\\ref{fig:ridgeintervals} supplies the paired time intervals.','Figure~\\ref{fig:ridgeintervals} now shows the expanded-grid intervals; original-grid intervals are retained in the supplementary evidence.')
s=s.replace('Table~\\ref{tab:ridge} preserves the original-grid twelve-hour results.','Table~\\ref{tab:ridge} presents both saved grids on identical twelve-hour targets.')
s=s.replace('At Yulara, the original linear errors arise from','At Yulara, the original linear errors are consistent with')
# Replace the affected main figure caption from its current generated source.
for stem in ['fig6_ridge_intervals']:
 pattern=r'(\\includegraphics[^\n]*'+stem+r'\.pdf\}\s*)\\caption\{.*?\}(\s*\\label)'
 cap=text(PAPER/'fixed_period_figures'/f'{stem}_caption.txt').replace('%',r'\%')
 s=re.sub(pattern,lambda m:m.group(1)+'\\caption{'+cap+'}'+m.group(2),s,flags=re.S)
methods=r'''
\subsection{Frozen next-year evaluation and scoring sensitivity}
Before calculating the new-period errors, forecast origins were fixed to 1 April--30 June 2018 at Yulara and NIST. The final origin is 30 June at 23:55 and its last target is 1 July at 11:55. Source data from 30 March through 1 July provide historical and target buffers. NIST retains fixed EST and strict five-minute aggregation; Yulara retains provider-local availability coordinates. Original 2017 processors, three Inverted checkpoints and both saved Ridge grids are applied without fitting any component or changing thresholds. This is a same-site next-year fixed-period evaluation, not zero-shot transfer to new sites or year-round validation.

Each new-period comparison scores the intersection of finite target power, valid origin power and available Daily values. Missing labels are never filled; negative finite external power remains eligible. This pointwise support is stated separately from the original complete-H144 Test support. Differences between periods therefore are not attributed solely to season. The same 48-hour primary and 24/72-hour sensitivity block procedure applies, with origin blocks anchored at 1 April.

For Alice, a separate support analysis distinguishes pointwise nonnegative targets (S1) from finite raw targets (S2), retaining the frozen input and Daily rules. Provider documentation does not establish that every finite negative measurement is invalid. A completed-window filter and a pointwise scoring mask answer different questions. The restored-origin neural comparison is not promoted here because the new forward path has not reproduced every historical checkpoint within the original numerical tolerance; its support counts and diagnostic status are reported in the Supplement.
'''
s=s.replace('\\section{Results}',methods+'\n\\section{Results}')
nextsection=r'''
\subsection{Same-site next-year fixed-period results}
Figure~\ref{fig:nextyear} compares the frozen models with Daily in the fixed April--June 2018 origin period. At Yulara, twelve-hour/full RMSE is 8.02 kW for original-grid B, 9.56 kW for Daily, 10.84 kW for expanded-grid B, and 13.86 kW for mean Inverted. Original B minus Daily is $-1.54$ kW with a 48-hour interval $[-1.92,-1.13]$ kW; expanded B minus Daily is $+1.28$ kW with $[0.42,2.27]$ kW. Inverted minus Daily is $+4.30$ kW with $[2.93,5.65]$ kW. Thus neither the earlier neural advantage over Daily nor the expanded-grid improvement over original B persists in this period. The two saved grids remain diagnostic comparisons, without reselection from these errors.

At NIST, mean Inverted has 38.19 kW RMSE, both B grids have 40.78 kW, and Daily has 51.03 kW. Inverted minus Daily is $-12.84$ kW with $[-17.40,-8.00]$ kW. B minus mean Inverted is $+2.59$ kW with $[-0.62,5.66]$ kW; this crossing interval does not establish equivalence. The original NIST full-target ordering therefore also changes. Full, active and low-power results, point counts and all fixed block lengths are retained separately for each period.

The NIST squared-error decomposition retains an active saving and a low-power cost, but their balance changes. The next-year mean seed contributions to Inverted-minus-Daily full MSE are $-1275.75$ and $+130.09$ kW$^2$, respectively, giving a net $-1145.65$ kW$^2$. These disjoint contributions explain the new full-target advantage without assigning low-power measurements to a particular weather or operating event.

\begin{figure*}[!tp]\centering
\includegraphics[width=0.97\textwidth]{fixed_period_figures/fig8_fixed_period.pdf}
\caption{Same-site next-year evaluation of cumulative twelve-hour windows, with origins fixed to April--June 2018. Negative differences favor the named method over Daily. Points and 95\% paired 48-hour-block intervals use common finite target and reference support within each panel; Inverted averages seed-specific effects. Both saved Ridge grids are retained. Panel scales differ.}
\label{fig:nextyear}\end{figure*}
'''
s=s.replace('\\section{Interpretation and limits}',nextsection+'\n\\section{Interpretation and limits}')
s=s.replace('The three geographic sites broaden the evidence beyond a single facility, but the external October--December period is not an all-season validation.','The three geographic sites broaden the evidence beyond a single facility. The original October--December performance splits and the added April--June next-year origin period do not constitute all-season validation. Period differences include model age, input exposure, operating conditions and scoring support, not an isolated seasonal effect.')
s=s.replace('Each model is fitted separately. The missing NIST operation log','Each model was fitted separately; no new-period refitting is performed. The incomplete NIST operation logs')
conclusion=r'''The benchmark identifies three conditions governing the interpretation of PV forecasting gains: reference history, target support and power composition, and the period in which frozen models are assessed. Daily changes the conclusion drawn from Last-value at Alice, while matched squared-error decomposition explains the changing NIST balance between active savings and low-power costs. Added daily inputs and expanded regularization have distinct, period-dependent effects within Ridge. The fixed next-year results show that neither a neural model nor a selected Ridge grid has a uniformly favorable ordering. Qcells support counts further show how whole-window rejection can select a different target population. Reporting these conditions alongside paired temporal effects makes the evidence useful without treating an architecture name as a universal performance guarantee.'''
s=re.sub(r'\\section\{Conclusion\}.*?(?=\\section\*\{Supplementary Material\})',lambda m:'\\section{Conclusion}\n'+conclusion+'\n',s,flags=re.S)
write(PAPER/'main.tex',s)
# Both grids are explicit in the main table; historical source CSVs remain untouched.
e=pd.read_csv(ROOT/'GFNODE_experiments/scheme_A_diagnostic_revision/results/expanded_ridge_metrics.csv');rows=[]
for site in ['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']:
 q=e[(e.site==site)&(e.scope=='full')].set_index('model');rows.append(f"{ {'YULARA_COMBINED':'Yulara','NIST_GROUND':'NIST'}.get(site,site)} & {q.loc['Ridge','old_RMSE']:.3f} & {q.loc['Ridge','new_RMSE']:.3f} & {q.loc['Ridge+day','old_RMSE']:.3f} & {q.loc['Ridge+day','new_RMSE']:.3f}"+r'\\')
write(PAPER/'review_ridge_table.tex',r'\begin{table}[!tp]\caption{Saved original and expanded-grid Ridge twelve-hour/full RMSE (kW), on identical complete-H144 Daily-matched targets. Both grids are post hoc.}\label{tab:ridge}\centering\small\begin{tabular}{lrrrr}\toprule & \multicolumn{2}{c}{Ridge A} & \multicolumn{2}{c}{Ridge B}\\ System & Original & Expanded & Original & Expanded\\\midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')
supp=r'''\section{Fixed-period extension and current verification scope}
This addition uses saved expanded-grid coefficients without a new regularization search. Paired differences cover all five systems and full, power-active and low-power twelve-hour targets. Expanded B is compared with expanded A, Daily and each real Inverted seed; expanded A/B are also compared with their original-grid counterparts. The same draw is used across methods and seeds. Empty blocks remain in the sampling universe, and zero-support replicates and undefined skills are explicitly counted. NIST B is unchanged between grids; its zero difference is not discarded.

For Yulara twelve-hour/full, expanded B minus Daily is $-0.224$ kW with $[-1.814,1.741]$ kW, while B minus mean Inverted is $+0.037$ kW with $[-1.175,1.336]$ kW. These intervals cross zero without proving equivalence. The new-period analysis applies unchanged 2017 transformations and coefficients; two source buffers make every scheduled six-hour and previous-day input available when source observations exist. Each of the 26208 April--June origins is generated before applying target masks. The final target is 1 July 11:55. Available source/operation metadata do not establish that no equipment change occurred. Missingness and transformed input ranges are reported independently of model score.

\subsection{Alice restored support: unresolved numerical replay}
The frozen neural Train routine accepts complete nonnegative H144 labels without requiring a valid origin label; Ridge does require a valid origin. For Qcells this gives neural Train/Validation counts 12747/2999 and Ridge 12648/2977. They are different supports, not interchangeable counts. The provider materials reviewed do not define all finite negative power as invalid. Keeping the historical rule provides traceability, not proof of its physical justification.

All 36 Alice checkpoints were loaded and evaluated without training. Sixteen reproduced the historical predictions within the original $2\times10^{-5}$ relative and absolute tolerances; twenty did not. The original neural preprocessing objects were not separately saved. A saved later Train-only processor reconstructs the current historical window builder exactly, but that is not a direct comparison with saved historical neural inputs. Backend and batch diagnostics did not resolve every discrepancy. Restored-origin prediction arrays are retained as unaccepted diagnostics; no formal all-model S1/S2 score or ranking is asserted. The support counts below are valid independently of that pending numerical issue.

S1 scores individual finite nonnegative targets, and S2 scores finite raw power, while retaining the historical nonnegative Last-value and Daily definitions. At Qcells twelve hours, historical Daily-matched support has 2996 origins and 431150 pairs; S1 has 6786 origins and 958668 pairs, and S2 has 6786 origins and 959382 pairs. Active counts are 187950, 442887 and 442887, respectively. These are overlapping forecast pairs, not independent observations. Their unique target timestamps are 6425, 6781 and 6786. This enlarged scoring population cannot establish the performance of a differently trained model.

\subsection{Current versus historical checks}
Six external Inverted checkpoints reproduced the complete original 2017 prediction arrays before the 2018 period was evaluated. Original and expanded external Ridge predictions also reproduced. Fit methods and weight serialization were prohibited at runtime. New-period metric and additive-MSE reductions, block replays, protected-file inventories and PDF checks are recorded with their actual scope. Earlier M1-R/M2 suite totals remain historical records, not newly repeated suites. No neural model was trained in this extension.
\clearpage
'''
write(PAPER/'supplementary_fixed_period.tex',supp)
ss=text(PAPER/'supplementary.tex').replace(title_old,title_new).replace('\\input{supplementary_diagnostic_front.tex}','\\input{supplementary_fixed_period.tex}\n\\input{supplementary_diagnostic_front.tex}');write(PAPER/'supplementary.tex',ss)
ss=text(PAPER/'supplementary_diagnostic_figures.tex').replace('diagnostic_figures/','fixed_period_figures/').replace('Horizon-specific (HS, solid) versus common-H144-origin (CO, dashed)','Horizon-specific support (solid) versus common complete-twelve-hour origins (dashed)')
ss=ss.replace('Yulara original Ridge distributions and weather-missing exposure.','Yulara original Ridge distributions and weather-missing exposure: 4/69333 Train origins (0.0058\%).').replace('Right: finite one-hour power-pair distributions in explicit bins.','Right: categorical points for unequal-width power bins, normalized by pairs in the displayed bins.').replace('Deterministic Ridge', '\\textbf{Original grid.} Deterministic Ridge')
write(PAPER/'supplementary_diagnostic_figures.tex',ss)
changelog='''# Fixed-period review changes — 2026-09-09

- Source: ad59ed2; historical raw/weights/predictions/grids preserved.
- Main title/abstract/conclusion now distinguish reference history, target support and fixed next-year period. Yulara diagnostic wording avoids a unique causal mechanism.
- Main Ridge table labels both grids. Fig.2 uses expanded intervals; old intervals remain in their original CSVs. Added Fig.7 fixed-period effects; original Figs.1/4/5/6 retain their explicitly historical conditions.
- Supplement front matter records A/C, actual neural versus Ridge support, and the unresolved Alice replay (20/36). No restored-origin all-model scores are fabricated.
- S1 denominator/range explanation, S3 categorical unequal bins, S4 caption expansion and S13 original-grid label updated.
- Remaining: B actual error panels and formal S1/S2 comparison await accepted historical forward reproduction. This is a scientific verification limitation, not an author-signoff delay.
'''
write(ROOT/'MANUSCRIPT_CHANGELOG.md',changelog+'\n## Historical changelog\n'+text(ROOT/'MANUSCRIPT_CHANGELOG.md'))
write(ROOT/'START_HERE_REVIEW.md','''# 最新固定范围审核（2026-09-09）

先读 PROJECT_REPORT_CN.md 开头及 GFNODE_experiments/scheme_A_fixed_period_review/REPORT_CN.md。A与C完成，B的36神经回放中20未过历史数值容差，因此尚不是最终投稿就绪稿。所有原重型证据和未接受的新Alice输出在完整包中区分；不要用后者作为正式比较。

论文在 manuscript/clean_pv_benchmark/main.pdf 和 supplementary.pdf；当前图件为 fixed_period_figures，旧目录保留。轻量入口为 scheme_A_fixed_period_review/verify_light.py；它不宣称神经预测已全部复现。对应Git与包校验见包内GIT_DELIVERY、清单和sidecar。
''')
write(HERE/'DOCUMENT_COUNTS.json',json.dumps({'abstract_whitespace_words':len(abstract.split()),'count_method':'English whitespace tokens in abstract source; not PDF full text'},indent=2))
print('Documents updated; abstract',len(abstract.split()),'words; B explicitly partial')

"""Assemble human-readable diagnostic reports from the retained numerical tables."""
from pathlib import Path
import json
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; R=HERE/'results'
def table(df):
 cols=list(df.columns);rows=[' | '.join(map(str,cols)),' | '.join(['---']*len(cols))]
 for row in df.itertuples(index=False,name=None):rows.append(' | '.join(f'{v:.9g}' if isinstance(v,float) else str(v) for v in row))
 return '\n'.join(rows)
def write(name,text): (HERE/name).write_text(text,encoding='utf8')
expanded=pd.read_csv(R/'expanded_ridge_metrics.csv');interval=pd.read_csv(R/'ridge_paired_intervals.csv');support=pd.read_csv(R/'qcells_support.csv')
write('RIDGE_DIAGNOSTIC_REPORT.md','''# P01 Yulara Ridge 真实矩阵诊断（2026-09-08—09）

结论：原 Yulara Ridge 极端误差不是此次检查发现的时间索引、单位、逆变换或求解器错误；主要证据指向罕见缺失模式的特征外推，以及原正则化候选范围不足。无约束线性预测可以远超 Train 功率范围。本轮没有删除高误差窗口、裁剪功率或改写原预测。

## 实际检查与定义

`designs.py` 从保存的 Train-only 处理器和原始三变量重建七通道及72步历史。A为504列；B额外加入144个逐目标−24h功率和144个缺失指示，共792列。Yulara维持provider-local时间与+5min availability；NIST保持固定−05:00。每个未来目标为origin+j×5min；Daily/B取该目标准确提前24h，在12h轨迹中最晚为origin−12h。因此均在起报时可用。比较实际origins、labels、target masks，不仅比较点数。

所有原始KNN、IF、scaler和最终标准化参数仅来自Train并直接复用。额外敏感性仅重新求解Ridge系数。目标函数是标准化输入、中心化目标上的 ||Y−XB||²_F+alpha||B||²_F，截距不惩罚；不能把alpha与按样本数平均的目标函数混用。

## 分布与误差集中

原A在完整H144、Daily匹配支持的Test RMSE=119.584750772 kW，B=35.950323857 kW。分布诊断表采用保存候选轨迹的全部有限标签，支持更宽，所以其Test RMSE=119.429239不能与headline混用。A Validation MSE=47033.794852 kW²（RMSE216.872762），Train最大真实功率109.903154 kW。A Validation预测约−1090.59至723.47，Test约−1124.40至729.39。

Train/Val/Test预测分位数、负值与超Train范围比例见 `ridge_distributions.csv`，逐分割最差20个起报见 `ridge_worst_origins.csv`。A的Test最高误差1%/5%窗口贡献总SSE的20.9512%/90.6781%；B为17.0578%/80.1562%。是集中爆发，不是所有时刻均同样恶化。诊断极端日期不作为代表性案例，全部仍进入原指标。

## 特征机制与数值核对

起报位置气温/GHI缺失指示在69333个Train窗口中各出现4次（0.0058%）；Validation18.5641%，Test5.4975%。Train标准差约0.007595，缺失标记1变成约131.65个标准化单位。两列一致、贡献相关；原A两通道各自Validation预测分量RMS约107.638 kW，Test各58.426 kW。不能把这些相关分量平方后当成独立MSE贡献相加。完整近常数、稀疏度、偏移、尺度见 `ridge_features.csv`，加性预测分量见 `ridge_feature_contributions.csv`。

使用实际69333×504/792矩阵的Cholesky独立求解，匹配标准化、alpha、截距和目标函数。原A Test最大差8.98945e−9 kW，B约2.91e−11；相对正规方程残差约1e−15，目标函数一致。Gram矩阵微小负特征值约−1.5e−10属于数值舍入，原A正则化条件数约8.96e7。真实矩阵结论不依赖之前小型合成测试。证据见 `ridge_solver_comparison.csv`、`ridge_alignment.csv`。

## 统一扩展网格：补充敏感性，不替换原结果

原10次选择中7次在原0.1—1000端点。本轮诊断后先记录统一13点网格10^−4至10^8，再对五系统A/B全部按完整H144 Validation MSE选择。130个Validation分数全部保留；Test未用于选择、决定是否报告或裁剪。只有NIST A仍在新下端点；没有继续搜索。

'''+table(expanded[expanded.scope=='full'][['site','model','selected_alpha','old_RMSE','new_RMSE','validation_MSE_kW2']])+'''

Yulara A/B选择1e7/1e6，Validation MSE降至816.895458/268.330155。Qcells B的Validation略改善但Test从0.236520变成0.259459，保留这个不利变化。不能以扩展后的Yulara结果反推此前方案正确，也不能把新拟合写成预注册结果。

NIST A新alpha1e−4时谱解/Cholesky出现0.002390 kW局部差；独立增广QR秩504，full RMSE差−1.834905e−7 kW。该微小数值敏感性已单独记录，QR预测另存、未替换谱解。见 `NIST_GROUND_Ridge_QR_CHECK.json`、`nist_QR_metric_sensitivity.csv`。它不是原Yulara异常的解释。

影响范围：主文摘要、线性参考结果和讨论；补充诊断方法、原Ridge与扩展网格图；中文报告。原神经指标、原Ridge表以及P03原网格区间全部保留。受影响图号见 `FIGURE_NUMBER_MAP.csv`。进一步限制：未做特征机制的独立因果干预；这里是实际矩阵分量和分布漂移支持的诊断，不是气象/运行事件归因。
''')
write('QCELLS_SUPPORT_DIAGNOSTIC_REPORT.md','''# P02 Qcells 共同起报选择机制

代码与冻结Test的时间戳、标签、掩码逐元素一致，没有发现索引错误。主要机制是原Alice对有限负功率标签的排除，随完整12h窗口扩展而影响大量相互重叠的起报。不能把该子集称为天然更稳健或更有代表性。

每个split的规则5分钟网格是候选全集。重叠标记：边界/历史不足、未来非有限/缺失、未来负值、origin无效、Daily不可用。互斥顺序固定为边界→未来非有限→未来负值→origin无效→保留。Daily是另外的逐目标交集，不用于解释primary起报筛除。`qcells_origin_flags.csv`保留全部标记，`qcells_reasons.csv`同时给出重叠和互斥数量。

Test从H12的6463个起报到H144的2996个，移除3467：117边界，3350未来负标签；重叠负标签标记3415，其中65已先归边界。未来非有限排除为0。54个有限负原记录（52在18点、2在7点）可被许多长窗口反复包含。数值从−0.010933 kW到接近零；没有官方依据将其解释为坏码或具体运行事件。

H12有77556起报—lead对，active36504（47.0679%）。共同H144的前1h有35952对，active42（0.116822%）；仅涉及25个active origins与25个唯一active物理时间戳。原H12有6760个唯一目标时间戳，共同前缀3260。被移除子集前1h的active占87.6406%，因此原本高输出起报被强烈筛走。

'''+table(support[(support.h==12)&support.support.isin(['H12_support','common_H144'])][['split','support','origins','valid_target_pairs','power_active_pairs','power_active_fraction','unique_physical_targets','unique_active_physical_targets','active_origins']])+'''

Train同样从44.84% active变为共同前1h约3.08%；Validation约44.96%变为0。各split的1h和12h、保留与剔除分布，小时、月份、唯一目标数完整见support/hours/months/power_histogram CSV。原记录见negative_records。不是只在Test偶然出现的现象。

历史Alice神经训练要求完整H144未来标签，但不要求origin有效；其Train12747、Validation2999，对照评估/Ridge的12648、2977。两者差异来自既有origin规则；本轮未默默改成同一训练样本，也未重训。完整窗口要求本身已使训练的起报时刻分布受该负值规则影响。共同起报固定lead曲线只能描述这一特定选择子集，不能解释为无条件的误差随lead增长。

共同1h Last-value约0.003983、Inverted平均0.124942 kW是上一轮保存逐点归约结果，本轮诊断重新核对其支持机制；未重新加载60个checkpoint复现此预测。原主分析未被替换。下一轮若修改Alice负值语义，需先取得物理/提供方依据并明确新协议；可能涉及训练样本变化，本轮不启动。
''')
write('RIDGE_PAIRED_UNCERTAINTY.md','''# P03 原网格 Ridge 配对时间块区间

本轮实际从保存预测计算。五系统完整H144、Daily-finite共同逐目标交集，A/B/Daily/三个真实Inverted种子的origin、label、target mask相同；full/active/low分别报告。这里全部使用原网格Ridge，扩展alpha结果不混入这些区间。

按forecast origin在Test起点午夜锚定的连续24/48/72h块聚合；48h为主，2000次、seed20260908。固定EST/provider-local坐标保留。每次抽样所有方法、所有真实种子使用同一索引，汇总SSE与有效点数后开平方。统计量B−reference RMSE，负值表示B误差较小。神经三种子分别求效应，再用同一时间抽样求均值；不对预测先做集成。确定性Ridge没有人为三个种子。

'''+table(interval[(interval.scope=='full')&(interval.block_hours==48)&interval.reference.isin(['Ridge A','Daily','Inverted mean'])][['site','reference','effect_kW','ci_low_kW','ci_high_kW','points']])+'''

全表270行覆盖所有三个范围、三个块长及三个神经种子和均值。主方案Alice12块、外部46块，空块也保留；每项有效块和起报/唯一目标支持见CSV。区间以冻结预测和这些日期为条件，不包含重新训练不确定性；重叠窗口使相邻块仍可能相关，不能说消除了所有时间依赖。

NIST原B相对Daily full差−6.9044 kW，48h区间[−8.4301,−5.3060]；相对Inverted均值−9.6142，[−12.0834,−7.1461]。这是观察期条件区间，不是已证明跨时间稳定的性能优势。Qcells B−Daily跨零；Yulara原B−Daily点估计约+19.88但区间跨零，与集中极端误差吻合。不能只按点估计宣布所有比较方向确定。

`ridge_block_sse.csv`足以不读取重型NPZ重新计算所有效应与区间；`verify_light.py`独立重采样复算270行。它验证统计归约，不代替重新核对原始数据和设计矩阵。扩展网格结果若要成为进一步主要结论，需另计算对应冻结新预测的区间，不能借用本表。
''')
write('VALIDATION_REPORT.md','''# 本轮实际验证与命令范围

运行时间：2026-09-08—09；起始提交14942f4b18bc85d43b168cafc5e7bad81eb9d311。命令在仓库根执行；PY指复现指南中的Python环境，路径参数仅位于忽略本地配置。

|实际命令|结果与边界|
|---|---|
|PY protect_sources.py --paths .local/review_paths.json --phase before/after|764文件SHA256、size、mtime_ns完全相同；无神经训练/原文件改写|
|PY diagnose_ridge.py --paths .local/review_paths.json|Yulara真实矩阵、三个split分布、原解/Cholesky、预测对齐；6求解记录|
|PY diagnose_qcells.py --paths .local/review_paths.json|三个split候选筛选，冻结Test timestamps/labels/masks逐元素一致|
|PY ridge_uncertainty.py --paths .local/review_paths.json|原A/B/神经预测配对，270区间行与块SSE；不是270独立实验|
|PY expanded_ridge.py --paths .local/review_paths.json --destination LOCAL_OUTPUT [--sites ...]|五系统×A/B，统一13alpha；130 Validation候选，10个选定拟合，30范围指标；无神经拟合|
|PY check_nist_qr.py --help（实际参数见该脚本入口）|真实增广矩阵QR复核；数值敏感性与RMSE影响单独保存|
|PY verify_light.py|独立CSV/块SSE算术复算，失败0、skip0；不是checkpoint前向复现|

原M1-R 43、M2普通17、artifact10、10432独立比较及上轮60checkpoint前向/训练日志记录是历史证据，本轮未完整重跑，不计作本轮通过。新增真实矩阵诊断、哈希和块重采样是本轮执行。

运行中真实问题：NIST原逐Timestamp列表设计导致内存/分页压力，C盘空间不足；仅清理本轮可重建临时目录并移至E盘，将等价时间连接改为纳秒向量化，保留固定时区。诊断源文件曾因空间不足写入中断，已恢复后完成实际矩阵运行。低alpha NIST求解差异没有隐去，改用增广QR量化。部分站点恢复运行曾将Yulara诊断小CSV写为空，已从本轮备份恢复，并修复只在有该站点数据时写出；旧证据未改动。TeX最初参数未引用及工作目录错误已修正，最终PDF/包验证另见QA记录。

不将语法、文件存在或历史记录冒充新模型预测复现。原始训练完整重跑未进行。包内轻量验证仅复算聚合证据，完整包提供实际矩阵/逐点复算所需数据但不宣称全训练已试运行。
''')
changelog='''# 本轮论文修改记录 P01–P06

主文保留题目 Reference Information and Target-Power Regimes in Multi-Window Photovoltaic Forecasting。贡献围绕历史信息、目标功率范围与支持筛选，不是新网络。

1. 摘要与结果前置线性对照、Yulara实际矩阵诊断及统一alpha敏感性；原网格结果不覆盖。
2. Qcells新增排除机制、物理时间戳与重复目标对区分，共同起报只作为有条件子集分析。
3. 新Ridge时间区间进入主图，神经seed SD不代替时间区间；NIST反转加条件范围。
4. NIST功率分解保留逐seed平均MSE的−420.64/+655.38/净+234.74 kW²，不平方平均RMSE。
5. Supplement前置诊断方法与14幅解释图，完整旧表后置。主文6图是为保留新关键证据，未为5图上限缩字号。
6. 全图使用诊断目录新版本，原review_figures及原统计证据保留。编号映射见FIGURE_NUMBER_MAP.csv。
7. 作者顺序、单位、声明状态不代填；投稿策略明确CAS分区尚待机构权威记录，RE为主题与高目标候选而非已核实分区第一。

未解决：Alice负功率标签原规则是否具物理依据；外部仅10—12月性能期；Yulara元数据和NIST日志不完整；扩展网格新拟合尚无另行时间区间（原网格已有完整区间）；期刊最终模板/作者签核待选择与确认。它们不能被“全部文件生成”替代。
'''
(ROOT/'MANUSCRIPT_CHANGELOG.md').write_text(changelog,encoding='utf8')
matrix='''# 审核响应矩阵（P01–P06，本轮）

|问题|原位置/影响|实际修改|本轮证据|状态|剩余限制|
|---|---|---|---|---|---|
|P01|旧Ridge与摘要，异常值未解释|真实矩阵诊断、统一扩展网格、旧新并列|ridge_*、expanded_*、NIST QR|完成诊断|特征贡献不是因果干预；新网格区间未算|
|P02|common-origin与fixed-lead解释|三split排除机制、重叠/互斥、唯一时间戳|qcells_*及ALIGNMENT|完成诊断，原规则合理保留于冻结结果|负值语义需提供方依据，变更或涉及下一轮拟合|
|P03|新Ridge无时间区间|五系统原网格48h及24/72h配对|270行+块SSE独立复算|完成|观察期条件性，不包括重训|
|P04|旧排名过强|摘要/结果/补充重排，解释前置|main/supp及CHANGELOG|完成|不宣称新架构或零样本|
|P05|全部图件小字、编码与支持|20图重绘，PDF/SVG/320dpi/source/alt|FIGURE_QA与最终PDF检查|以FIGURE_QA实际检查为准|审美可继续精修，不能替代科学限制|
|P06|选刊与分区混淆|官方范围/费用，CAS待核实，最小未来期设计|JOURNAL_STRATEGY_CN|部分完成|CAS需机构访问；未补充新季节|

历史R01–R12响应保存在上轮Git提交和完整历史报告；本轮不把其历史测试重新标为实际执行。
'''
old=ROOT/'REVIEW_RESPONSE_MATRIX.md'
if old.exists() and 'P01–P06，本轮' not in old.read_text(encoding='utf8'):
 matrix+='\n## 上轮响应原文（历史记录）\n\n'+old.read_text(encoding='utf8')
old.write_text(matrix,encoding='utf8')
report='''# Scheme A 本轮完整诊断与修订报告

日期：2026-09-09。原神经实验共60次，Alice三共址阵列17通道36次；Yulara/NIST分别7通道24次。本轮没有神经训练、checkpoint更新或原结果改写，执行了Ridge实际矩阵检查和13点统一正则化敏感性。

## 导师应先审核的三个发现

Yulara原A的119.584751 kW不是已证实的求解错误。训练中4次出现的气象缺失指示，在Validation达到18.56%，经小尺度标准化形成强外推。原Test最差5%窗口占90.68% SSE。统一扩大alpha后，A/B为29.838229/15.842110，旧结果仍保留。新增线性信息对照必须同时报告正则化敏感性，不能据旧异常宣布神经普遍优越。

Qcells完整12h条件将1h active目标对比例47.0679%压至0.116822%，机制为117个边界和3350个负未来标签排除，非未来缺测。42个active对只有25个唯一物理时间戳。该子集不代表完整测试分布；原负值规则的物理合理性仍待确认。

原网格Ridge B的配对时间区间补齐。NIST相对Daily full差−6.9044 kW，48h区间[−8.4301,−5.3060]；相对Inverted−9.6142，[−12.0834,−7.1461]。Qcells及Yulara部分比较跨零。不是所有场址上的一致简单模型优势，更不是重新训练不确定性。

## 实验与证据路线

原冻结神经训练→保存逐origin/lead预测、标签与mask→本轮真实矩阵/索引诊断→按Validation统一扩展Ridge网格→原网格Ridge配对块重采样→论文与图件重排。原始60次训练流程、数据字段、split与模型定义见下方历史项目报告及冻结实现，不重复训练来解决解释问题。

本轮详细方法、全部数字与限制分别见新目录RIDGE_DIAGNOSTIC_REPORT.md、QCELLS_SUPPORT_DIAGNOSTIC_REPORT.md、RIDGE_PAIRED_UNCERTAINTY.md；执行范围见VALIDATION_REPORT.md。764源文件前后SHA/size/mtime完全相同。轻量复算270行区间、6组支持归因、10个Validation选择等通过；这些不是独立实验数量。旧43/17/10和10432审计均标为历史。

## 论文、选刊与下一步

论文结论调整为“参考信息、功率子集和支持条件改变收益判断”。原Alice Daily22/24、外部预指定Inverted对LV16/16对Daily15/16保持冻结；不将其合成40独立试验。Yulara线性模型不再仅作为失败对照。Supplement前置新增诊断与解释，旧表保留。

候选首投Renewable Energy，下一顺位Solar Energy，稳妥备选JRSE：属于主题与证据适配判断，不是已经核实的CAS排序。中科院大类/小类及版本仍需机构正式记录。高分区主要科学风险是跨季节/年份覆盖及Alice负值筛选的外推限制；格式、篇幅、图件模板可在定刊后完成。建议最小未来实验为外部两站2018年4—6月固定模型前向评价，Yulara本地已有覆盖，NIST需下载91天；本轮未执行，也不为此重训网络。

## 交付阅读方式

完整包含可用raw、60神经checkpoint/预测、原10Ridge和新10Ridge及独立QR证据。轻量包含聚合/块SSE和代码，可复核区间与数值表，不能重建原设计矩阵或复现神经前向。图件包独立提供最新图、数据、脚本和QA。所有ZIP校验/试解压结论单列，不等同于全训练复现。

'''
p=ROOT/'PROJECT_REPORT_CN.md';old=p.read_text(encoding='utf8')
if '本轮完整诊断与修订报告' not in old:report+='\n# 以下为上一轮完整项目流程（历史记录）\n\n'+old
else:report=old
p.write_text(report,encoding='utf8')
(ROOT/'START_HERE_REVIEW.md').write_text('''# 导师审核入口：2026-09-09诊断修订

先读PROJECT_REPORT_CN.md，然后新诊断目录中的P01/P02/P03报告与MANUSCRIPT_CHANGELOG.md，再读最新主文PDF和Supplement。REVIEW_RESPONSE_MATRIX.md区分已解决问题与剩余科学限制。

本轮不重训神经网络，旧结果完整保留。重点是Yulara缺失模式外推、Qcells负标签与完整窗口选择、原Ridge的条件时间区间。新增统一alpha为事后敏感性，不替换原网格。选择期刊前需机构核实CAS版本和分区。

完整包支持重型证据审核；轻量包仅支持聚合/块SSE复算；图件包支持图数据与版式交接。各包清单、校验与轻量验证见PACKAGE_VERIFICATION.md。不得将其解释为重新跑通60次训练。
''',encoding='utf8')
print('Reports generated')

# Scheme A Multisite M1-R 审核报告

## 判定与授权

**SCHEME_A_MULTISITE_DATA_READY_FOR_FROZEN_TRAINING**

仅授权下一轮执行冻结的24次GPU训练。本轮训练执行：否；未执行风险模型拟合、Validation/Test预测、checkpoint选择或论文结果改写。

唯一来源 origin/research/scheme-a-multisite-data-confirmation，commit 626b664b9083df2685e29868cf1409d4b547059e；独立分支 research/scheme-a-multisite-data-confirmation-r1。Draft PR base 为 research/scheme-a-multisite-data-confirmation。未修改PR #14/#16、master、C1、NWP、原Scheme A结果或原始文件；不合并、不rebase、不force push。仅更新六个审核文件，本地路径配置不提交。

## 文档缺口重新分级

- **NIST_2017_OPERATION_LOG_UNAVAILABLE_NONBLOCKING**：2026-09-07复核官方论文、字典、目录并检索日志；门户仍无可解析正文，未取得2017 Ground运行清单。未找到不等于没有维护、停机或异常。全部有限电表功率、气温与GHI保留，不根据人工观察或功率-GHI关系清洗。登记为非阻塞数据限制，须在未来论文限制中披露；本轮不改论文。
- **YULARA_PROVIDER_METADATA_PARTIAL_NONBLOCKING**：官方术语页将 Weather Temperature Celsius 与 Global Horizontal Radiation 列于Yulara的 Environment.DG_Weather_Station。温度按Celsius字段使用；GHI采用原生数值，不换算，仅Train拟合输入缩放，不作跨场址绝对辐照量推断。GHI的完整单位、采样/聚合和无效码定义仍未确认，不声称传感器型号、不确定度或完整质量标志。

仅功率标签含义/单位不明、非唯一时间顺序、错误物理字段、大量不明无效码改变样本定义或不安全聚合继续阻塞。本轮未发现这些情形，两项文档缺口按用户回退规则不再阻止训练。

## 时间语义修正

NIST解析、聚合、规则网格、split、窗口及summary均保留固定UTC−05:00；time_basis=FIXED_EST_LST，utc_offset=-05:00。删除读取后的naive转换；聚合入口拒绝naive或其他时区。配置保存显式−05:00的三组split边界。无DST规则。Daily先对唯一时间轴精确−24小时连接，再索引窗口，减少重叠时间戳重复查找，不改变配对。

NIST仍使用[T−5,T)、五个不同且有限的一分钟观测、availability=T。Yulara仍排除两条离网格记录、不取整，availability仍为原时间+5分钟。字段、标签和可用时间没有变化。

## 有限值与运行状态

负功率、负GHI、高GHI非正功率及超经验范围有限值保留。只有空值、非有限值、明确结构错误或官方定义的无效码作为缺失。公共字段没有−999/−7999候选码，InvPAC不进入输入或目标。未来输入通过Train-only处理和IF表达异常，不删除标签。负值不自动等于夜间；没有日志不虚构维护/积雪/停机数量。描述性GHI>500使用各场址原生坐标，只计数，不筛选或作跨场址绝对量推断。

## 数据结论

NIST：365 个准确日文件、312,588,499 字节、100 列且完全一致。525,595 个唯一分钟、无重复或逆序。首末为 2017-01-01 00:00 和 2017-12-31 23:59，所有字符串固定 -05:00。缺失五分钟为：2017-10-20 23:59；2017-10-21 00:00、00:01、00:02、00:03，均为 EST。完整文件名/size/mtime_ns/行数在 DATA_AUDIT_SUMMARY.csv 的 file 行，完整表头在 audit/header 行，未生成原始数据 SHA。

正式候选：PwrMtrP_kW_Avg、AmbTemp_C_Avg、Pyra1_Wm2_Avg。Pyra1_Wm2_Avg 由提供的 CSV 直接包含；没有 Pyra1_mV_Avg，不执行重复单位转换。逆变器列 4,712 条 -999 及其他异常不进入目标。三个公共字段中未发现 -999/-7999 候选码；没有用数值大小猜测并删掉有限负值。空值、非数值、正/负无穷分别统计，分位数含 min、1%、5%、median、95%、99%、max。

Yulara：准确原文件名及 size/mtime_ns/14 列表头在摘要。文件名“2016”表示安装年份线索，不是完整数据覆盖年；实际 2016-04-01 23:45 至 2026-02-19 23:55，共 1,036,791 行。2017 年 105,122 行且唯一，无重复或逆序，105,120 个规则五分钟点全部存在，另有两条离网格记录。规则点存在不等于字段非缺失。

- 2017-02-22 15:30:02：功率 75.150801518689 kW，气温和 GHI 缺失。
- 2017-08-07 05:20:01：功率 -0.047513291616942 kW，气温和 GHI 缺失。

两条的 floor/ceil 规则时间均存在，但 floor 功率为空，无法证明把离网格值搬到该时间就正确。因此不 round，不合并，不修复标签；在派生规则网格排除原离网格行。摘要保存原值、相邻行和理由。106.6 kW 是 Sails in the Desert 联合系统 3，官网列出其 3-A/B/C 构成，一个外部 facility，不重复计场址。

## Train-only 聚合诊断及冻结规则

NIST 用 Train（1–8 月）电表累计接收能量减累计输出给逆变器的能量之差，比较五分钟增量与功率积分；仅对能量诊断屏蔽非有限增量和显然重置候选，未改原数据。不同一分钟功率对齐候选的平均绝对能量差：

| 功率移位（分钟） | 配对数 | 平均绝对差 kWh |
|---:|---:|---:|
| -1 | 348107 | 0.225899 |
| 0 | 348134 | 0.163658 |
| +1 | 348134 | 0.226019 |

Train 功率与 GHI 的 -5/0/+5 分钟相关系数为 0.929261 / 0.972166 / 0.928662。它们是原始测量语义诊断，不是预测性能，且从未使用 Validation/Test 预测。相同时间更一致，但不把其当成当前门户全部区间定义的证明。

冻结 NIST [T-5,T) 五个不同分钟、右标签 T、固定 EST、午夜 anchor、origin=2017-01-01 00:00、每变量五个有限值才求均值。availability=T；原分钟含义不唯一时，取其最晚可能测量结束时间。Yulara 规则记录按原时间+5分钟作为保守 availability，不伪造 UTC。各 split 的第一个输入测量区间必须完全位于 split 内，所以首个 origin 为 06:00。这些选择没有被任何 held-out 预测成绩驱动。

## 原协议差异必须显式披露

复用 KNN -> IF -> 原始缺失mask/IF追加 -> 七列feature scaler -> target scaler 的 Train-only 顺序，但本轮不拟合任何组件。原 _valid_power 排除负数，本轮明确保留非无效码的有限负数。原 validate 固定17维，不能直接用于7维数据。外部完整H144 fitting/Validation support 也要求 origin真实功率有效，以与 Last-value 共享口径；原 _build_full_h144 未作这个要求。以上是已冻结的外部适配，不修改原 Scheme A，也不宣称完全相同输入任务。

## 样本数（纯数据，不含模型预测）

下表 full/daylight 来自主 horizon-specific 支持。完整 first/last origin、月覆盖、输入窗口缺失率、split 标签缺失率、Daily matched 数目在 DATA_AUDIT_SUMMARY.csv，共 96 个 site/split/horizon/analysis/scope 项。

| 场址 | Split | H | Full origins | Full points | Daylight origins | Daylight points |
|---|---|---:|---:|---:|---:|---:|
| YULARA_COMBINED | train | 12 | 69861 | 838332 | 35109 | 388983 |
| YULARA_COMBINED | train | 48 | 69717 | 3346416 | 43727 | 1551419 |
| YULARA_COMBINED | train | 96 | 69525 | 6674400 | 55170 | 3091616 |
| YULARA_COMBINED | train | 144 | 69333 | 9983952 | 66248 | 4623382 |
| YULARA_COMBINED | validation | 12 | 8459 | 101508 | 4329 | 47952 |
| YULARA_COMBINED | validation | 48 | 8315 | 399120 | 5305 | 189051 |
| YULARA_COMBINED | validation | 96 | 8135 | 780960 | 6556 | 371232 |
| YULARA_COMBINED | validation | 144 | 8037 | 1157328 | 7802 | 547225 |
| YULARA_COMBINED | test | 12 | 26412 | 316944 | 14784 | 164664 |
| YULARA_COMBINED | test | 48 | 26376 | 1266048 | 18063 | 658095 |
| YULARA_COMBINED | test | 96 | 26328 | 2527488 | 22393 | 1313250 |
| YULARA_COMBINED | test | 144 | 26280 | 3784320 | 26192 | 1963842 |
| NIST_GROUND | train | 12 | 68837 | 826044 | 34613 | 380214 |
| NIST_GROUND | train | 48 | 66958 | 3213984 | 41646 | 1457504 |
| NIST_GROUND | train | 96 | 64717 | 6212832 | 50964 | 2790328 |
| NIST_GROUND | train | 144 | 62643 | 9020592 | 59066 | 4062345 |
| NIST_GROUND | validation | 12 | 8290 | 99480 | 4132 | 45799 |
| NIST_GROUND | validation | 48 | 7543 | 362064 | 4548 | 159943 |
| NIST_GROUND | validation | 96 | 6690 | 642240 | 5099 | 274705 |
| NIST_GROUND | validation | 144 | 5915 | 851760 | 5625 | 364699 |
| NIST_GROUND | test | 12 | 25219 | 302628 | 10161 | 109258 |
| NIST_GROUND | test | 48 | 24240 | 1163520 | 12577 | 413664 |
| NIST_GROUND | test | 96 | 23037 | 2211552 | 15700 | 775182 |
| NIST_GROUND | test | 144 | 21885 | 3151440 | 18602 | 1106213 |

## 四模型 synthetic forward-only 验证

实际环境：Python 3.12，PyTorch 2.7.1+cu118（本轮强制 CPU），NumPy 2.0.0、pandas 2.2.2、scikit-learn 1.5.0。用户环境中已有 PyTorch；未下载新 torch 训练栈。临时 venv 仅用于本地检查且被忽略，不提交环境或用户路径。

| 模型 | 原17通道参数 | 7通道参数 | 差值 | 检查参数张量数 |
|---|---:|---:|---:|---:|
| Discrete recurrent | 99362 | 96562 | -2800 | 42 |
| Inverted-variate | 194960 | 102800 | -92160 | 16 |
| Joint-patch | 148112 | 140432 | -7680 | 16 |
| Depthwise TCN | 683024 | 682384 | -640 | 20 |

四模型均在 B=1 和 B=2 得到 [B,144] 有限输出。逐参数张量前向扰动均影响输出，测试后恢复参数且梯度为空。以新随机17维模型的内存 state 验证 strict 维度拒绝；没有加载任何旧 checkpoint。参数变化不构成新架构声明。

## 普通测试及执行边界

M1原27项保留；本轮完整结果见文末。包括真实365文件/精确缺口、显式bin成员、缺一个分钟或一个变量值时不接受部分均值、Wm2不重复转换、目标字段和sentinel检查、离网格不round、时间缺口不拼接、真实mask及7列顺序、标签不插补、Train-only fit拒绝测试、availability、目标方向和split边界、按timestamp的Daily join、全方法逐元素mask、Train daylight阈值、四模型forward及参数参与、17维随机state拒绝、Test score sentinel及held-out数据变更不影响冻结配置/Train诊断、原始size/mtime保持一致。

预处理fit测试用记录数组的test double，并未对实际KNN、scaler、IF拟合；M1不宣称完成下一轮训练runner的端到端验证。Forward运行时将实际fit、训练/真实预测helper、backward、AdamW及torch save/load设为拒绝调用。测试没有以源码字符串检查替代数组行为。数据审核代码不计算实际模型或基线Test误差；Test支持计数是本轮明确要求的只读检查。

## 主次分析与24-run矩阵

主模型已锁定 INVERTED_VARIATE_TRAJECTORY：原冻结CSV 11,328行复核的primary mean RMSE排序为12/9/2/1，原报告平均排名1.875最佳；在外部Test预测前选定。主要比较为三个seed均值和sample SD，以及分别对Last-value、Daily的matched RMSE skill。其他三模型、排名、每seed、MAE/nRMSE/bias/R²和scope差异为次要；best-of-four包络仅描述性。完整24-run矩阵在 MULTISITE_PROTOCOL.md 和 multisite_config.json 中，候选预算/输入/划分已冻结，authorization_next_round=true（仅下一轮）。

## 官方来源、已确认与未确认边界

- [NIST Campus数据目录](https://catalog.data.gov/dataset/nist-campus-photovoltaic-pv-arrays-and-weather-station-data-sets)：2015–2018，一分钟/一秒数据和DOI。公开测试场旧说明仅列2015–2016，采用目录及本地2017文件核实年份，不把页面年代差异误判为文件不存在。
- [NIST Data Dictionary v1.0](https://www.nist.gov/document/datadictionarysupplementalcontentpdf)：array部分定义TIMESTAMP为LST/Max，电表AC有功、Ground Pyra1 GHI及平均单位。当前CSV的转换后Wm2列存在性由实文件确认。
- [NIST数据说明论文](https://nvlpubs.nist.gov/nistpubs/jres/122/jres.122.040.pdf)：说明数据错误/中断与清洗日志存在，不能据此推断2017具体事件。
- [NIST门户/DOI](https://doi.org/10.18434/M3S67G)、[测试场说明](https://www.nist.gov/el/beed/heat-transfer-alternative-energy-systems/photovoltaic-testbeds)：已访问，当前门户内容未能读出日志。
- [DKASC Yulara下载页](https://dkasolarcentre.com.au/download?location=yulara)、[联合系统metadata](https://dkasolarcentre.com.au/source/yulara/yulara-3-roof-sails-in-the-desert-2)：106.6 kW、mono-Si、roof、2016安装，子阵列合计系统，不是全部1.8MW Yulara。
- [DKASC Glossary](https://dkasolarcentre.com.au/glossary)：AC功率五分钟平均；Yulara resource部分需提供方进一步说明。没有从Alice Springs气象传感器段落直接套用其参数给Yulara。
- [Notes on the Data](https://dkasolarcentre.com.au/download/notes-on-the-data)及[2017相关页](https://dkasolarcentre.com.au/download/notes-on-the-data/p8)、[相邻记录页](https://dkasolarcentre.com.au/download/notes-on-the-data/p7)：2017-02-14网站迁移影响两处数据访问；2017-05-09 pyranometer角度调整明确属于Alice Springs，不能套给Yulara。日志并非穷尽所有短时事件，不据此删数据。


## 2026-09-07官方复核与数据使用条款

上列NIST论文、字典、Data.gov目录，以及DKASC下载页、术语页和联合系统metadata本轮均已复核。NIST门户未能提供可核验日志，Yulara资源字段详情仍未取得，未虚构资料或联系提供方。

[DKASC数据使用条款](https://dkasolarcentre.com.au/download/terms-conditions)要求注明分析日期、单位、安装年份及影响比较因素，并准确引用来源及相关声明。5000单元格门槛明确针对Alice Springs，250000门槛针对NT Solar Resource，不冒充Yulara明确授权范围。未来公开前核对适用再分发条款，本轮不发布原始数据；训练就绪不代表无限制再分发许可。未确认真实下载日期，访问日期不冒充下载日期。

## 本轮执行验证

43 passed，0 failed，0 errors，0 skipped（原27项加16项M1-R测试，含对既有行为的回归复用）。最终完整运行48.953秒。首次完整运行42项通过、1项因Git所有权限制报错；以仅该只读命令的safe.directory修正后完整重跑通过，无全局Git配置改动。更早的重复时间连接慢路径审核运行已中止，未计入通过测试。

逐项对比固定M1 commit中的96条支持记录：origins、valid-target points、输入/标签缺失率、月份与daylight阈值全部相同；NIST first/last origin仅增加显式−05:00。五个缺失分钟及两个离网格记录一致。24-run矩阵、三个seed、主模型、训练预算、模型配置、split和七通道顺序与M1逐项一致。

全部366个原始文件size和mtime_ns前后相同。无checkpoint、预测数组、训练日志或结果缓存生成；摘要仅为小型审核CSV。本轮复用已有Python环境，CPU synthetic forward，无实际预处理拟合。训练runner的端到端实现属于下一轮范围。

复现：`python -B GFNODE_experiments/scheme_A_multisite_extension/test_multisite_protocol.py --paths .local/multisite_paths.json --write-summary`。仅全部测试通过且无skip才更新审核CSV。

原始2017观测描述性数量：

| 场址 | 负功率 | 负GHI | 原生GHI>500且功率≤0 |
|---|---:|---:|---:|

| NIST_GROUND | 14418 | 271221 | 751 |
| YULARA_COMBINED | 52807 | 50504 | 0 |

以上为原始观测计数：NIST一分钟、Yulara五分钟并包含两条离网格原记录；正式派生数据仍排除离网格记录，不据此筛选任何有限标签。
Train能量诊断配对数、中位数及诊断边界计数不变；重新运行的均值/相关系数最大差异约2.22e-16，属于浮点数值末位差异，不改变时间规则或样本定义。

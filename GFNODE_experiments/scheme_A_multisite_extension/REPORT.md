# Scheme A-M1 数据确认、协议冻结和训练授权审查

## 最终判定

**SCHEME_A_MULTISITE_DATA_REQUIRES_CORRECTION**

这不是数据规模不足，也不是模型不支持七通道。候选协议和精确 24-run 矩阵已冻结，27 项普通测试通过，但以下提供方证据尚未闭合，故下一轮训练授权仍为 false：

1. **NIST Ground 2017 运行/质量日志尚未取得。** 本地 365 CSV 之外没有随附 metadata、readme 或 quality log。官方数据说明明确提到维护数据丢失/错误及清洗日志，但当前门户在 web 工具中不返回可解析正文，直接只读请求返回 HTTP 403；针对日志的公开检索未得到可核实的 2017 Ground 事件清单。不能把“未取得日志”写成“没有事件”。需提供或取得该日志/对应官方说明，确认哪些异常属于仪器无效记录，哪些属于真实停机、维护或积雪。不得根据本轮诊断自动删除事件。
2. **Yulara 资源字段数据谱系尚不完整。** 官网确认联合系统身份和 AC 功率五分钟平均；资源部分列出气温/GHI字段，但要求向提供方索取进一步说明，没有给出对应传感器单位、聚合区间和无效码的完整说明。当前气温/GHI单位候选由字段名称和观测范围支持，不能冒充已获完整官方确认。需补充下载附带元数据或提供方资源字段说明。

Yulara 的未知 UTC 偏移本身不是失败原因；已按用户允许的固定本地坐标处理。区间起止不确定本身也采用允许的保守 availability-time 假设，未通过查看 Test 预测选边界。上述待修正项聚焦提供方字段/质量证据，不通过标签插值、Test调规则或换新场址绕过。

## Git 与内容隔离

已 fetch origin。以 ede66987e56eb8863287624476f8b8ff3e201897 创建独立 research/scheme-a-multisite-data-confirmation 分支/worktree。目标 Draft PR base 为 manuscript/clean-pv-benchmark-jrse-final-polish。不合并 PR #14，不读取或修改 master 主工作树，不修改 C1/NWP/旧稿或原 Scheme A 数据和指标。提交范围限新增扩展目录与忽略本地配置/运行环境的 .gitignore。

指定 base 的正文不是前一任务本地已润色正文；这属于用户明确要求的内容来源选择，不擅自搬运稿件更改。原始数据仅按用户此前明确提供路径读入忽略的 .local/multisite_paths.json，不提交机器绝对路径。

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

27/27 passed，0 failures，0 errors，0 skipped。包括真实365文件/精确缺口、显式bin成员、缺一个分钟或一个变量值时不接受部分均值、Wm2不重复转换、目标字段和sentinel检查、离网格不round、时间缺口不拼接、真实mask及7列顺序、标签不插补、Train-only fit拒绝测试、availability、目标方向和split边界、按timestamp的Daily join、全方法逐元素mask、Train daylight阈值、四模型forward及参数参与、17维随机state拒绝、Test score sentinel及held-out数据变更不影响冻结配置/Train诊断、原始size/mtime保持一致。

预处理fit测试用记录数组的test double，并未对实际KNN、scaler、IF拟合；M1不宣称完成下一轮训练runner的端到端验证。Forward运行时将实际fit、训练/真实预测helper、backward、AdamW及torch save/load设为拒绝调用。测试没有以源码字符串检查替代数组行为。数据审核代码不计算实际模型或基线Test误差；Test支持计数是本轮明确要求的只读检查。

## 主次分析与24-run矩阵

主模型已锁定 INVERTED_VARIATE_TRAJECTORY：原冻结CSV 11,328行复核的primary mean RMSE排序为12/9/2/1，原报告平均排名1.875最佳；在外部Test预测前选定。主要比较为三个seed均值和sample SD，以及分别对Last-value、Daily的matched RMSE skill。其他三模型、排名、每seed、MAE/nRMSE/bias/R²和scope差异为次要；best-of-four包络仅描述性。完整24-run矩阵在 MULTISITE_PROTOCOL.md 和 multisite_config.json 中，候选预算/输入/划分已冻结，authorization_next_round=false。

## 官方来源、已确认与未确认边界

- [NIST Campus数据目录](https://catalog.data.gov/dataset/nist-campus-photovoltaic-pv-arrays-and-weather-station-data-sets)：2015–2018，一分钟/一秒数据和DOI。公开测试场旧说明仅列2015–2016，采用目录及本地2017文件核实年份，不把页面年代差异误判为文件不存在。
- [NIST Data Dictionary v1.0](https://www.nist.gov/document/datadictionarysupplementalcontentpdf)：array部分定义TIMESTAMP为LST/Max，电表AC有功、Ground Pyra1 GHI及平均单位。当前CSV的转换后Wm2列存在性由实文件确认。
- [NIST数据说明论文](https://nvlpubs.nist.gov/nistpubs/jres/122/jres.122.040.pdf)：说明数据错误/中断与清洗日志存在，不能据此推断2017具体事件。
- [NIST门户/DOI](https://doi.org/10.18434/M3S67G)、[测试场说明](https://www.nist.gov/el/beed/heat-transfer-alternative-energy-systems/photovoltaic-testbeds)：已访问，当前门户内容未能读出日志。
- [DKASC Yulara下载页](https://dkasolarcentre.com.au/download?location=yulara)、[联合系统metadata](https://dkasolarcentre.com.au/source/yulara/yulara-3-roof-sails-in-the-desert-2)：106.6 kW、mono-Si、roof、2016安装，子阵列合计系统，不是全部1.8MW Yulara。
- [DKASC Glossary](https://dkasolarcentre.com.au/glossary)：AC功率五分钟平均；Yulara resource部分需提供方进一步说明。没有从Alice Springs气象传感器段落直接套用其参数给Yulara。
- [Notes on the Data](https://dkasolarcentre.com.au/download/notes-on-the-data)及[2017相关页](https://dkasolarcentre.com.au/download/notes-on-the-data/p8)、[相邻记录页](https://dkasolarcentre.com.au/download/notes-on-the-data/p7)：2017-02-14网站迁移影响两处数据访问；2017-05-09 pyranometer角度调整明确属于Alice Springs，不能套给Yulara。日志并非穷尽所有短时事件，不据此删数据。

核查日期 2026-09-05。没有联系提供方或发送邮件。可以在获得缺失文档后继续M1 correction，复核受影响规则和计数，再决定是否READY；当前不授权下一轮训练。

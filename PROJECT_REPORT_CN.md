# Scheme A 完整项目报告：本轮投稿前补强

日期：2026-09-08。当前分支 manuscript/clean-pv-benchmark-multisite-revision，继续PR #19。本轮在原60次神经训练之外完成10个确定性Ridge参考，没有神经重训、原始数据改写或旧checkpoint更新。

## 应先理解的科学问题

本研究不是提出一个新网络。我们研究同一预测任务中，参考历史、目标功率范围和评价样本如何改变对预测收益的判断。原60次训练提供固定证据，本轮增加解释与简单可学习对照。主文新标题为 **Reference Information and Target-Power Regimes in Multi-Window Photovoltaic Forecasting**。

五个目标系统分布于三个地理场址：Alice的Sanyo/Hanwha/Qcells三个共址阵列（2018，17通道，36 runs）；Yulara联合系统与NIST Ground各独立拟合（2017，7通道，24 runs）。六小时输入预测未来十二小时五分钟轨迹。H12/48/96/144分别为未来1/4/8/12小时全部前缀的累计评价，不是四个孤立时刻误差。

## 从数据到结果的完整流程

1. 按明确来源读取数据，保留原始文件。NIST固定EST−05:00且不切换DST，每个五分钟结束时刻只聚合此前五个不同、有限分钟。Yulara保留provider local坐标，两个离网格点不取整，采用+5分钟可用规则。
2. 保留规则时间网格；缺失标签不插补。外部有限负功率/GHI和异常运行状态保留；Alice沿用原非负功率标签规则。
3. 按冻结日期划分Train/Validation/Test。外部1–8月Train、9月Validation、10–12月Test；Alice4月1日至7月15日Train、7月16日至8月7日Validation、8月8–31日Test。
4. 原始缺失mask先生成，Train拟合KNN5；Train插补数值拟合Isolation Forest100棵树、污染率0.01、seed42；追加mask和IF；feature与target scaler仅Train拟合。
5. 四种紧凑实现分别训练：recurrent固定context隐状态轨迹解码、Inverted-variate、Joint-patch、Depthwise TCN。recurrent没有上一预测功率反馈；其余三种用直接多输出头。不能称为官方完整模型复现。
6. 历史神经训练按完整H144窗口、固定预算、Validation全局SSE/count选择checkpoint；Test不参与。三seed衡量训练随机性。外部主模型在外部Test预测前固定为Inverted-variate；Alice焦点模型和事后包络为描述性选择。
7. 主分析按各累计窗口保留合法origins；Daily用目标时间精确−24小时连接。任何Daily匹配比较将同一逐点mask应用于所有方法。真实未来功率>Train最大值1%的集合改称power-active；补集low-power，不等于夜间。
8. 本轮只读60份预测NPZ，新增common-H144、固定144个lead、分月、固定日历/明确事后极端案例、active/low-power可加和误差、配对时间块区间。
9. 事后Ridge A展平原72步输入；B增加144个可用前一日功率与144个缺失mask。Train中位数补lag输入、Train标准化；alpha从0.1/1/10/100/1000按完整H144 Validation MSE选择。五系统均执行，确定性方法不伪造三seed。
10. 图件、表格和文章根据全部结果修改，保留不利结果。完整审核包附预测、标签、时间戳、mask、checkpoint、处理器、代码和来源说明；它不等于公开数据许可或实际投稿。

## 本轮实际发现

原冻结结论不变：Alice主比较12/9/2/1/0；Daily对神经事后包络22/24较优；Hanwha1h full/active为两个例外。外部预指定Inverted相对Last-value16/16、Daily15/16；外部事后包络对Daily是Yulara8/8、NIST7/8，不能把两层结果合成40项。

NIST12h Daily交集共3,128,676点，其中active 1,098,178、low-power 2,030,498（64.90%）。Inverted逐seed RMSE均值full44.609、active60.544、low32.913 kW；Daily分别41.899、69.748、8.602 kW。神经active对full MSE的贡献减少420.640 kW²，但low贡献增加655.384 kW²，净增加234.744 kW²。低功率正偏差解释统计上的抵消，不据此推断夜间、积雪或故障。

48h配对时间块区间（2000次）显示Yulara12h/full对Daily skill为1.62%，区间−9.47%至9.85%；NIST12h/full为−6.47%，区间−17.14%至2.28%。这些是条件于日期和已拟合模型的时间变化范围，与seed SD分开。24/72h敏感性全部提供，不按区间是否跨零调整方案。

全部60份训练历史均核对了最佳epoch；NIST TCN三seed均在25轮上限最佳。这限制了充分调优的架构结论，但没有触发重训。

## 新线性参考：12h full Daily-matched RMSE（kW）

|系统|Inverted三seed均值|Daily|Ridge A|Ridge B（日周期）|
|---|---:|---:|---:|---:|
|Sanyo|0.775|0.118|0.914|0.212|
|Hanwha|0.805|0.203|0.844|0.248|
|Qcells|0.844|0.244|0.839|0.237|
|YULARA_COMBINED|15.805|16.066|119.585|35.950|
|NIST_GROUND|44.609|41.899|48.956|34.995|

日周期信息降低了所有五系统的Ridge12h误差，但没有形成统一最优方法。NIST的Ridge B超过Daily与现有主模型；Yulara线性参考明显较差，仍完整报告。不能据此宣称同样增强神经输入后的效果。

Qcells共同起报敏感性进一步揭示样本构成：1h原6463 origins/77556 full points/36504 active points，改用完整H144 origins后为2996/35952/42。共同子集Last-value RMSE0.00398 kW，Inverted0.12494 kW，且Last-value低于全部四神经模型。原主分析不变，但不能把其收益推广到这个几乎全为low-power的子集。

## 验证和交付边界

- 本轮：60份原预测读取；2400个可加和SSE检查；8个synthetic forward检查；565个Ridge/历史记录检查（含480指标行）；6720个点数组归约与冻结CSV对照，失败/跳过均0。
- 605个已有保护快照文件的size与mtime_ns在本轮再次比较，无差异。Alice原始CSV的本轮最初读取前没有单独快照，因此不冒称有该阶段的前后对照；代码仅只读访问，完整打包阶段另对所有源文件做前后size/mtime检查。
- 历史43/17/10测试和10432独立比较没有在本轮全套重跑，明确标为历史。也未重做60checkpoint前向复现；原M3结果和本轮数组重算并列保留。
- 新主文10页、Supplement36页；5主图＋9补图。摘要196词、正文TeXcount2895词（旧4245，同口径）；图表/参考另计，详见WORD_COUNTS.json。
- 当前图件全部交付矢量PDF/SVG、320dpiPNG、CSV、caption/alt；实际逐页检查记录见FIGURE_QA.md。原旧图保留为历史对照。
- 选刊建议：Renewable Energy冲刺→Solar Energy→JRSE。中科院正式分区仍待机构核验，不能称已按核实分区递降；风险与近年同类文章详见JOURNAL_STRATEGY_CN.md。
- 尚无跨全年/跨年份的同协议确认，没有经济效益实测。下一步最小实质补强是已有外部场址的不同季节或次年连续性能期，而不是扩充复杂网络。本轮不启动。
- 本轮作者身份与声明最终签核、许可证、公开URL和费用路线未代填；未改变公开状态、未投稿。

## 后附：原M3完整流程与冻结数值背景（历史记录）

下文为此前项目报告，保留其完整实验背景。涉及旧标题、13/29页、旧图数、旧包不含artifact等内容均属于历史M3交付，当前状态以上文及START_HERE_REVIEW.md为准；原冻结数字仍有效。

---

# Scheme A 光伏预测多场址项目完整报告

**版本：2026-09-08｜用途：导师审核、团队交流与项目归档**

**事实基线：**M3 提交 `8070d6b8fc9f11f596ef9990ab1b8005321160ae`，Tree `84fea74ba489f0907d512d194a13140d2f487295`；[Draft PR #19](https://github.com/Zhujiangkun-Yohkoh/PV-forecast/pull/19)。本报告依据该版本的正式配置、审计报告、指标表及论文编写。阶段性文档中的“下一轮允许训练”属于当时的授权记录，不代表本次报告编写启动训练。

本报告不是论文正文，因此保留必要的实验过程、修正原因、审核步骤和交付状态。历史阶段结论与当前最终结论分开说明；不把旧稿页数、旧路径或未经确认的声明当作当前事实。本次只整理既有证据，没有执行训练或重新生成预测。

## 1. 项目概述与核心问题

本项目研究：在明确历史信息、预测时间、数据划分和评价样本的条件下，四种紧凑神经网络在光伏多时域预测中的表现如何，以及结论是否随基线、场址、预测时域和评价范围而变化。

研究对象是未来 AC 有功功率。模型每次读取六小时历史，输出未来十二小时的五分钟轨迹，再分别评价未来 1、4、8、12 小时。项目不是新增算法论文，也不以证明某个网络普遍最佳为目标。贡献在于构建可审计的比较条件，并保留会改变结论的基线与场址差异。

当前证据由两个层次组成：

| 层次 | 研究设置 | 目的 | 历史训练规模 |
|---|---|---|---:|
| Level 1 | Alice Springs，同一设施的三个共址技术阵列，2018 年，17 通道 | 比较共享场址天气背景下不同阵列与模型的表现 | 3 阵列 × 4 模型 × 3 seed = 36 |
| Level 2 | Yulara 与 NIST Ground，2017 年，7 个公共通道，分别拟合 | 检查预先指定模型的优势是否在不同外部设施重现，并观察排名变化 | 2 场址 × 4 模型 × 3 seed = 24 |

合计 60 次历史训练，对应五个功率目标系统、三个地理场址。Alice 的三个阵列不能被算作三个独立气候场址；Yulara 的联合系统不能拆成多个独立场址。三个地理场址的数据没有池化训练，外部实验没有使用 Alice checkpoint 做零样本迁移。

最重要的结果是：神经预测相对 Last-value 的优势具有重复出现的证据，但 Daily 基线的结论具有明显的场址依赖性。Inverted-variate 在 Yulara 保持领先，在 NIST 的平均神经排名为 2.25。由此支持的是有条件的 benchmark 结论，而不是跨气候普遍泛化或纯架构因果结论。

## 2. 项目从原始 Scheme A 到 M3 的推进过程

| 阶段 | 主要工作 | 阶段产物与结论 |
|---|---|---|
| Alice 最终公平修正 | 保留历史功率输入，明确 17 通道、缺失 mask、Train-only 处理；核对 horizon-specific 样本；修正 Daily 的目标交集；从 artifact 核对效率数据 | 冻结 36 runs、正式 corrected_metrics.csv 和独立证据审计 |
| 原 Scheme A 投稿整理 | 将稿件定位为应用/benchmark，保留 Daily 的不利结果，整理作者和投稿材料 | 形成共址技术比较的完整稿件；最终作者签核与代码发布仍待确认 |
| M1：外部数据确认 | 只读审查两场址原始文件、字段和时间；冻结 7 通道、窗口、基线及 24-run 矩阵；synthetic forward-only 检查 | 不训练，不用预测选择数据规则 |
| M1-R：证据缺口分级与时间修正 | 固定 NIST tz-aware EST；将无法取得的运行日志和部分 Yulara 元数据按明确回退规则列为非阻塞限制 | 43 项测试通过，协议可进入下一轮冻结训练 |
| M2：授权训练 | 执行固定 24 次 GPU 训练；按 Validation 选择 checkpoint；冻结后生成 Test 预测；保存 artifact 并独立复算 | 24/24 完成，无数值发散，10,432 项独立比较通过 |
| M3：论文整合 | 重新核查证据；程序化跨设置比较；重构叙事、图表、补充材料与投稿包；编译逐页审核 | 当前状态为供作者审核，未实际投稿 |

原 Alice 的 Daily 比较曾存在“Daily 用自己的有效点、神经模型用另一套点”的口径问题。最终修正是把相同的逐元素交集同时用于所有方法。最终 Daily 22/24 的数字虽与历史表述一致，但正式证据来自修正后的相同目标点比较；旧的非匹配口径不再作为胜场依据。Qcells H12 样本数和效率统计也以最终 artifact 核验为准。

M1-R 没有通过忽略物理标签或时间问题来降低要求，而是区分：哪些缺口会造成错误样本或未来信息泄漏，哪些只是提供方详细文档不完整。后者在规则明确、原始有限观测保留的情况下作为限制记录。

M2 曾出现审计 JSON 不能直接序列化 NumPy int64 的实现错误。处理方式是转换为原生 Python 标量，并完整重跑 artifact 测试；没有因此增加训练 run，也没有改变预测或指标算法。

M3 的 Alice artifact 定位一度阻塞。作者明确给出正确目录后，核实 36 组文件完整、身份和形状正确，才解除阻塞。路径修正没有改变协议或结果。

## 3. 数据来源与研究对象

### 3.1 Alice Springs

数据来自 DKA Solar Centre 的官方 Alice Springs 服务。三个目标阵列为：

| 阵列 | 模块技术与标识 | 描述性 DC 容量 |
|---|---|---:|
| Sanyo | Site 17，HIT-210NKHE5，混合硅技术 | 6.30 kW |
| Hanwha | Site 25，HSL 60S，多晶硅 | 5.83 kW |
| Qcells | Site 38，Q.PEAK-G4.1，单晶硅 | 5.90 kW |

三个阵列共享场址天气背景，但朝向、系统部件、传感器和运行行为仍可能不同。因此，阵列差异不被归因为“电池技术的纯因果效应”。容量用于描述设施，误差归一化采用各目标 Train 功率范围，不能把 DC 容量直接当作 AC 功率归一化分母。

每个阵列使用 8 个数值字段，正式顺序为 Performance_Ratio、Weather_Temperature_Celsius、Weather_Relative_Humidity、Global_Horizontal_Radiation、Diffuse_Horizontal_Radiation、Radiation_Global_Tilted、Radiation_Diffuse_Tilted、Active_Power。历史功率和 PR 是阵列特异变量，其余六个是天气变量。之后追加对应的 8 个原始缺失 mask 和 1 个 Isolation Forest 标记，共 17 通道。这里只使用历史字段，没有将未来 PR 或天气提供给模型。

Alice 保留原冻结 ACST 时间坐标和五分钟网格。其正式标签有效性规则排除负功率；这一点与外部保留有限负功率的规则不同，必须在跨设置解释中承认，不能追溯修改原结果。

### 3.2 Yulara

Yulara 使用 Sails in the Desert 的 106.6 kW rooftop combined system，一个设施级联合功率目标。文件名中的 2016 与系统页面的年份标签不能代替实际数据覆盖范围；实际文件覆盖 2016-04-01 23:45 至 2026-02-19 23:55。本实验只取 2017 年。

2017 年共有 105,122 条独立时间记录，其中 105,120 条位于完整五分钟规则网格，另外两条离网格记录为：

| 原时间戳 | Active_Power（kW） | 处理 |
|---|---:|---|
| 2017-02-22 15:30:02 | 75.150801518689 | 排除，不取整；温度和 GHI 缺失 |
| 2017-08-07 05:20:01 | -0.047513291616942 | 排除，不取整；温度和 GHI 缺失 |

相邻的规则时间记录已经存在，不能把这两条记录 round 到网格后覆盖正常观测。排除只发生在派生读取逻辑，原 CSV 保持不变。“规则时间完整”也不等于“每个变量没有空值”。

三个字段固定为 Active_Power、Weather_Temperature_Celsius、Global_Horizontal_Radiation。温度按提供方 Celsius 字段使用；GHI 按提供方原生数值使用，不增加单位转换。尚未完整确认的传感器型号、不确定度和无效码定义不作猜测，也不用该原生 GHI 做跨站绝对辐照量推断。

### 3.3 NIST Ground

NIST Ground 位于 Gaithersburg，描述性 DC 容量 271 kW。2017 年下载包含 365 个日 CSV；表头一致，合计 525,595 个唯一且有序的一分钟时间点。完整非闰年应有 525,600 分钟，缺失的五分钟准确为：

- 2017-10-20 23:59:00−05:00；
- 2017-10-21 00:00:00、00:01:00、00:02:00、00:03:00，均为 −05:00。

公共字段及谱系如下：

| 用途 | 正式字段 | 含义与使用方式 |
|---|---|---|
| 历史功率、未来标签 | PwrMtrP_kW_Avg | AC 电表有功功率，kW |
| 气温 | AmbTemp_C_Avg | 环境温度，摄氏度 |
| GHI | Pyra1_Wm2_Avg | Ground GHI，CSV 已提供 W/m²，直接使用 |

当前下载没有 Pyra1_mV_Avg，不进行重复 mV 转换。InvPAC_kW_Avg 有 4,712 个 −999 记录，但该列从一开始就不属于输入或标签；不是先拿它作目标再事后清洗。

### 3.4 外部原始字段质量摘要

以下统计是 2017 年原始字段审计，发生在可用时间平移和五分钟聚合之前。空值之外，所列六字段的非数值计数和无穷值计数均为 0；NIST 缺失时间行不等同于下表字段空值。

| 场址 | 字段 | 空值 | 最小值 | 中位数 | P95 | 最大值 |
|---|---|---:|---:|---:|---:|---:|
| Yulara | AC 功率 | 64 | -0.059 | -0.035 | 88.133 | 109.903 |
| Yulara | 气温 | 2,950 | -2.400 | 22.433 | 36.100 | 43.210 |
| Yulara | GHI（原生） | 2,950 | -12.872 | 1.458 | 970.915 | 1,391.547 |
| NIST | AC 电表功率 | 2,127 | -0.647 | 0.000 | 199.400 | 258.400 |
| NIST | 气温 | 2,127 | -12.820 | 15.230 | 29.260 | 38.850 |
| NIST | GHI（W/m²） | 2,147 | -18.771 | -5.703 | 787.316 | 1,412.584 |

有限负值不等于无效码。NIST 原始分钟中有 14,418 个负功率、271,221 个负 GHI，以及 751 个 GHI 大于 500 而功率不大于零的记录。Yulara 有 52,807 个负功率、50,504 个负 GHI，类似高 GHI/非正功率记录为零。这些是描述性运行分布计数，不是筛除条件，也不用于推断站间绝对辐照差异。

## 4. 时间语义与五分钟聚合

### 4.1 为什么时间戳不足以保证无泄漏

一个标为 10:00 的平均值可能代表前五分钟，也可能代表从 10:00 开始的五分钟。若后者在 10:00 就作为输入，会隐含未来观测。因此协议不仅记录时间戳，还记录 measurement interval 和 availability time（完整值最早可以使用的时刻）。

### 4.2 NIST 固定 EST/LST

解析、聚合、规则网格、split、窗口和 Daily join 全程保留固定 UTC−05:00 的 tz-aware 坐标，time_basis 为 FIXED_EST_LST。不使用 America/New_York，不引入夏令时重复小时或缺失小时。

聚合以当地午夜为 anchor，origin 为 2017-01-01 00:00:00−05:00；左闭右开、右端标记。对输出时刻 T，只允许 T−5、T−4、T−3、T−2、T−1 五个不同分钟，即区间 [T−5 min,T)。每个变量分别判断五个值是否全部有限：全部满足才取均值，否则该变量的整个 bin 为 missing。不能用四个点的均值伪装完整五分钟，也不能从相邻 bin 借值。

例如输出 10:05 对应 10:00–10:04 的五个分钟记录，最早在 10:05 使用。若 10:03 的气温缺失，则该 bin 的气温缺失；若功率五点完整，功率仍可有效。标签沿相同冻结可用时间坐标定义。

Train-only 时间诊断比较过电表累计能量增量与平均功率，以及功率/GHI 对应关系。冻结报告记录未平移平均绝对能量差 0.164 kWh，相邻分钟平移约 0.226 kWh；无五分钟错位时功率/GHI 相关约 0.972，左右错位约 0.929。这些是数据诊断，不是用 Validation/Test 预测误差调规则。由于当前导出区间约定仍未完全证明，最终保留完整测量区间结束后才可用的保守假设。

### 4.3 Yulara 保守可用时间

保留固定 provider local coordinate，状态为 PROVIDER_LOCAL_TIME_OFFSET_UNCONFIRMED，不凭地理常识伪造 UTC 时间。规则记录 t 的派生可用时刻固定为 t+5 分钟，不取整两条离网格记录，不引入 DST。

本实验不进行跨场址同步或池化，固定偏移尚未完整确认不自动导致实验无效；但这一规则不能被描述为精确完成了跨站 UTC 同步。

## 5. 时间划分、窗口与目标有效性

| 设置 | Train | Validation | Test |
|---|---|---|---|
| Alice，2018 | 04-01 00:00 至 07-15 23:55 | 07-16 00:00 至 08-07 23:55 | 08-08 00:00 至 08-31 23:55 |
| 两个外部场址，2017 | 01-01 至 08-31 | 09-01 至 09-30 | 10-01 至 12-31 |

外部日期按各数据集冻结固定坐标解释至当日 23:59:59。时间划分顺序固定，不随机打乱 Train/Validation/Test 身份。训练批次的内部抽样不等于随机分割时间序列。

输入为 `[B,72,D]`，D 为 17 或 7；输出为 `[B,144]`。forecast origin 是发出预测的时间，最后一步历史可用时间不晚于 origin，第一目标为 origin+5 分钟。

| 前缀 | 点数 | 对应预测长度 |
|---|---:|---|
| H12 | 12 | 1 小时 |
| H48 | 48 | 4 小时 |
| H96 | 96 | 8 小时 |
| H144 | 144 | 12 小时 |

规则网格不能通过删除缺失行后重新拼接。输入测量区间和目标不能跨 split。外部 split 起点午夜那个结束 bin 的测量区间属于前一 split，不能作为本 split 输入；第一段历史从午夜开始、00:05 可用，72 步形成的理论最早合法 origin 为 06:00。实际首个合法 origin 还取决于真实功率和目标是否有效。

训练和 checkpoint 选择使用完整 H144 标签窗口；外部还要求 origin 真实功率有限。Test 则对每个 horizon 独立判断完整前缀：H12 不会因为第 13–144 步缺失而被删除。输入允许缺失，经 Train-fitted 处理；标签永不插补。这里并非以插值提高样本数，也不是按预测误差挑样本。

forecast-origin count 是具有可评价点的起报次数；valid-target count 是“起报 × 提前步”有效点数。不同起报的预测轨迹可覆盖同一物理时间，因此后者不是独立时间观测数。daylight origin count 表示至少包含有效 daylight 点的起报数，不等于 origin 自身处于白天。

外部 DATA_AUDIT_SUMMARY.csv 保存 2 场址 × 3 split × 4 horizon × 2 analysis × 2 scope = 96 组支持记录，包含首末 origin、月份、输入缺失率及标签缺失率。合法前缀选定后的标签缺失率为零，不意味着原 split 没有缺失。

## 6. 公共输入与 Train-only 预处理

外部七通道严格依次为：历史 AC 功率、气温、GHI、power_missing、temperature_missing、ghi_missing、isolation_forest_flag。没有额外时刻编码、年内日期、湿度、PR、未来 NWP 或场址特异字段。

处理顺序如下：

1. 从原始数值输入记录缺失 mask；非数值、NaN 和无穷值视为缺失。
2. KNNImputer，n_neighbors=5，仅在 Train 数值输入拟合；Validation/Test 只 transform。
3. Isolation Forest 仅在 Train 已插补数值上拟合：100 棵树，contamination=0.01，random_state=42。
4. 将原始 mask 和 IF 标记追加到数值变量，组成固定七通道。
5. feature MinMaxScaler 仅在 Train 七通道上拟合，指示列也遵循冻结缩放顺序。
6. 独立 target MinMaxScaler 只使用 Train 有限、未插补功率标签拟合。Alice 遵循其冻结标签有效性及八数值变量版本。

IF 是输入异常提示，不是删除标签的工具。停机、雪、维护、高辐照低功率、有限负功率和负 GHI 不因经验规则被删掉。提供方没有明确无效码定义时，不把有限异常数值擅自重编码为缺失。

Train-only 处理意味着未用 Validation/Test 拟合变换，不表示本项目证明了严格在线逐时更新能力。KNN 可以利用 Train 集中的其他历史样本；整个任务是按固定历史训练期建立模型，再评价后续 held-out performance split。

## 7. 四种紧凑模型与可比性

| 模型 | 本项目实现的核心机制 | 关键冻结配置 | 17 通道参数 | 7 通道参数 |
|---|---|---|---:|---:|
| Discrete recurrent | 多尺度卷积与紧凑 Transformer 分支、门控融合、双向 GRU 摘要和递归轨迹解码 | embedding=64，卷积分支=24，4 heads，1 Transformer layer，dropout=0.1 | 99,362 | 96,562 |
| Inverted-variate | 将每个变量的 72 点历史作为 token，编码后输出轨迹 | d_model=64，4 heads，1 layer | 194,960 | 102,800 |
| Joint-patch | 对多变量时间片段联合编码 | d_model=64，4 heads，1 layer，patch=12，stride=6 | 148,112 | 140,432 |
| Depthwise TCN | 输入投影、深度/逐点时间卷积块与直接轨迹输出 | channels=64，4 layers，kernel=5 | 683,024 | 682,384 |

这些是 compact project implementations，不是官方完整 iTransformer、PatchTST 或 ModernTCN 复现。模型结构和参数量不同，因此结果不能被解释为只改变某个架构因素的因果实验。

7 通道引起的参数变化来自必要的输入或输出投影维度，不构成新模型，也不允许把原 17 通道权重转换成“已迁移”的 7 通道权重。M1/M1-R 的 synthetic forward-only 测试检查 `[B,72,7]→[B,144]`、参数张量对前向输出的参与及 strict 17 通道状态不兼容；这些测试不等于训练效果验证。

## 8. 训练设计与 Test 隔离

两层实验采用相同的冻结训练预算：

| 参数 | 值 |
|---|---|
| seeds | 42、43、44 |
| optimizer | AdamW |
| learning rate / weight decay | 0.001 / 1e-5 |
| batch size | 256 |
| 最大训练轮数 / patience | 25 / 5 |
| min_delta | 1e-8 |
| gradient clip norm | 1.0 |
| num_workers | 0 |
| checkpoint 选择 | 完整 H144 Validation global masked MSE |

上述训练操作是历史 M2 已授权执行的过程说明，不是本报告下发的新训练命令。没有超参数搜索、best-seed 选择或依据模型成绩分配不同预算。实际停止时间可以因统一 early stopping 规则而不同。

Validation 目标为 `Σ所有有效点平方误差 / 有效目标总数`，不能将每个 batch 的平均损失等权平均，因为最后一个 batch 大小可能不同。每个 run 保存 Validation 最优 checkpoint。H12/H48/H96 Test、Daily 成绩、胜场和曲线观察都不参与选择。

M2 在全部 24 个 checkpoint、处理器、Train range、daylight 阈值、seed、模型结构和评价配置冻结并写入 test_release.json 后，才构建 Test loader。Test 曾用于数据质量审核，因此准确表述是 held-out performance split；不能称为完全未查看的独立外部验证集。

M2 使用 NVIDIA GeForce RTX 3060 Laptop GPU，PyTorch 2.7.1+cu118、CUDA 11.8、Float32、无 AMP、4 CPU 线程。累计训练 3.607 小时；脚本端到端（含预处理与首次预测）3.680 小时。这不是原 60 runs 的总耗时，也不是纯推理耗时。

24/24 runs 完成、无神经预测 NaN/Inf 或数值发散。出现非有限预测应令整个 run 失败，不能删除这些预测点后继续报好成绩。

## 9. 基线、共同目标与 daylight

Last-value 使用 origin 的真实有效功率，未来每步重复该值；不使用插补后的 origin。它衡量模型是否超越短时连续性。

Daily 对每个未来目标时间 τ 取真实功率 P(τ−24h)。通过精确 timestamp join 获取，不按行数 shift。缺失时间不能导致“昨天”错位。24 小时滞后值只要源记录存在且在预测时可用，可以来自 split 边界之前；这与禁止神经输入/目标窗口跨 split 的规则不矛盾。

primary 中神经模型与 Last-value 使用相同 origins、labels、point masks。supplementary_daily_matched 再与 Daily 有效滞后点取交集，并把完全相同的逐元素 mask 用于 Daily、Last-value 和四模型全部 seeds。两个 analysis 的目标点数可能不同，不能跨 analysis 直接拿 RMSE 作 skill 分子分母。

相同目标 mask 保证比较的是同一批预测任务，但不保证输入信息相同：神经历史为六小时，Daily 使用提前 24 小时的目标对齐轨迹。这是信息策略比较，而非纯架构比较。

Daylight 定义为真实未来功率大于对应目标 Train 最大功率的 1%。阈值只由 Train 得出，membership 依赖真实未来标签，所以它是事后描述性 scope，不是可部署 daylight 检测器。full 评价所有有效点，包括合法夜间或低功率运行状态。

## 10. 指标与统计规则

记同一评价 mask 上观测为 y、预测为 ŷ，共 N 点：

- RMSE = sqrt[Σ(ŷ−y)²/N]，单位 kW。
- MAE = Σ|ŷ−y|/N，单位 kW。
- bias = Σ(ŷ−y)/N；正值表示平均高估。
- R² = 1−Σ(ŷ−y)²/Σ(y−平均y)²；负值保留，目标方差零时 undefined。
- Train-range nRMSE = RMSE/(Train 最大功率−Train 最小功率)。不使用 Test range 或假设容量。
- RMSE skill = 1−RMSE_model/RMSE_reference。正值更优，负值参考方法更优，参考 RMSE=0 时 undefined，不加 epsilon。

每个 seed 先独立评价，再报告三个指标值的均值和 sample SD（ddof=1，分母为 2）。不是平均三个预测后得到 ensemble 成绩。确定性基线按 seed 键重复用于配对，不代表三次独立拟合。

SD 只反映这三个初始化/训练随机性的差异，不是时间采样总体不确定性或置信区间。相邻 origins、不同 horizon 和 full/daylight 大量重叠，因此 8/8、15/16、22/24 都是相关的描述性比较，不能直接据此计算 p 值。

不平均跨场址 kW 形成总体冠军。Alice 三阵列的 rank 汇总也不是三个独立地理重复。原五方法排名与 M3 神经四方法排名的候选集合不同，均值可不同；这不代表冻结结果被修改。

## 11. 预先指定分析与结果解释顺序

外部主模型固定为 INVERTED_VARIATE_TRAJECTORY，依据它在冻结 Alice 中的 12/24 primary wins 和最佳平均排名，在外部 Test 预测生成前选择。

主要分析分别报告其两个外部场址、四个 horizon、两个 scope 的 RMSE、MAE、nRMSE、Last-value skill 和 Daily-matched skill。次要分析保留全部四模型、三个 seed、bias、R² 和场址内排名。

post hoc descriptive envelope 是每个条件下从四模型的三 seed 平均 RMSE 中事后取最小值，不是选 best seed。它是偏向神经方法的描述性上界，不能当成预指定模型或可部署模型。Alice 的 Inverted-variate 对照称为 descriptive focal model，不能倒推说它在 Alice 本身也是预注册主模型。

## 12. Alice Springs 正式结果

冻结 primary 最低 RMSE 胜场为 Inverted-variate 12、Depthwise TCN 9、Joint-patch 2、Discrete recurrent 1、Last-value 0，共 24 个 array×horizon×scope 条件。至少一个神经模型在每项中优于 Last-value，但不能由此声称每个神经模型每项都优于 Last-value。

Inverted-variate 焦点模型相对 Last-value 为 24/24，相对 Daily 为 0/24；事后四模型包络相对 Daily 为 2/24。Daily 的 22/24 广泛优势是 Alice 结果的关键部分。

两个神经例外均为 Hanwha H12：full 中 Depthwise TCN 约 0.176 kW 对 Daily 0.185 kW；daylight 中约 0.231 kW 对 0.273 kW。这些结果不能扩展为其他阵列或长时域的优势。

Qcells H12 的支持保留 6,463 origins、77,556 full 点、36,504 daylight 点；full/daylight Last-value RMSE 约 0.471/0.682 kW，Inverted-variate 约 0.327/0.457 kW。H144 仅 2,996 origins，说明不同 horizon 的可用样本本来就不同，不能为了统一 H144 样本而丢弃合法 H12 任务。

原效率测试在 RTX 3060 Laptop GPU 上测得单样本平均延迟：recurrent 32.204 ms、Inverted-variate 0.558 ms、Joint-patch 0.535 ms、TCN 0.706 ms。Joint-patch 较快，Inverted-variate 平均精度排名较好。这是特定环境的模型推理比较，不含数据读取/预处理，也不证明其他硬件上相同排序。

## 13. 外部预指定主模型完整结果

下表直接取自冻结 M2 报告。均为三 seed 的均值 ± sample SD；RMSE/MAE 为 kW，nRMSE 与 skill 以百分比显示。Daily-matched RMSE 具有自己的共同目标交集，因此 NIST 中可能与 primary RMSE 略有不同。

| 场址 | H | Scope | RMSE | MAE | nRMSE % | Last skill % | Daily-matched RMSE | Daily skill % |
|---|---:|---|---|---|---|---|---|---|
| Yulara | 12 | full | 10.2894 ± 0.0320 | 5.8883 ± 0.2815 | 9.3580 ± 0.0291 | 17.3445 ± 0.2573 | 10.2894 ± 0.0320 | 36.2783 ± 0.1984 |
| Yulara | 12 | daylight | 13.9882 ± 0.0627 | 9.4546 ± 0.3257 | 12.7220 ± 0.0570 | 18.7298 ± 0.3641 | 13.9882 ± 0.0627 | 37.2786 ± 0.2810 |
| Yulara | 48 | full | 13.3676 ± 0.1396 | 8.3598 ± 0.1466 | 12.1576 ± 0.1270 | 49.2413 ± 0.5302 | 13.3676 ± 0.1396 | 17.2469 ± 0.8644 |
| Yulara | 48 | daylight | 17.6587 ± 0.2419 | 12.8699 ± 0.0307 | 16.0603 ± 0.2200 | 48.5792 ± 0.7044 | 17.6587 ± 0.2419 | 20.8304 ± 1.0845 |
| Yulara | 96 | full | 15.6990 ± 0.0556 | 10.0281 ± 0.0671 | 14.2779 ± 0.0506 | 61.3217 ± 0.1371 | 15.6990 ± 0.0556 | 2.7132 ± 0.3448 |
| Yulara | 96 | daylight | 20.9265 ± 0.1501 | 16.0784 ± 0.0327 | 19.0322 ± 0.1365 | 56.0975 ± 0.3148 | 20.9265 ± 0.1501 | 6.1000 ± 0.6734 |
| Yulara | 144 | full | 15.8051 ± 0.1553 | 10.2257 ± 0.0838 | 14.3744 ± 0.1412 | 67.1005 ± 0.3233 | 15.8051 ± 0.1553 | 1.6245 ± 0.9666 |
| Yulara | 144 | daylight | 21.2026 ± 0.2208 | 16.5189 ± 0.3122 | 19.2833 ± 0.2008 | 59.4622 ± 0.4222 | 21.2026 ± 0.2208 | 4.4970 ± 0.9946 |
| NIST Ground | 12 | full | 17.8224 ± 0.1131 | 10.2601 ± 0.3503 | 6.9083 ± 0.0439 | 17.9290 ± 0.5210 | 17.7851 ± 0.1158 | 57.7292 ± 0.2753 |
| NIST Ground | 12 | daylight | 28.4044 ± 0.0934 | 20.1259 ± 0.2955 | 11.0100 ± 0.0362 | 20.2400 ± 0.2624 | 28.3641 ± 0.0903 | 59.0101 ± 0.1305 |
| NIST Ground | 48 | full | 28.5043 ± 0.2474 | 17.0150 ± 0.6350 | 11.0488 ± 0.0959 | 41.1576 ± 0.5106 | 28.4930 ± 0.2419 | 32.0866 ± 0.5767 |
| NIST Ground | 48 | daylight | 44.4344 ± 0.9937 | 32.9668 ± 0.7233 | 17.2235 ± 0.3852 | 37.9786 ± 1.3870 | 44.4196 ± 0.9822 | 36.0656 ± 1.4137 |
| NIST Ground | 96 | full | 40.2155 ± 0.7406 | 25.0934 ± 0.7715 | 15.5882 ± 0.2871 | 42.1821 ± 1.0647 | 40.2074 ± 0.7514 | 3.8822 ± 1.7963 |
| NIST Ground | 96 | daylight | 55.6752 ± 0.7810 | 43.2489 ± 0.4544 | 21.5807 ± 0.3027 | 39.2546 ± 0.8521 | 55.6119 ± 0.7839 | 20.2194 ± 1.1245 |
| NIST Ground | 144 | full | 44.6173 ± 0.6843 | 28.5578 ± 0.4564 | 17.2944 ± 0.2652 | 43.2438 ± 0.8705 | 44.6093 ± 0.6903 | -6.4675 ± 1.6476 |
| NIST Ground | 144 | daylight | 60.6101 ± 1.0335 | 48.4096 ± 0.6992 | 23.4935 ± 0.4006 | 37.7957 ± 1.0606 | 60.5445 ± 1.0406 | 13.1951 ± 1.4919 |

Yulara 主模型对 Last-value 为 8/8、对 Daily 为 8/8；NIST 对 Last-value 为 8/8、对 Daily 为 7/8。外部主模型合计对 Last-value 16/16，对 Daily 15/16；这里的合计仅描述两场址这组预先固定的外部条件，不与 Alice 的 24 项相加。

Yulara H96/H144 对 Daily 的提升较小，例如 H144/full 约 1.62%，需保留幅度，不能只写“全部获胜”。NIST H144/full 的 Daily skill 为 −6.47%，意为主模型 RMSE 比 Daily 高约 6.47%；同一 horizon 的 daylight skill 为 +13.20%，清楚表明 scope 能改变比较方向。

## 14. 模型排名、包络与跨设置结论

| 设置 | Recurrent 神经平均排名 | Inverted 神经平均排名 | Joint-patch 神经平均排名 | TCN 神经平均排名 |
|---|---:|---:|---:|---:|
| Alice（三共址阵列的 24 条件） | 3.583 | 1.875 | 2.375 | 2.167 |
| Yulara（8 条件） | 3.500 | 1.000 | 2.500 | 3.000 |
| NIST（8 条件） | 1.750 | 2.250 | 4.000 | 2.000 |

Yulara 的八项神经比较均由 Inverted-variate 领先。NIST 四项 full 由 recurrent 领先；H12/daylight 由 Inverted-variate 领先；H48、H96、H144/daylight 由 TCN 领先。排名不具备场址不变性。

程序化包络对 Daily 为：Yulara 8 胜 0 负，NIST 7 胜 1 负。NIST H144/full 即使使用有利的事后四模型选择也仍是 Daily 获胜；包络 skill 约 −5.94%。包络与主模型在外部恰好有相同胜场数，不意味着两种分析等价或误差相同。

Last-value 只延续当前水平，无法自行描绘日夜转换与未来功率曲线；利用近期功率和气象历史的模型在这些设置下提供了额外信息。Daily 则提供昨日对应时刻的整段日周期轨迹，Alice 结果显示该信息在其 August Test 期很有价值。

外部 Daily 模式不同的可能解释包括逐日天气与运行状态变化、评价月份差异，以及昨日轨迹对当前设施的适配程度；现有实验没有隔离这些因素，因此只是可能解释。NIST 的 full/daylight 权重不同，可能反映夜间/低功率与日间轨迹上的误差分布差异，也不能直接归因为某个结构造成了胜负。

17 与 7 通道、2018 与 2017 年、不同容量、不同月份和标签有效性规则共同限制了跨设置排名合并。外部证据增加了对条件依赖性的观察，而不是证明一个统一输入模型跨气候无条件泛化。

## 15. 完整外部运行矩阵与训练记录

以下为已完成 M2 的历史记录；本次没有重跑这些训练。

| 场址 | 模型 | Seed | 参数量 | 实际epochs | Best epoch | Validation global MSE | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| Yulara | Discrete recurrent | 42 | 96562 | 13 | 8 | 0.0423451287 | 1394.83 |
| Yulara | Discrete recurrent | 43 | 96562 | 18 | 13 | 0.0291177843 | 2175.77 |
| Yulara | Discrete recurrent | 44 | 96562 | 11 | 6 | 0.0422639589 | 1152.86 |
| Yulara | Inverted-variate | 42 | 102800 | 25 | 20 | 0.015971821 | 198.39 |
| Yulara | Inverted-variate | 43 | 102800 | 16 | 11 | 0.0168455096 | 128.18 |
| Yulara | Inverted-variate | 44 | 102800 | 15 | 10 | 0.0178320063 | 120.14 |
| Yulara | Joint-patch | 42 | 140432 | 12 | 7 | 0.0314961706 | 96.32 |
| Yulara | Joint-patch | 43 | 140432 | 17 | 12 | 0.0275610553 | 138.18 |
| Yulara | Joint-patch | 44 | 140432 | 13 | 8 | 0.0315627834 | 103.31 |
| Yulara | Depthwise TCN | 42 | 682384 | 6 | 1 | 0.0460016082 | 47.93 |
| Yulara | Depthwise TCN | 43 | 682384 | 15 | 10 | 0.0280538075 | 126.44 |
| Yulara | Depthwise TCN | 44 | 682384 | 14 | 9 | 0.0212051758 | 117.30 |
| NIST Ground | Discrete recurrent | 42 | 96562 | 21 | 16 | 0.0162022513 | 1850.30 |
| NIST Ground | Discrete recurrent | 43 | 96562 | 16 | 11 | 0.0168051737 | 1404.92 |
| NIST Ground | Discrete recurrent | 44 | 96562 | 20 | 15 | 0.0160847985 | 2360.15 |
| NIST Ground | Inverted-variate | 42 | 102800 | 24 | 19 | 0.0179006374 | 171.11 |
| NIST Ground | Inverted-variate | 43 | 102800 | 21 | 16 | 0.0185443183 | 150.32 |
| NIST Ground | Inverted-variate | 44 | 102800 | 25 | 24 | 0.0179460971 | 179.37 |
| NIST Ground | Joint-patch | 42 | 140432 | 25 | 24 | 0.0174976746 | 179.48 |
| NIST Ground | Joint-patch | 43 | 140432 | 25 | 21 | 0.0175885667 | 178.59 |
| NIST Ground | Joint-patch | 44 | 140432 | 23 | 18 | 0.0172189594 | 164.64 |
| NIST Ground | Depthwise TCN | 42 | 682384 | 25 | 25 | 0.017714833 | 184.40 |
| NIST Ground | Depthwise TCN | 43 | 682384 | 25 | 25 | 0.0182565444 | 181.15 |
| NIST Ground | Depthwise TCN | 44 | 682384 | 25 | 25 | 0.0187940589 | 181.88 |

四个模型的不同实际轮数由统一 early stopping 规则产生。best epoch 不等于实际训练停止轮数；停止前若若干轮没有足够改善，保存的仍是更早的 Validation 最优状态。表中 Validation global MSE 是训练选择指标，不应与逆缩放后的 Test kW RMSE 混用。

## 16. Artifact、恢复机制与独立证据链

Alice 每个 run 保存 best_validation.pt、completed.json、test_H144.npz；外部保存 best_validation.pt、completed.json、test_predictions.npz、训练历史，以及两个场址的 Train-fitted 处理器和 test_release.json。大型文件留在本地忽略目录，Git 保存代码、聚合 CSV 和小型审计材料。

安全恢复不以“文件存在”作为成功依据，需检查 site/array、model、seed、输入维度、lookback、H144、split、shape、origins、target starts、mask 和配置身份；checkpoint 必须能够在容差内无梯度复现保存预测。不一致应报告 STALE_ARTIFACT，不能静默复用。

独立验证器从保存 artifact 和原始功率重建目标、Last-value、准确提前 24 小时的 Daily 与共同 mask，再独立计算误差、skill、计数、seed 均值/SD、排名和胜负。它不调用训练主脚本的指标函数，避免同一公式错误在两处被复用而得到假一致。

证据链是：原始来源及时间字段 → 冻结处理协议 → split/窗口与 mask → 已选 checkpoint 和保存预测 → 独立指标复算 → 正式 CSV → 程序生成论文表图。文件 size/mtime_ns 的前后比对证明本轮读取未改动受保护文件；它不是替代一切溯源的密码学证明。本项目没有新增原始文件 SHA 清单或复杂 contract。

| 检查层级 | 已完成结果 | 时间范围说明 |
|---|---|---|
| 原 Alice 独立指标核验 | 4,414/4,414 | 原正式证据阶段的冻结记录 |
| M3 重跑 M1-R 协议测试 | 43/43 | 无失败、错误、skip |
| M3 重跑 M2 普通测试 | 17/17 | 同上 |
| M3 重跑外部 artifact 测试 | 10/10 | 24 checkpoint 复现保存预测 |
| M3 独立外部数值比较 | 10,432/10,432 | 最大绝对差约 4.55e-13 |
| M3 论文比较检查 | 1,360/1,360 | 对照 CSV、包络、计数和均值/SD |
| M3 原 Alice 文件配套核验 | 36 完整组、108 文件 | 身份、形状和 17 通道 strict load |
| M3 文件保护 | 605 文件 size/mtime_ns 不变 | 497 外部/来源/冻结证据文件 + 108 Alice artifact |

这些数量不是独立科研样本数，也不能简单相加声称统计证据强度。M3 没有重新运行原 Alice 的全部历史数值核验，也没有访问原 Alice 原始数据所在 master；报告保留这一区别。本次报告编写引用既有检查结果，没有重新执行上述完整测试。

## 17. M3 论文组织与图表职责

最终英文标题：**Leakage-Aware Multi-Horizon Benchmarking of Compact Neural PV Forecasts Across Technologies and Sites**。

主文按研究问题、相关工作、研究设计与场址、模型/评价、Alice 结果、外部结果、跨设置解释、限制和结论展开。核心是两个层次的证据结构，不是把一张外部表简单附在旧论文末尾。

| 主图 | 论证职责 |
|---|---|
| Figure 1 | 显示三个地理场址、两套输入、各目标独立拟合，防止读者误以为池化训练 |
| Figure 2 | 展示 Alice 全部 24 条件的完整 RMSE ratio，保留 Daily 22/24 与两个例外 |
| Figure 3 | 展示焦点/预指定 Inverted-variate 对 Last-value 的五目标、四时域、两 scope skill |
| Figure 4 | 展示同样布局下对 Daily 的 skill，直观体现站点和 scope 条件性 |
| Figure 5 | 展示神经模型在三个 Alice 阵列及两个外部场址的排名变化 |

主表包含场址与输入/时间协议、Alice 主要端点、外部主模型全部 16 条件、神经排名以及分设置/选择身份的 matched 胜场。补充材料保留完整 Alice 表、外部质量、96 组支持、24-run、完整模型与 seed RMSE、mean/SD 和包络；全部其他逐 seed 指标仍由 CSV 提供，未将上万行明细直接排进 PDF。

语言区分 replication、validation、transfer 和 generalization，不使用 unsupported novel、state-of-the-art 或统计显著性表述。限制集中交代，同时摘要和结论突出有证据的记忆点。不因结果不利而删掉 NIST H144/full 或较小的 Yulara skill。

M3 最终摘要 205 词；主文 13 页、5 图、5 表；Supplement 29 页、3 图、18 表。42 页均渲染检查；字体嵌入、图为矢量，无缺失引用、未定义交叉引用、重复 label 或 overfull box。3 条非致命 REVTeX float placement fallback 提示已通过成品检查。全部 8 图和 23 表提供 alt text，投稿包 PDF 与主目录副本一致。

## 18. 研究结论的有效范围与限制

可以支持的结论是：在所冻结的历史信息、数据处理、训练预算与目标支持下，Inverted-variate 相对 Last-value 的改善在两个外部场址重现；Daily 的相对价值和四模型排序随设置、时域与 scope 而不同。匹配支持和明确历史信息使这种比较可解释。

当前不能支持以下外推：

- 三个地理场址足以覆盖所有气候或运行环境；
- 两个外部场址的 October–December 代表全年季节；
- Alice 三阵列差异就是模块技术的纯效果；
- 7 通道结果等于 17 通道 checkpoint 的迁移性能；
- 神经模型或 Daily 在所有设置普遍较好；
- 三 seed SD 代表未来年份/天气样本总体不确定性；
- 对紧凑项目实现的排序就是对官方完整架构的排序。

NIST 2017 完整运行日志未取得，Yulara 部分元数据尚不完整。二者已在 M1-R 依据明确回退规则列为非阻塞文档限制，但不是“没有异常”的证据。外部有限异常运行状态被保留，因此模型评价包含这些真实分布；未做基于 Test 误差的清洗。

未来若研究全年、多年、统一输入跨站、零样本迁移、概率预测或天气预报输入，需要独立设计与授权，不能把它们当成当前结果已经证明的内容，也不能为本次报告私自增加实验。

## 19. 投稿、作者与代码发布状态

当前为 **SCHEME_A_MULTISITE_MANUSCRIPT_READY_FOR_AUTHOR_REVIEW**，并非已经完成作者最终签核或实际投稿。现有稿件和投稿材料面向 JRSE；导师要求的中科院分区优先策略属于后续选刊决策，不在本报告中推定新的分区、影响因子或收费政策。

用户已确认过作者顺序、ORCID、贡献、Funding、无利益冲突、无需伦理审批和独家投稿意向；这些事实在中文签核表中与“完整 M3 新稿及最终措辞审批”分开。作者仍需审核正文、Supplement、最终声明、任何必要 AI 披露、Cover Letter、metadata、相关工作披露及投稿授权。本报告不补填未知 ORCID、AI 模型版本或许可证。

代码计划公开，但最终仓库 URL、公开范围与许可证仍需作者确定。原始 NIST/DKASC 数据从官方提供方获取并遵守条款，不声称代码仓库重新分发原始数据。尚未建立 GitHub Release、改变可见性、选择付费 OA 或实际投稿。

当前 M3 分支为 manuscript/clean-pv-benchmark-multisite-revision，Draft PR #19 的 base 为 research/scheme-a-multisite-frozen-training。本报告新增为说明性文件，不修改任何实验协议、论文结果或 artifact。

## 20. 复核导航与资料索引

本报告中的数值应优先沿以下正式文件核对。路径均相对于本报告所在目录，便于移动整个项目后继续阅读。

| 文件 | 作用 |
|---|---|
| [原 Alice config](../../GFNODE_experiments/scheme_A_submission_correction/config.json) | 日期、17 通道、模型和训练参数 |
| [原 Alice 实现](../../GFNODE_experiments/scheme_A_submission_correction/run_corrected_benchmark.py) | 原正式处理、窗口、训练与评价实现 |
| [原 Alice 指标](../../GFNODE_experiments/scheme_A_submission_correction/corrected_metrics.csv) | 原冻结数值事实 |
| [原 Alice 报告](../../GFNODE_experiments/scheme_A_submission_correction/REPORT.md) | 原公平修正、样本计数、Daily 与效率审计 |
| [原独立审计](../../GFNODE_experiments/scheme_A_submission_correction/INDEPENDENT_EVIDENCE_AUDIT.json) | 原 4,414 比较证据 |
| [外部冻结 config](../../GFNODE_experiments/scheme_A_multisite_extension/multisite_config.json) | 七通道、时间、split、24-run 与模型配置 |
| [外部协议](../../GFNODE_experiments/scheme_A_multisite_extension/MULTISITE_PROTOCOL.md) | 时间语义、有效性、匹配与分析边界 |
| [外部数据审计](../../GFNODE_experiments/scheme_A_multisite_extension/DATA_AUDIT_SUMMARY.csv) | 质量及 96 组支持记录 |
| [M2 主脚本](../../GFNODE_experiments/scheme_A_multisite_extension/run_multisite_benchmark.py) | 冻结训练与预测实现，仅供追溯 |
| [M2 独立复算](../../GFNODE_experiments/scheme_A_multisite_extension/independent_verify_multisite.py) | 与训练指标函数分离的复核路径 |
| [每 seed 指标](../../GFNODE_experiments/scheme_A_multisite_extension/metrics_per_seed.csv) | 外部逐 run、horizon、scope 全指标 |
| [mean/SD 指标](../../GFNODE_experiments/scheme_A_multisite_extension/metrics_summary_mean_sd.csv) | 外部完整汇总 |
| [M2 训练报告](../../GFNODE_experiments/scheme_A_multisite_extension/MULTISITE_TRAINING_REPORT.md) | 本报告结果表和运行表来源 |
| [M3 比较 CSV](multisite_manuscript_comparison.csv) | 程序生成的 400 行跨设置比较 |
| [M3 证据审计](M3_EVIDENCE_AUDIT.json) | 最新证据和 PDF 检查记录 |
| [论文审核记录](REVIEW.md) | M3 科学边界、版式与待确认项 |
| [中文签核表](submission_package/AUTHOR_SIGNOFF_CHECKLIST.md) | 作者确认状态 |
| [主文](main.pdf) / [补充材料](supplementary.pdf) | 当前 M3 审阅版本 |

提供方来源在冻结协议和论文参考文献中记录，包括 [Yulara 系统页](https://dkasolarcentre.com.au/source/yulara/yulara-3-roof-sails-in-the-desert-2)、[NIST 数据论文](https://doi.org/10.6028/jres.122.040)、[NIST 数据 DOI](https://doi.org/10.18434/M3S67G)、[NIST Data Dictionary](https://www.nist.gov/document/datadictionarysupplementalcontentpdf)。这是既有来源记录，访问核查日期为 M3 的 2026-09-07；本次报告没有将其当作重新在线审核的结果。

## 21. 导师审核时建议重点核对的问题

1. 两层证据与论文定位是否准确反映研究价值，是否接受条件性 benchmark 而非算法创新的定位。
2. 是否接受固定保守 availability time，以及对未知元数据的限定解释。
3. 是否认同对 Last-value 与 Daily 的信息策略区分、相同目标 mask 和不合并跨设置排名的口径。
4. 主模型预先指定、三 seed 均值/SD、post hoc envelope 是否清晰分开。
5. NIST 排名变化、唯一 Daily 胜项和 Yulara 长时域小幅 skill 是否得到适度解释。
6. 正文和补充材料的详细程度、作者声明及代码公开方案是否可以进入最终投稿准备。

以上问题属于科学与稿件审核，不是要求为获得更好胜场而追加训练。当前已有实验结果全部保留。

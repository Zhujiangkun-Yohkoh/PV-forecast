# PV预测路线关闭与稿件挽救计划

## 执行性结论

当前GFNODE稿件**不能以clean结果直接投Applied Soft Computing（ASOC）**。这不是一次补表或小修可以解决的问题：原始Tables 9–13和Figures 10–14缺乏可接受的来源链，旧数据/训练协议无效，原ODE实现不使用时间`t`，而修正后的clean GFNODE又没有稳定性能优势。此后对离散decoder、跨技术共享、高频动态、概率区间、联合事件、梯度冲突、冻结事件头和ALICD的真实筛选，也均未形成可投稿的新算法。

当前唯一可靠的预测核心是clean protocol下的Independent ModernTCN / `TRAJECTORY_ONLY`。它是可信工程基线，但ModernTCN是既有算法，不能包装为本项目原创方法。

**唯一推荐：停止GPU训练和模型迭代，选择Route B，将工作重构为严谨的公开PV数据、无泄漏协议和确定性多步预测应用/benchmark论文；先增加必要baseline与未使用时间段确认，再选择匹配的应用/太阳能期刊。** 如果导师坚持ASOC，则必须选择Route C并把它视为新项目，而不是GFNODE revision。

## 1. 证据边界

### 1.1 原稿来源

主张定位以实际EPSR提交版为准：

- `PV_improve_v1/Submision/EPSR/UT_PV_Forecasting_GFNODE_Manuscript.docx`
- 对应PDF、Cover Letter、Highlights、Title Page、declaration statement及Figures目录均存在。
- EPSR正式拒稿邮件/decision letter未在可访问工作区中定位到；“lack of sufficient novelty”来自项目背景记录，不能伪装为本地已归档文件证据。建议投稿前由作者人工补存正式decision letter。
- `GFNODE_Revision_Plan.md`是内部改进计划，不是期刊审稿意见或clean证据；其中对架构价值和旧结果的乐观评价已被后续真实实验取代。

### 1.2 旧稿不可恢复的核心

Stage 0确认：KNN/Isolation Forest/scaler在split前全数据拟合；窗口构造后split导致重叠；Test参与early stopping、学习率或checkpoint选择；删除夜间后数组行不再等距，却继续把行数解释为5分钟物理时间。Tables 9–13没有单一可复算来源，且均为单seed、无SD/CI。

ODE方面，状态为`[B,128]`，`ODEFunc.forward(t,z)`没有使用`t`；输出网格是归一化`[0,1]`，不同Horizon分别训练；所谓0.1不是实际物理步；Figure 12使用随机128维状态和随机起点，且绘图向量场遗漏残差项。故连续时间、任意分辨率、误差累积抑制和向量场收敛主张均失效。

## 2. 可靠资产清单

### 2.1 数据资产

| 资产 | 当前状态 | 可用于主结果？ | 使用限制 |
|---|---|---|---|
| 权威DKASC 5分钟PV下载 | 可用；四阵列文件已做结构审计 | 是，需重新定义新论文数据版本 | 明确保留malformed、缺失与单位元数据；不得回用旧全数据预处理 |
| 2018 Sanyo/Hanwha/Qcells项目CSV | 时间戳与规则网格可用 | 有条件 | 只能称“best available project source layer”，不能称原始未加工下载；原始missingness可能已丢失 |
| 2022 Site17 Sanyo PV | 可用 | 是 | 使用UTC/ACST约定、明确容量6.3 kW及字段来源 |
| 2022秒级MB0/MB1/MB2辐照 | UTC结构完整，按ACST UTC+09:30映射 | 是，带独立通道mask | 严格共同完整区段是分段的；不得直接使用文件Local字段 |
| 旧损坏2023高频文件 | 不可用于连续研究 | 否 | 只能作为数据损坏审计记录 |
| 2023或其他未使用PV/辐照 | 项目中部分文件存在，但尚未形成确认协议 | 尚否 | 可作为未来独立时间确认集，必须先做与2022同等的只读完整性/对齐审计 |

### 2.2 代码资产

可直接复用：clean timestamp reindex、Train-only KNN/Isolation Forest/scaler、feature/target分离缩放、split内连续窗口、Validation-only early stopping、ModernTCN标准实现、H144前缀或H12直接评价、多horizon指标、prediction/label/timestamp/mask保存、checkpoint恢复、seed控制和普通协议测试。

需要保持的原则：先按原始时间戳split，再fit预处理器和构造split内窗口；Test不进入训练接口；每个模型使用相同origins/labels；保存逐seedartifact；不把夜间删除后的行号当物理时间。

不得复用：旧`gfnode_solo_benchmark.py`的数据管线、旧fair comparison的Test选参逻辑、旧seasonal retraining脚本作为“无需重训练”证据、Figure 12随机向量场生成逻辑，以及任何不能关联clean artifact的表图生成脚本。

### 2.3 结果资产分级

**可用于未来论文主结果（在新论文完整设计中）：**

- `TRAJECTORY_ONLY / MEAN_ONLY ModernTCN`的三seed clean结果与保存artifact；
- discrete viability中ModernTCN/iTransformer/PatchTST的clean benchmark结果，但正式稿需核实全部配置、样本一致性，并补足论文所需基线；
- Train-only clean protocol的数据量、split和普通协议测试结果。

**只能用于补充材料或Discussion：**

- HF_DYNAMICS在H12三seed均失败，说明过去秒级波动未稳定提供未来方向信息；
- ramp-step风险相对简单因果基线的探索性可预测性；
-轨迹误差解剖：高变化日间过度平滑、H9–H12误差贡献、历史信息可识别困难度；
- ALICD、共享跨技术、概率宽度调制等规范负结果，若期刊允许negative/supplementary evidence且篇幅合适。

**只能作为内部开发记录：**

- Clean GFNODE、Discrete Candidate、Shared–Private、联合Hazard、梯度投影、Frozen Hazard和ALICD的路线选择过程；
- 这些结果可证明项目做过严谨决策，但不应把一连串失败模块堆成论文“贡献”。

**完全不得用于投稿：**

- 原Tables 9–13全部数字；Figures 10–14全部结论性用法；
- “9/12 R²最优”“10/12 RMSE最优”“3.6%–27.2%下降”“within one standard deviation”；
- “seasonal transfer without retraining”“96.5%–97.6% retention”；
- Figure 12的latent convergence；
- arbitrary temporal resolution、continuous-time temporal consistency、error-accumulation suppression；
- 132.77M FLOPs、106.37ms及deployment-ready表述，除非重新做有环境清单的测量。

完整逐项处置见`CLAIM_DISPOSITION_MATRIX.csv`。

## 3. 三条稿件路线

### Route A：直接修改GFNODE后投ASOC

**判定：NO-GO。**

- 科学可行性：不足。原方法实现与论文描述不一致；修正后的time-conditioned GFNODE在Sanyo H144比离散对照差9.42%，跨数据集没有稳定优势。
- 创新性：不足。MSD-TCN、Transformer、gated fusion、BiLSTM aggregator和Neural ODE均是普通模块组合；核心ODE价值没有clean性能或特定能力支持。
- 失效结论：连续时间、任意分辨率、长horizon、误差累积、seasonal transfer、cross-technology generalization、SOTA排名、向量场稳定和deployment全部需要删除。
- 实验工作量：不是“补跑”；必须重建全部Tables/Figures、完整baseline、公平消融、独立时间/站点验证和计算环境测量。即使完成，已有clean ODE结果仍不支持方法主线。
- 与旧稿关系：若强行保留GFNODE名称与方法，科学论点为空；若删除ODE和旧结果，实质已是新论文。因此不能称为普通revision。

### Route B：基于TRAJECTORY_ONLY形成干净应用型论文

**判定：最快的可行方向，但当前材料尚不足以立即投稿。**

可定位为：公开DKASC数据上的无泄漏PV多步轨迹预测复现/benchmark/application study，重点是时间协议、源数据恢复、跨时间评价、误差分解和对应用边界的诚实说明，而不是新网络结构。

可能贡献：

1. 可复算的timestamp-level clean protocol和普通防泄漏测试；
2. 公开PV与秒级辐照的权威时间对齐、missing mask和公平样本定义；
3. 多种标准forecasting模型在同一协议下的三seed比较；
4. 高变化场景、lead-time和轨迹过度平滑的系统误差分析；
5. 未使用时间段上的独立确认，以及负结果说明过去高频统计的边界。

限制：ModernTCN必须明确引用为既有算法；“采用ModernTCN”不是算法创新。稿件标题、摘要和贡献不得使用“novel architecture”。

最低新增工作：

- 在**未用于设计的时间段**建立独立确认集；优先完成2023或后续PV/气象完整性与时间映射审计；
- 用固定配置复跑必要基线。最低集合建议：persistence/seasonal persistence、线性直接多步模型、PatchTST、iTransformer、ModernTCN；若投稿期刊要求，再加入一个现代卷积/频域模型。所有模型三seed或更多，Validation-only选择；
- 报告forecast skill、mean±SD、daylight与完整时间轴、样本数、参数/延迟环境；
- 预先声明2022 Test已被反复查看，只能作为开发期探索集；
- 重建所有图表，不继承原Table/Figure编号或数字。

可复用原稿：PV预测背景、DKASC站点介绍和评价指标定义可大幅重写后使用；原方法、结果、Discussion、Abstract、Highlights和Conclusion不能直接沿用。

### Route C：重新开展ASOC新项目

**判定：科学上可以，但必须立项为新项目。**

当前误差证据指向输入信息限制：历史秒级波动可以说明“当前环境不稳定”，却不能观察forecast origin之后的云层运动方向；MSE下模型趋向条件均值和轨迹平滑。ALICD的精确投影也未改善高变化Validation误差，说明继续改变decoder不能凭空创造未来信息。

真正需要的资源，而不是另一个模块：

| 资源 | 当前状态 | 新项目作用 |
|---|---|---|
| 可在预测时获得的NWP forecast | 当前clean主实验没有 | 为1–12h提供未来外生驱动；需下载并按发布时刻/lead time防泄漏匹配 |
| 卫星云图 | 当前没有 | 提供上游云场与运动信息；需外部数据源、空间配准和可用时延协议 |
| 天空图像 | 当前没有 | 提供分钟级局部云演化；需要站点同步硬件/公开数据 |
| 云运动矢量 | 当前没有 | 从图像或卫星提取方向/速度；依赖可验证的过去帧和实时可用性 |
| 多站点空间测量 | 2018同站三阵列不是多站点 | 提供云团传播信息；需重新下载或外部匹配 |
| 条件生成式概率轨迹协议 | 当前只有失败的三分位NCQ筛选 | 可处理多模态未来，但需独立校准集、proper scores、可靠性和强概率baseline |
| 2022权威PV+秒级辐照 | 已拥有 | 可做基础时间对齐与历史观测输入，但不能替代未来外生信息 |

ASOC官方scope涵盖神经网络、混合方法、power and energy与time-series prediction，但强调高质量soft-computing research和真实问题中的方法推进。当前稿件缺少的是：明确的新软计算机制、由缺失未来信息驱动的合理问题定义、强baseline、clean多seed、独立时空验证、规范消融/统计与可复现来源链。[Applied Soft Computing官方Aims & Scope](https://www.sciencedirect.com/journal/applied-soft-computing)；[官方Guide for Authors](https://www.sciencedirect.com/journal/applied-soft-computing/publish/guide-for-authors)。

## 4. 期刊路线核查（2026-08-25官方页面）

### 快速挽救候选方向

1. **Solar Energy Advances**：官方scope明确包含solar energy data analytics、AI和forecasting，适合严谨应用/数据协议/benchmark定位；当前为开放获取，具体APC、索引和作者要求须在提交前由作者人工复核。[官方期刊页面](https://www.sciencedirect.com/journal/solar-energy-advances)
2. **Journal of Renewable and Sustainable Energy**：AIP官方将其定位为覆盖renewable/sustainable energy物理与工程、solar energy及energy meteorology的跨学科期刊。是否接受纯benchmark/application稿仍取决于新时间确认和工程贡献，不能承诺送审或接收。[AIP官方页面](https://publishing.aip.org/publications/journals/special-topics/rse/)
3. **IET Renewable Power Generation**：官方scope涵盖PV、renewable power forecasting、模型验证与实际电力系统相关研究，但也说明多数论文需显著方法或应用新颖性。若稿件只有单站算法排行而无系统意义，仍有desk-reject风险。[IET官方Aims & Scope](https://ietresearch.onlinelibrary.wiley.com/hub/journal/17521424/homepage/productinformation.html)

不推荐仅凭网页显示的审稿时长选择期刊，也不使用第三方分区网站作为依据。Clarivate/Scopus实时收录状态和学校分区政策应在正式选刊时由图书馆或官方数据库人工确认。

### 新ASOC项目方向

只有在获得预测时可用的未来外生信息或真正空间/多模态数据、重新定义问题并建立独立确认后，才值得重新评估ASOC。那时需要的不是恢复GFNODE数字，而是从零形成方法假设、数据可用性协议、强baseline、消融、统计和外部验证。

## 5. 原表图处理清单

| 项目 | 处置 | 原因 |
|---|---|---|
| Table9 / Figure10 | 永久退出；未来按clean protocol重建新表 | 数字与artifact不符；Test作validation；预处理泄漏 |
| Table10 / Figure11 | 永久退出 | 仅找到H48且不匹配；无H144来源；不能证明模块贡献 |
| Table11 | 永久退出 | 没有一个run复算七horizon；旧小时语义无效 |
| Figure12 | 永久退出，不重画为同类证据 | 随机状态/随机起点、错误向量场、非held-out latent dynamics |
| Table12 / Figure13 | 永久退出 | autumn重新训练且数字不匹配；“without retraining”被直接反证 |
| Table13 / Figure14 | 永久退出 | Test选参、无统一GFNODE结果、无SD；SOTA排名无效 |

## 6. 审稿人式最终判断

1. 当前GFNODE不能凭clean结果直接投ASOC。
2. 没有可辩护的ODE核心贡献：正确time-conditioned实验没有跨数据集稳定优势。
3. ModernTCN结果足以作为可靠基线，不足以成为新算法论文。
4. ALICD失败且进一步降低总变差比，继续修改decoder缺乏实证依据。
5. 当前主要限制是forecast origin之后的未来外生信息缺失，其次才是模型；clean训练协议本身已经可用。
6. 原Tables 9–13、Figures 10–14及其所有性能、季节、连续时间和部署主张必须永久退出投稿材料。
7. 最快稿件定位是公开PV数据上的clean deterministic multi-step benchmark/application paper。
8. 坚持ASOC必须视为新项目，因为需要新数据、新问题、新方法和全套新实验，旧稿核心不可保留。
9. 当前不应继续GPU训练；先由导师选择稿件路线并锁定未使用确认数据和期刊定位。
10. 唯一推荐：选Route B，放弃ASOC目标，完成最小必要baseline与独立时间确认后投稿更匹配的应用/太阳能期刊。

## 7. 下一步（不在本轮执行）

1. 导师确认Route B和目标论文类型；不再讨论模型v2/v3。
2. 对未使用时间段完成只读完整性/对齐审计并冻结评价角色。
3. 在运行前写定最小baseline、seed、指标和图表清单。
4. 仅执行缺失的公平实验，重建全新论文结构和图表。
5. 投稿前由导师/图书馆复核目标期刊最新scope、索引、费用与格式。


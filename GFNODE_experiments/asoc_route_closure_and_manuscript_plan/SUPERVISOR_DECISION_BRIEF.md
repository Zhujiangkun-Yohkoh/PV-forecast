# 给导师的决策简报：停止模型迭代，选择稿件路线

## 1. 原稿状态

EPSR对稿件的项目记录结论为“lack of sufficient novelty”；工作区中未发现正式decision letter文件，建议作者补存原信。更关键的是，此后Stage 0只读审计发现原稿不仅有创新性问题，还有会使性能评价失效的协议问题：预处理器在全数据拟合、滑窗后split导致时间重叠、Test参与选择、删除夜间后仍把数组行当固定5分钟，以及Tables 9–13无法从保存结果复算。

原Neural ODE实现的`t`未进入向量场，网格是归一化`[0,1]`，不同Horizon独立训练。Figure 12来自随机128维状态而不是held-out latent trajectory。因此“连续时间”“任意时间分辨率”“抑制误差累积”“稳定向量场”等核心表述不成立。

项目随后完成了严格clean protocol和多轮真实GPU筛选：

- 修正的time-conditioned GFNODE没有稳定胜过参数匹配离散decoder：Qcells H144有优势，但Sanyo H144恶化9.42%，ODE路线FAIL。
- Discrete Candidate对三种强baseline平均RMSE排名2.667，8/12组合落后最佳模型超过5%，不具ASOC竞争力。
- 同站三阵列Shared–Private和Early-Fusion在12/12组合均输给Independent ModernTCN；当前实现NO-GO。
- 过去秒级辐照动态在H12三seed全部恶化；它能识别困难状态，但不能提供未来云运动方向。
- Ramp head能预测ramp step，但风险调制没有改善Pinball/Winkler；联合Hazard虽改善事件识别，却使H12功率RMSE恶化3.45%。
- 非对称梯度投影仅恢复极少精度；Frozen Hazard不能保留事件优势；联合轨迹—事件路线终止。
- ALICD投影精确满足锚点，但Validation高变化日间RMSE仍恶化，总变差比从0.437降到0.425，过度平滑更严重。

因此当前ASOC结论是：**旧稿不能修补后直接投稿，且现有任何新候选都没有形成算法主线。**

## 2. 为什么继续调模型不合理

这不是“再加一个模块”能解决的问题。

1. ODE已由参数匹配实验否定为稳定中心优势，继续改solver、步长或激活只会形成v2/v3试错。
2. 三阵列共享输入包含相同天气，但联合模型仍全面落后；共享容量膨胀没有转化为收益。
3. 高频历史波动告诉模型“当前可能困难”，却不知道forecast origin之后云往哪移动。
4. Ramp事件任务与功率轨迹共享训练存在可测的梯度冲突，但投影后功率损失仍基本保留，说明冲突不是唯一根因。
5. ALICD的固定数学投影实现正确，仍未改善目标场景，进一步支持“缺未来信息而不是decoder不够复杂”。
6. MSE面对多种可能未来时会趋向条件均值，表现为轨迹平滑。没有NWP、云图、天空图像或空间上游测量时，结构改造不能恢复未观测的未来方向。

继续GPU训练的边际价值已经很低，并会扩大多重尝试与Test反复查看问题。现在应该关闭模型路线。

## 3. 两个导师决策选项

### 选项一：最快完成可投稿版本

**放弃ASOC；把稿件重构为clean deterministic PV trajectory benchmark/application paper。**

可复用：权威DKASC数据、2022 UTC→ACST对齐、Train-only预处理、split内窗口、ModernTCN及三seedartifact、多horizon指标、协议测试和误差解剖。

必须新增：一个真正未用于方法设计的后续时间确认集；persistence、线性、PatchTST、iTransformer、ModernTCN等最小公平baseline；forecast skill、mean±SD、参数/延迟环境；全新表图。ModernTCN必须作为既有算法，贡献放在数据协议、benchmark、时间确认和误差分析。

工作量：中等、可控。不是零实验，但不再发明模型；只补齐必要比较和独立确认。

主要风险：若只有单站/单年和标准算法排行，应用贡献仍可能不足；必须用严格数据协议、独立时间验证和可复现性形成价值。

期刊匹配：优先人工核查Solar Energy Advances、Journal of Renewable and Sustainable Energy或IET Renewable Power Generation的最新scope/费用/索引。它们官方scope包含solar/PV、energy engineering或forecasting，但不保证送审或接收。

与原稿关系：背景、数据介绍和评价方法可重写复用；方法、摘要、结果、Tables 9–13、Figures 10–14、Discussion和Conclusion必须新写。它是“数据与应用研究重构”，不是GFNODE revision。

### 选项二：坚持ASOC

**把它立为新项目，不再把旧稿当基础。**

先获得预测时真正可用的未来外生信息：NWP forecast、卫星/天空云图、云运动矢量或多站点空间测量；或者建立严格的条件生成式概率轨迹协议。然后重新定义科学问题、方法和实验。

可复用：clean data engineering、时间对齐、artifact/测试框架、ModernTCN作为baseline、失败实验形成的边界知识。

新工作量：高。包括数据许可/下载、时间与发布时延匹配、多模态或空间协议、新方法、强baseline、消融、统计、跨时间/跨站验证和论文全部重写。

主要风险：数据可能无法获得或实时语义不明确；新方法仍可能没有优势；周期明显长于稿件挽救。

科学竞争力：如果未来信息真实可用、方法针对明确不确定性并有外部验证，才可能达到ASOC对soft-computing方法和实验完整性的要求。

与原稿关系：仅能复用研究背景和工程框架；不能沿用GFNODE结果或方法主张。它是新项目。

## 4. 推荐意见

**唯一推荐：选择选项一。**

理由不是它“最容易发”，而是它与当前可靠证据一致，也最符合“最快形成最终可投稿版本”的目标。停止所有新模型和GPU训练，先锁定未使用时间确认集与最小baseline清单，再重构为诚实、可复算的应用/benchmark论文。

不建议继续尝试ODE、decoder、事件head、损失权重、共享—私有或梯度算法。若导师明确要求ASOC，则应正式关闭本稿并另立选项二项目，避免继续以“revision”名义消耗时间。

### 需要导师现在确认的唯一事项

批准将目标从“GFNODE投ASOC”切换为“clean deterministic PV forecasting应用/benchmark论文”，并允许下一阶段只做未使用时间确认和最小必要baseline，不再进行模型创新试错。


# P02 Qcells 共同起报选择机制

代码与冻结Test的时间戳、标签、掩码逐元素一致，没有发现索引错误。主要机制是原Alice对有限负功率标签的排除，随完整12h窗口扩展而影响大量相互重叠的起报。不能把该子集称为天然更稳健或更有代表性。

每个split的规则5分钟网格是候选全集。重叠标记：边界/历史不足、未来非有限/缺失、未来负值、origin无效、Daily不可用。互斥顺序固定为边界→未来非有限→未来负值→origin无效→保留。Daily是另外的逐目标交集，不用于解释primary起报筛除。`qcells_origin_flags.csv`保留全部标记，`qcells_reasons.csv`同时给出重叠和互斥数量。

Test从H12的6463个起报到H144的2996个，移除3467：117边界，3350未来负标签；重叠负标签标记3415，其中65已先归边界。未来非有限排除为0。54个有限负原记录（52在18点、2在7点）可被许多长窗口反复包含。数值从−0.010933 kW到接近零；没有官方依据将其解释为坏码或具体运行事件。

H12有77556起报—lead对，active36504（47.0679%）。共同H144的前1h有35952对，active42（0.116822%）；仅涉及25个active origins与25个唯一active物理时间戳。原H12有6760个唯一目标时间戳，共同前缀3260。被移除子集前1h的active占87.6406%，因此原本高输出起报被强烈筛走。

split | support | origins | valid_target_pairs | power_active_pairs | power_active_fraction | unique_physical_targets | unique_active_physical_targets | active_origins
--- | --- | --- | --- | --- | --- | --- | --- | ---
train | H12_support | 27951 | 335412 | 150395 | 0.448388847 | 29282 | 12959 | 13083
train | common_H144 | 12648 | 151776 | 4674 | 0.0307953827 | 13748 | 453 | 486
validation | H12_support | 6194 | 74328 | 33418 | 0.449601765 | 6480 | 2864 | 2921
validation | common_H144 | 2977 | 35724 | 0 | 0 | 3230 | 0 | 0
test | H12_support | 6463 | 77556 | 36504 | 0.470679251 | 6760 | 3133 | 3173
test | common_H144 | 2996 | 35952 | 42 | 0.0011682243 | 3260 | 25 | 25

Train同样从44.84% active变为共同前1h约3.08%；Validation约44.96%变为0。各split的1h和12h、保留与剔除分布，小时、月份、唯一目标数完整见support/hours/months/power_histogram CSV。原记录见negative_records。不是只在Test偶然出现的现象。

历史Alice神经训练要求完整H144未来标签，但不要求origin有效；其Train12747、Validation2999，对照评估/Ridge的12648、2977。两者差异来自既有origin规则；本轮未默默改成同一训练样本，也未重训。完整窗口要求本身已使训练的起报时刻分布受该负值规则影响。共同起报固定lead曲线只能描述这一特定选择子集，不能解释为无条件的误差随lead增长。

共同1h Last-value约0.003983、Inverted平均0.124942 kW是上一轮保存逐点归约结果，本轮诊断重新核对其支持机制；未重新加载60个checkpoint复现此预测。原主分析未被替换。下一轮若修改Alice负值语义，需先取得物理/提供方依据并明确新协议；可能涉及训练样本变化，本轮不启动。

# Scheme A有限收尾报告

## 最终状态：B——收窄新增核心范围后，可进入投稿准备

核心证据链已完成：参考历史信息、Alice已验证模型的目标支持敏感性、两外部场址相同支持规则下的跨时期四格对照。Sanyo新增支持不报告完整三seed Inverted均值；TCN/Recurrent不作为新支持核心排名。历史四模型结果保持原证据身份。该收窄不破坏论文完整性，也不要求次要模型逐点一致才推进准备。作者签核、期刊格式与公开范围仍待对应决策；没有实际投稿或PR描述对外更新。

## R1：容差与科学影响分开

36组仍为16通过/20未通过原容差。最大单点差1.286983 W，最大预测差RMS 0.020312 W；已评分RMSE变化最大0.000566719 W，没有改变这些原支持下的排名或Daily效应方向。全部逐组、范围、单位和界限在ALICE_REPLAY_IMPACT_REPORT.md及CSV；有限代表GPU探针未恢复历史数值状态，不把CPU当真值，也未放宽容差。

## R2：已验证Alice支持比较

主表为12h/full，效应=方法RMSE−Daily RMSE，负数有利于方法。48h块、2000次、同抽样；1h、active/low、24/72h与新增起报分别保存。

| site | support | method | rmse | reference_rmse | effect_kW | ci_low_kW | ci_high_kW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hanwha | Original | Original B | 0.248024 | 0.203069 | 0.0449548 | -0.010056 | 0.149575 |
| Hanwha | Original | Expanded B | 0.248024 | 0.203069 | 0.0449548 | -0.010056 | 0.149575 |
| Hanwha | Original | Inverted mean | 0.805442 | 0.203069 | 0.602373 | 0.515016 | 0.769413 |
| Hanwha | S1 | Original B | 0.2296 | 0.18377 | 0.0458306 | -0.0144802 | 0.139928 |
| Hanwha | S1 | Expanded B | 0.2296 | 0.18377 | 0.0458306 | -0.0144802 | 0.139928 |
| Hanwha | S1 | Inverted mean | 0.769427 | 0.18377 | 0.585657 | 0.486591 | 0.736334 |
| Hanwha | S2 | Original B | 0.229486 | 0.183647 | 0.0458393 | -0.0144335 | 0.139887 |
| Hanwha | S2 | Expanded B | 0.229486 | 0.183647 | 0.0458393 | -0.0144335 | 0.139887 |
| Hanwha | S2 | Inverted mean | 0.769363 | 0.183647 | 0.585716 | 0.486711 | 0.736412 |
| Qcells | Original | Original B | 0.23652 | 0.243616 | -0.00709587 | -0.0524758 | 0.0494194 |
| Qcells | Original | Expanded B | 0.259459 | 0.243616 | 0.0158429 | -0.0281455 | 0.0678084 |
| Qcells | Original | Inverted mean | 0.843614 | 0.243616 | 0.599998 | 0.493794 | 0.722665 |
| Qcells | S1 | Original B | 0.297615 | 0.274896 | 0.0227195 | -0.000907266 | 0.0664748 |
| Qcells | S1 | Expanded B | 0.29303 | 0.274896 | 0.018134 | -0.0212591 | 0.0733109 |
| Qcells | S1 | Inverted mean | 1.07967 | 0.274896 | 0.804772 | 0.653993 | 0.985737 |
| Qcells | S2 | Original B | 0.297542 | 0.274794 | 0.0227481 | -0.000860946 | 0.0665005 |
| Qcells | S2 | Expanded B | 0.292992 | 0.274794 | 0.018198 | -0.0211776 | 0.0734033 |
| Qcells | S2 | Inverted mean | 1.07968 | 0.274794 | 0.804883 | 0.654084 | 0.985658 |

Qcells的Inverted误差在S1增大，Hanwha略降，模型权重没有变化。二者相对Daily和Ridge B更高、相对Last-value更低的方向保留；新增起报子集独立报告。Ridge原/扩展可在某些支持上换序，不能将原支持上的Qcells扩展恶化推广到全部恢复支持。S1与S2只改变可评分目标，未改历史输入和Daily/Last-value的非负规则。完整目标对、唯一目标时间戳数（不是独立观测数）及active比例见support CSV。新增起报按每个horizon自己的原支持补集定义。

训练支持仍有选择偏移。本次前向回答“冻结模型在更广评分总体上如何比较”，不回答“重新训练后会怎样”。当前主张不需要启动重训；若未来要研究训练选择效应，最小另立实验可限于Qcells Inverted三个seed、同输入/日期/预算、逐目标masked训练与历史整窗规则的对照。这不是本轮完成事项，也不启动全部60次重训。

## R3：相同规则的四格对照

| site | period | support | method | rmse | reference_rmse | effect_kW | ci_low_kW | ci_high_kW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YULARA_COMBINED | 2017 | complete_H144 | Original B | 35.8431 | 16.1274 | 19.7157 | -2.73802 | 35.445 |
| YULARA_COMBINED | 2017 | complete_H144 | Expanded B | 15.8556 | 16.1274 | -0.271768 | -1.84505 | 1.67395 |
| YULARA_COMBINED | 2017 | complete_H144 | Inverted mean | 15.8275 | 16.1274 | -0.299933 | -1.77301 | 1.28872 |
| YULARA_COMBINED | 2017 | pointwise_finite | Original B | 35.8431 | 16.1274 | 19.7157 | -2.73802 | 35.445 |
| YULARA_COMBINED | 2017 | pointwise_finite | Expanded B | 15.8556 | 16.1274 | -0.271768 | -1.84505 | 1.67395 |
| YULARA_COMBINED | 2017 | pointwise_finite | Inverted mean | 15.8275 | 16.1274 | -0.299933 | -1.77301 | 1.28872 |
| YULARA_COMBINED | 2018 | complete_H144 | Original B | 8.01757 | 9.55316 | -1.53559 | -1.92368 | -1.12289 |
| YULARA_COMBINED | 2018 | complete_H144 | Expanded B | 10.8307 | 9.55316 | 1.27749 | 0.421777 | 2.26781 |
| YULARA_COMBINED | 2018 | complete_H144 | Inverted mean | 13.8662 | 9.55316 | 4.31302 | 2.94042 | 5.6797 |
| YULARA_COMBINED | 2018 | pointwise_finite | Original B | 8.01881 | 9.5597 | -1.5409 | -1.92481 | -1.12965 |
| YULARA_COMBINED | 2018 | pointwise_finite | Expanded B | 10.8351 | 9.5597 | 1.27536 | 0.416224 | 2.26544 |
| YULARA_COMBINED | 2018 | pointwise_finite | Inverted mean | 13.8591 | 9.5597 | 4.29935 | 2.92882 | 5.65394 |
| NIST_GROUND | 2017 | complete_H144 | Original B | 34.9991 | 41.8485 | -6.84938 | -8.36551 | -5.24437 |
| NIST_GROUND | 2017 | complete_H144 | Expanded B | 34.9991 | 41.8485 | -6.84938 | -8.36551 | -5.24437 |
| NIST_GROUND | 2017 | complete_H144 | Inverted mean | 44.5461 | 41.8485 | 2.69769 | -1.02695 | 6.47028 |
| NIST_GROUND | 2017 | pointwise_finite | Original B | 35.4064 | 42.2102 | -6.80378 | -8.23747 | -5.33935 |
| NIST_GROUND | 2017 | pointwise_finite | Expanded B | 35.4064 | 42.2102 | -6.80378 | -8.23747 | -5.33935 |
| NIST_GROUND | 2017 | pointwise_finite | Inverted mean | 44.7518 | 42.2102 | 2.54164 | -0.995964 | 6.1046 |
| NIST_GROUND | 2018 | complete_H144 | Original B | 38.6783 | 49.0796 | -10.4013 | -12.3068 | -8.44133 |
| NIST_GROUND | 2018 | complete_H144 | Expanded B | 38.6783 | 49.0796 | -10.4013 | -12.3068 | -8.44133 |
| NIST_GROUND | 2018 | complete_H144 | Inverted mean | 36.6306 | 49.0796 | -12.449 | -17.5695 | -7.0116 |
| NIST_GROUND | 2018 | pointwise_finite | Original B | 40.777 | 51.0279 | -10.2509 | -11.9317 | -8.40567 |
| NIST_GROUND | 2018 | pointwise_finite | Expanded B | 40.777 | 51.0279 | -10.2509 | -11.9317 | -8.40567 |
| NIST_GROUND | 2018 | pointwise_finite | Inverted mean | 38.1861 | 51.0279 | -12.8417 | -17.4041 | -7.99565 |

同一时期：Yulara2017全候选未来标签均有限，因此两种支持数值完全相同；2018少量缺失使两格略有差异。NIST完整/逐目标的样本数和绝对误差差异更大，但Inverted相对Daily在2017正、2018负的样本均值方向在两种支持下均保留。2017区间跨零，不能声称已证明稳定劣势；2018区间为负也只条件于观察时期与冻结预测。

Yulara原B由2017高误差转为2018低于Daily；扩展B由2017接近Daily转为2018高于Daily。两网格不根据2018成绩重新选择。跨期不做日期一一配对，不解释为纯季节因果效应。

统一候选2017-10-01 00:00至12-31 23:55、2018-04-01 00:00至06-30 23:55，均有历史/目标缓冲。2017的Jan1只是12月末目标缓冲，来源为已有官方NIST年ZIP/已有Yulara数据，不新增起报月份。归档从06:00开始并截断期末，原数字保留；boundary_audit与archive_metric_replay明确对照。NIST固定−05:00，无DST；Yulara保持提供方坐标/+5min可用规则。

2017扩展候选初次混合批次中Yulara Inverted43最大差0.03624 W未过原容差，原批次通过。正式四格采用明确的推理分组：原起报按原批次重新计算，新增起报单独前向，同一权重和处理器；没有把旧保存预测拼进新预测。6个原2017神经回放、8个Ridge回放通过。新2017全候选数组单独保存，2018数组直接复用。

## 实际验证与未验证边界

本次：36组保存新旧数组差异及216行指标影响；5组代表GPU批次/strict加载检查；诊断行独立标识，全部新分析共有1836行逐点指标和114个支持组独立复算；6120行时间块效应/区间由不导入生产interval函数的块计数矩阵实现复算。检查数量不是独立实验数量。932个受保护文件的SHA-256、size与mtime_ns前后完全一致。主文12页、Supplement47页已完成实际渲染检查；归档结果由提交后的交付sidecar记录。

原M1-R/M2整套测试与完整神经训练复现未重跑。原Alice神经处理器未单独保存的限制仍存在，不能由很小指标差恢复它的身份。无神经训练、处理器fit、alpha搜索、新模型或起报月份。新脚本不写死失败20项作为必须满足的新条件；旧verify_light仍是历史快照。

## 投稿与交付

可进入缩窄后的投稿准备，先导师审核新的支持与四格主图。Renewable Energy保持有条件冲刺、Solar Energy为直接主题匹配选项；既有CAS年份/大类/小类未获得权威条目，本次未重新查询，继续待机构核实。不是按JCR推断CAS，也不保证录用。PR_DESCRIPTION_DRAFT.md仅本地，不对外写入描述。

三个包保留全量本地证据/轻量独立复核/当前图件；新旧结果目录分开。打包CRC/试解压只验证归档和轻量算术，不等于神经训练复现。最终提交与包校验信息由交付sidecar记录。

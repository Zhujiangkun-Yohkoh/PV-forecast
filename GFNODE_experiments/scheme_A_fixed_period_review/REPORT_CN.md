# Scheme A 固定范围补强报告（2026-09-09）

起点 ad59ed234de457f251ea16bfb51521f976c8fd83；当前分支 manuscript/clean-pv-benchmark-multisite-revision。本报告取代此前“本轮”状态；历史报告保留在各原目录。

## 当前判定
A 与 C 的分析已完成。B 完成原规则来源核查、真实训练支持、36 个 checkpoint 加载和前向、S1/S2 支持组成，但 **20/36 神经前向未满足历史 rtol=atol=2e-5 容差**；16/36 满足。12/12 Alice 原/扩展 Ridge 系数回放通过。新增 Alice 神经数组保留为未接受的诊断输出，没有据此发布正式 S1/S2 全模型分数。因此本次科研补强为部分完成，不能报告已具备最终投稿条件。

没有神经训练、alpha 搜索或重新拟合处理器。没有覆盖旧原始数据、预测、系数或 checkpoint。前向程序以运行时补丁禁止 KNN、IF、MinMaxScaler 的 fit、Tensor.backward 和 torch.save。

## A：扩展网格的时间不确定性
五系统 × 三功率范围 × 三块长 × 八配对 = 360 行区间；这些是相关描述性统计条目，不是360个独立实验。逐元素核对原/扩展 origin、target_start、标签、有效掩码、Daily 和 Last-value。目标时间为 origin+5…720min。共同完整 H144、Daily-finite 支持，48h主方案，24/72h敏感性，2000次、seed20260908。SSE与计数先合计再开平方；种子效应先分别计算，再平均，不集成预测。零参考RMSE时skill未定义。

Yulara 的关键结果：
| scope | reference | rmse | reference_rmse | effect_kW | ci_low_kW | ci_high_kW |
| --- | --- | --- | --- | --- | --- | --- |
| full | Daily | 15.8421 | 16.0661 | -0.223991 | -1.81356 | 1.74103 |
| full | Inverted mean | 15.8421 | 15.8051 | 0.0370058 | -1.17498 | 1.33605 |
| power-active | Daily | 20.6127 | 22.201 | -1.58831 | -3.30906 | 0.489129 |
| power-active | Inverted mean | 20.6127 | 21.2026 | -0.589917 | -1.84875 | 0.74905 |
| low-power | Daily | 7.9603 | 2.20617 | 5.75413 | 3.13447 | 8.74175 |
| low-power | Inverted mean | 7.9603 | 5.85304 | 2.10726 | 0.0456201 | 4.04847 |

full 中与 Daily、Inverted 的区间都跨零；不能称显著占优、等效或无效。低功率损失单独保留。Qcells 扩展 B 从0.236520升至0.259459kW；NIST B原/扩展预测相同，零差和退化区间未删除。各行提供非空块数、有效重采样次数及零支持次数。原网格区间未覆盖。

## B：负标签支持与尚未解除的前向复现问题
提供方资料确认AC功率含义，但没有找到“所有有限负值为无效”的定义。`_valid_power` 的 finite & >=0 是项目实现规则；历史保留不等于已证明合理，也不能把负数或近零值直接解释为正常净功率。

真实Train/Validation支持：
| site | split | method_support | origins | points | active_fraction |
| --- | --- | --- | --- | --- | --- |
| Sanyo | train | neural | 20409 | 2938896 | 0.412432 |
| Sanyo | train | Ridge | 20343 | 2929392 | 0.413605 |
| Sanyo | validation | neural | 3619 | 521136 | 0.400101 |
| Sanyo | validation | Ridge | 3601 | 518544 | 0.402101 |
| Hanwha | train | neural | 20183 | 2906352 | 0.411773 |
| Hanwha | train | Ridge | 20122 | 2897568 | 0.412804 |
| Hanwha | validation | neural | 4239 | 610416 | 0.407203 |
| Hanwha | validation | Ridge | 4224 | 608256 | 0.408622 |
| Qcells | train | neural | 12747 | 1835568 | 0.400996 |
| Qcells | train | Ridge | 12648 | 1821312 | 0.404134 |
| Qcells | validation | neural | 2999 | 431856 | 0.396753 |
| Qcells | validation | Ridge | 2977 | 428688 | 0.399685 |

神经完整H144训练不要求origin功率有效；Ridge另要求有效origin，不能混称。各小时分布在独立CSV中。

Daily匹配后的支持变化（不是已经接受的新模型评分）：
| site | support | origins | points | unique_targets | active_points |
| --- | --- | --- | --- | --- | --- |
| Sanyo | Original | 4160 | 598317 | 6726 | 259345 |
| Sanyo | S1 | 6816 | 966542 | 6807 | 431594 |
| Sanyo | S2 | 6816 | 967835 | 6816 | 431594 |
| Sanyo | S1_new_origins | 2656 | 368225 | 5060 | 172249 |
| Sanyo | S2_new_origins | 2656 | 369518 | 5069 | 172249 |
| Hanwha | Original | 4521 | 649735 | 6657 | 287368 |
| Hanwha | S1 | 6816 | 966467 | 6807 | 435619 |
| Hanwha | S2 | 6816 | 967761 | 6816 | 435619 |
| Hanwha | S1_new_origins | 2295 | 316732 | 4137 | 148251 |
| Hanwha | S2_new_origins | 2295 | 318026 | 4146 | 148251 |
| Qcells | Original | 2996 | 431150 | 6425 | 187950 |
| Qcells | S1 | 6786 | 958668 | 6781 | 442887 |
| Qcells | S2 | 6786 | 959382 | 6786 | 442887 |
| Qcells | S1_new_origins | 3790 | 527518 | 6755 | 254937 |
| Qcells | S2_new_origins | 3790 | 528232 | 6760 | 254937 |

候选起报只要求72个历史网格位置和Test内至少一个未来点。S1逐点排除负值；S2保留有限原值；超出Test的目标不评分。Last-value和Daily仍沿用原非负规则，未与标签规则混改。负标签不是唯一可能的支持限制；表中是Daily及origin参考交集之后的数量。未发布恢复起报后的正式神经排名，不能断言比较改变或不变。

### 实际定位经过
新增入口初版把“未来标签超出Test”同时作用于过去一天的Ridge输入，已在任何评分前修正：未来标签不可评分，不表示其-24h输入不可用。修正后原/扩展Ridge回放通过。

随后发现Alice部分神经回放误差。Sanyo TCN42相同原起报、原窗口构造与相同批次下最大差0.000247717kW，125/6589个起报至少一点评价容差未通过，差的中位数为0。重新构造的历史窗口与使用已保存Ridge处理器的窗口逐元素相同，但原Alice神经处理器没有独立历史文件，因此这不是与历史神经输入数组的直接对照。CPU/禁用TF32的最大差约0.006973kW，未消除差异；改变推理批量或矩阵TF32也未解决。不能把这些观察证明为唯一后端原因。全部36个记录见 Alice_replay_summary.csv；没有静默放宽容差或替换旧预测。

最小下一步是恢复原Alice推理环境/确切处理状态并通过36个原起报复现，再评分已经生成的新起报。当前不建议先重训，因为尚未确定复现差异来源。若之后接受S1并发现训练选择影响，需要另立Qcells Inverted三seed、原17通道/原日期/原预算、逐目标mask损失的最小定向训练设计；本次未执行，也不据未接受的评分主张其必要性。

## C：同场址次年固定时段
计划先于2018误差计算写入。两场址origin均固定2018-04-01 00:00至06-30 23:55，各26208个候选；最后目标为07-01 11:55。NIST官方年档案提取03-30至07-01共94日作缓冲，原ZIP CRC通过。Yulara用已有同一combined-system文件。2017处理器、6个Inverted checkpoint、两套Ridge系数和阈值全部冻结；6个外部神经原2017完整起报回放及8个Ridge回放通过。

2018是逐目标有限值与有效origin/Last-value/Daily的共同交集；原2017主表使用完整H144合法支持。时期和支持定义均标明，不能把二者差异全归因为季节。数据集此前用于覆盖审计，本次固定期误差按本计划生成，不包装成整个项目从未查看数据的确认性验证。

12h/full，kW：
| site | Daily | Expanded A | Expanded B | Inverted mean | Last-value | Original A | Original B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NIST_GROUND | 51.0279 | 51.8961 | 40.777 | 38.1861 | 91.3884 | 51.8961 | 40.777 |
| YULARA_COMBINED | 9.5597 | 25.7208 | 10.8351 | 13.8591 | 40.9371 | 17.8025 | 8.01881 |

主要48h配对区间：
| site | method | reference | effect_kW | ci_low_kW | ci_high_kW |
| --- | --- | --- | --- | --- | --- |
| YULARA_COMBINED | Inverted mean | Daily | 4.29935 | 2.92882 | 5.65394 |
| YULARA_COMBINED | Original B | Daily | -1.5409 | -1.92481 | -1.12965 |
| YULARA_COMBINED | Expanded B | Daily | 1.27536 | 0.416224 | 2.26544 |
| YULARA_COMBINED | Expanded B | Inverted mean | -3.024 | -3.71293 | -2.28505 |
| NIST_GROUND | Inverted mean | Daily | -12.8417 | -17.4041 | -7.99565 |
| NIST_GROUND | Original B | Daily | -10.2509 | -11.9317 | -8.40567 |
| NIST_GROUND | Expanded B | Daily | -10.2509 | -11.9317 | -8.40567 |
| NIST_GROUND | Expanded B | Inverted mean | 2.59087 | -0.618689 | 5.66456 |

Yulara原B在新时期低于Daily，扩展B高于Daily；Inverted也高于Daily，2017中神经对Daily均值占优的模式未延续。不能用已知新分数重新选择alpha或把原网格改成新的预注册主模型。NIST中Inverted低于Daily，原/扩展B仍相同且低于Daily，B−Inverted区间跨零。模型、网格与时期的排序不稳定是本次新证据，不是实验失败。

NIST 12h的Inverted−Daily加权MSE贡献（先每seed分解后平均）：
| scope | delta_MSE | points | full_points |
| --- | --- | --- | --- |
| low-power | 130.092 | 1749335 | 3502466 |
| power-active | -1275.75 | 1753131 | 3502466 |

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

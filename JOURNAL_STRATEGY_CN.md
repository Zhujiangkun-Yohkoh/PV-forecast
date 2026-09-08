# 本轮投稿判断更新（2026-09-08，诊断修订）

## 已核实事实

- [Renewable Energy 官方范围及文章类型](https://shop.elsevier.com/journals/renewable-energy/0960-1481)：可再生发电工程，包含光伏、气候/气象；接收原创研究，综述需编辑邀请。作者指南及出版选项页本轮返回403，具体格式、APC和订阅作者路线仍待官方页面核实。不能将读者订阅价格当成作者费用。
- [Solar Energy 的 ISES 官方介绍](https://www.ises.org/what-we-do/publications/solar-energy-journal)：接收太阳能研究、测量和应用的原创研究或综述。范围直接匹配本研究；ScienceDirect作者指南与Insights本轮未能读取，格式和当前费用待核实。页面7.9未标指标年，不当作最新版JIF。
- [JRSE 官方 About](https://pubs.aip.org/aip/jrse/pages/about)：范围含光伏及能源气象；页面明确2025 JCR（Clarivate, 2026）JIF 2.4；JCR在Energy & Fuels、Green & Sustainable Science & Technology均为Q4。这是JCR信息，**不是中科院分区**。
- [JRSE 编辑政策](https://pubs.aip.org/aip/jrse/pages/policies)：本研究应按Article评估；Article无固定长度上限，但需有足够增量并可复现。仅将常规计算移到新系统不自动构成贡献。
- [JRSE费用页](https://pubs.aip.org/aip/jrse/pages/charges)：无强制版面费；可选Author Select开放获取3800美元。[About](https://pubs.aip.org/aip/jrse/pages/about)确认可选择传统订阅或混合OA。
- [AIP作者格式指引](https://publishing.aip.org/resources/researchers/author-instructions/)：摘要单段、250词以内；图件按最终尺寸、嵌入字体、面板(a)(b)…，一般双栏最大宽6.69英寸、文字最低8pt。本轮仍是审阅模板；若改投Elsevier，需按对应刊最新指南换版，不声称AIP模板通用合规。
- [中科院官方分区入口](https://www.fenqubiao.com/)仍要求账号/CARSI，未取得三刊条目。**最新版年份、大类/小类及分区均待机构核实**；主页版权所有2026不是分区版本证明。

查询日期：2026-09-08。费用作为可选信息，未代选OA或支付。用户此前明确说过“出版用混合APC，不支付出版费”；这是有出处的既往偏好，保留为最终费用选择时需要核对的背景，而不把它扩展为未经确认的永久硬性限制。

## 适配性判断（不是已核实的分区排序）

候选仍为 **Renewable Energy冲刺 → Solar Energy → JRSE**。本轮Yulara诊断、共同起报筛选机制、Ridge配对区间使稿件从模型清单转向可验证的评价问题，Solar Energy的读者匹配最直接。若导师坚持高分区优先，应先取得机构CAS条目再正式排列，不能据此表直接宣布RE分区高于其余期刊。

冲刺仍有科学风险：原Alice负值标签规则造成训练和评价支持选择；三地、两种信息宽度、不同年份不支持普遍泛化。Yulara线性参考的异常解释已得到实际矩阵与贡献证据支持，不能再简单写成“线性模型不适合Yulara”。NIST配对区间对所观察日期有支持，不等同跨年份稳定。继续堆叠架构不能消除这些风险。

## 现有覆盖及最小可执行扩展（尚未执行）

本轮读取真实文件时间列：三Alice文件各44064行，覆盖2018-04-01至08-31；Yulara覆盖2016-04-01至2026-02-19，2018有104155行（覆盖不等于规则时间完整）；NIST本地仅2017年365个日文件。逐年清单在available_data_coverage.csv。

最小跨季节/次年检查可固定为 **Yulara和NIST的2018年4–6月**，与已评估2017年10–12月间隔约半年。先补齐NIST这91个日文件及当年的字段资料，核查Yulara相同季度时间和缺失。保持七通道、6h历史、12h输出、同样目标范围和Daily匹配；使用原2017已拟合处理器、Inverted三个checkpoint、原网格及本轮扩展网格Ridge和确定性参考，仅做前向评价，**无需先新增神经训练**。主评价先固定12h/full及power-active，其他窗口为次要；模型年龄和缺失分布漂移独立报告，不根据新季节成绩再调规则。数据和运行预算大致为两季度数据审计加数分钟至数十分钟前向与统计计算，实际时长待运行；NIST下载体积应先查文件清单，不能先承诺固定MB。

这项计划检验同场址次年时间迁移，不是新场址零样本泛化，也不能由两个季度宣布全年覆盖。原Alice规则是否有提供方物理/无效码依据仍是另一项科学问题：若决定采用保留有限负值的新训练协议，应另设更正实验，不能静默改写现有36次训练。本轮完整报告了该规则的选择机制。

## 待确认事项

科学层面：不同年份/季节的独立时间段、Alice负值规则的适用依据及是否另做协议敏感性。格式层面：目标期刊最终模板、图宽、Highlights/声明文件及引文格式。作者层面：最终声明、公开范围/许可证、投稿授权和出版选项。三类事项分开，未填报或提交任何作者声明。

---

# Scheme A 投稿策略（2026-09-08 核查）

## 上一轮候选判断（保留背景）

**建议首投 Renewable Energy（冲刺），下一顺位 Solar Energy，稳妥备选 JRSE。** 这是依据本轮修改后的科学内容与已核实 scope 给出的编辑策略，**不是已经核实的中科院分区递降表**。导师要求按中科院大类由高到低执行时，正式投递前须通过机构账号核实下表分区；如实际分区与顺序不符，应调整顺序。不能用 JCR、SJR 或影响因子代替这一步。

本轮已把文章从“四种网络比较”改为“参考历史与目标功率范围如何改变预测收益判断”。最有价值的新证据是可加和的 NIST 误差反转解释，以及覆盖五系统的 Ridge／Ridge＋日周期信息对照。尚没有跨全年、跨年份的同输入确认性证据，也没有调度收益或误差经济成本。这些是冲刺期刊的具体拒稿风险，不能靠语言润色消除。Renewable Energy 首投应接受较高的编辑拒稿可能性；若导师更重视范围匹配和时间成本，Solar Energy 是更直接的学术归宿。

## 分区、收录与指标：分别登记

查询日均为 **2026-09-08**。中科院官方入口仅返回登录/CARSI界面，未得到七刊的期刊条目；首页“版权所有2026”不证明存在“2026版分区”。本轮没有机构 JCR/Master Journal List 登录结果。因此以下空缺是未核实，而不是低分区或未收录。[中科院官方入口](https://www.fenqubiao.com/)、[Clarivate Master Journal List](https://mjl.clarivate.com/home)。

|期刊|最新正式中科院版本年／大类及分区／小类及分区|JCR年份／学科分区|JIF及其年份|当前SCIE/ESCI等收录|
|---|---|---|---|---|
|Renewable Energy|均待机构核实|待核实|待核实|待MJL条目核实|
|Solar Energy|均待机构核实|待核实|官方页面存在未标年7.9与期次页6.6的差异，不指定为最新JIF|待MJL条目核实|
|Energy and AI|均待机构核实|待核实|待核实|待MJL条目核实|
|Energy|均待机构核实|待核实|待核实|待MJL条目核实|
|Energy Reports|均待机构核实|待核实|待核实|待MJL条目核实|
|Solar Energy Advances|均待机构核实|待核实|待核实|待MJL条目核实|
|JRSE|均待机构核实|待核实|**2.4**；AIP脚注原文为“2025 Journal Citation Reports Science Edition (Clarivate, 2026)”|待MJL条目核实|

JRSE 的 JIF 与混合开放获取来自 [AIP 官方能源期刊页](https://publishing.aip.org/resources/topical-portfolios/explore-energy-with-aip-publishing/)，不是第三方倒推分区。Solar Energy 的冲突来源为 [ISES期刊介绍](https://www.ises.org/what-we-do/publications/solar-energy-journal) 与 [ScienceDirect第315卷](https://www.sciencedirect.com/journal/solar-energy/vol/315/suppl/C)。不要将无年份页面中的值直接写入导师选刊表。

## 七刊适配、近年比较与风险

### 1. Renewable Energy：首投冲刺

官方 scope 包含光伏发电、气象及可再生能源工程，接受原创研究；本稿应作为 research paper，而非算法创新或综述。[出版社scope](https://shop.elsevier.com/journals/renewable-energy/0960-1481)。

近年相近研究：*The added value of combining solar irradiance data and forecasts: A probabilistic benchmarking exercise*（2024，DOI [10.1016/j.renene.2024.121574](https://doi.org/10.1016/j.renene.2024.121574)）。它研究辐照观测与预报信息的组合及概率预测。本稿增量是实际AC功率的匹配目标比较、低功率MSE抵消、同一线性方法中增加可用日周期信息的对照，不宣称覆盖其概率/NWP任务。

主要拒稿风险：编辑可能认为三个场址、不同年份和两套输入仍不足以支撑具有广泛工程意义的发现；新增分析为事后，Ridge在Yulara的分布稳定性也不足。若要求补强，最小方向是现有两个外部场址增加一个明确的不同季节或次年 held-out 时间段，维持公共输入与参考定义，而非增加复杂网络。当前没有授权启动这一扩展。

费用：出版社销售页确认存在订阅发行，但这本身不是作者免APC证明。投稿前须核实作者指南中的订阅发表路线、OA选项及其他费用；本轮未选择OA、未支付或承诺免收费。

### 2. Solar Energy：下一顺位，主题匹配最直接

ISES官方将其定位于太阳能科学、应用、测量与技术研究。本稿对太阳能预测评价参考的实证解释与其读者直接相关。[官方介绍](https://www.ises.org/what-we-do/publications/solar-energy-journal)。建议原创研究稿。

近年比较：*On the use of sky images for intra-hour solar forecasting benchmarking: Comparison of indirect and direct approaches*（2024，DOI [10.1016/j.solener.2024.112649](https://doi.org/10.1016/j.solener.2024.112649)）。其重点是图像驱动的短时辐照预测路径；本稿提供历史功率任务中参考历史、累计窗口和功率范围的分离，不直接比较其误差值。

风险：太阳能预测审稿人可能要求解释Daily为何随场址改变、清空/晴空参考的关系以及全年气象覆盖。本文已以可核验的误差分解代替未经证明的天气故事，但尚未证明具体气象机制。费用金额及当前订阅/OA作者选项待官方投稿页面核实，不依据历史经验承诺零费用。

### 3. JRSE：稳妥备选

AIP官方scope包含太阳能、能源气象与气候，允许Articles/Methods等类型，明确为混合OA。适合将本稿作为可复核的经验研究；不是因为原稿使用AIP模板才入选。[官方scope、类型与模式](https://publishing.aip.org/resources/topical-portfolios/explore-energy-with-aip-publishing/)。

相近研究：*Robust day-ahead solar forecasting with endogenous data and sliding windows*（2024，DOI [10.1063/5.0190493](https://doi.org/10.1063/5.0190493)；[作者机构记录](https://repository.kaust.edu.sa/items/d1ed4eb0-3506-4707-867b-1dcba4b32e8d)）。该工作研究内生数据、滑动窗口和日剖面预测。本文增量是同一目标交集下的双参考反转、逐功率范围的可加和解释，以及共址与独立拟合外部结果分层。

风险仍包括经验结果的增量、短Alice时间支持和模型预算上限。混合OA不等于必须缴OA APC；具体费用、许可和最终非OA路线仍须作者在投稿时确认。

### 4. Energy and AI：科学适配，但不自动排入无费用路线

近期直接竞争工作：*A unified benchmark of deep learning and classical machine learning models for weather-based solar power forecasting: Balancing complexity and skill*（2026，DOI [10.1016/j.egyai.2026.100833](https://doi.org/10.1016/j.egyai.2026.100833)）。它已比较多种经典与深度模型；“加入Ridge”本身不构成独有贡献。本稿须突出信息控制和目标功率分解，而非再做模型榜单。

scope和研究稿适配可由该期刊已发表的天气预测benchmark支持，但本轮官方完整作者指南未成功读取。当前OA/APC金额、减免资格及收录分区待核实；在作者“不支付出版费”的偏好下，不承诺可行投稿路线。主要风险是与近期benchmark相比规模较小、缺少概率或实际决策价值。

### 5. Energy：本轮不优先

当前官方scope强调综合能源系统、规划、管理及热能；纯电力单技术研究应特别评估适配。[官方scope](https://shop.elsevier.com/journals/energy/0360-5442)。近例 *Improving ultra-short-term photovoltaic power forecasting using a novel sky-image-based framework considering spatial-temporal feature interaction*（2024，DOI [10.1016/j.energy.2024.130538](https://doi.org/10.1016/j.energy.2024.130538)）不保证当前历史功率benchmark同样匹配。

本稿增量在评价解释而非图像架构，但目前没有把预测差异映射到储能/调度成本；这是具体范围风险。不要为投该刊临时编造应用收益。研究论文类型、订阅存在可核实；作者费率和OA选项仍待核实。

### 6. Energy Reports：条件备选

近年原创比较：*Long-term power forecasting of photovoltaic plants using artificial neural networks*（2024，DOI [10.1016/j.egyr.2024.08.062](https://doi.org/10.1016/j.egyr.2024.08.062)）比较多种ANN结构与一年气象/功率数据。本稿的增量是精确的预测窗口、匹配参考信息及可复算分解，不用不同任务的数值争高低。[出版社文章页](https://www.sciencedirect.com/science/article/pii/S2352484724005535)。

该篇为开放获取研究论文；不能仅凭单篇OA推断期刊当前全部收费要求。完整期刊scope、当前OA/APC金额和减免仍待官方指南核实。没有确认符合作者费用偏好前，不排在JRSE前作为“稳妥”路线；分区也不能凭声誉判断。

### 7. Solar Energy Advances：范围匹配的OA条件备选

ISES官方明确为完全开放获取，scope包括太阳能数据分析、AI、测量与预测。[官方期刊页](https://www.ises.org/what-we-do/publications/solar-energy-advances)、[官方scope说明](https://www.ises.org/sites/default/files/SEJ%20%2B%20SEA/4275%20Solar%20Energy%20Advances%20A5%20PROOF.pdf)。

近例 *Very short-term probabilistic and scenario-based forecasting of solar irradiance using Markov-chain mixture distribution modeling*（2024，DOI [10.1016/j.seja.2024.100057](https://doi.org/10.1016/j.seja.2024.100057)）研究概率/场景短时预测，本文则贡献确定性功率参考和范围解释。它不是相同任务的数值对标。费用金额待核实；会议特刊历史减免不适用于本稿，不能据此声称免APC。

## 最小补强与停止边界

本轮已完成无需神经重训的主要解释补强与十个线性参考。更高分区最有价值的下一步是**已有两个外部场址的跨季节/跨年份同协议性能重复**，至少加入与10–12月不同的完整连续时段；需先确认原始数据覆盖和固定Train/Validation/Test关系，评估原权重前推还是独立拟合，不能把已知Test重新包装为新确认。成本取决于原始记录是否完整；若保持四模型三seed并独立拟合两个场址，将是额外24次神经训练，不能称为零成本润色。当前不启动。

另一个更小的解释方向是对NIST低功率正偏差做固定阈值下的时序残差分析，不需要增加模型。当前结果已足够支持条件性benchmark文章，但不足以证明跨全年稳定收益、普遍架构优越或可观经济价值。

## 仍需人工完成的外部信息

需要机构账号导出七刊的最新版正式中科院大类/小类、版本年份，以及JCR学科分区和MJL收录条目；需要官方作者指南确认费率/非OA路线。以上未核实项明确保留，不阻止本轮科学修改与私下导师审核包交付。报告不包含实际投稿或费用承诺。

# 问题—修改—证据对应表（2026-09-08）

路径均相对项目根目录。`review/`在本表中指 `GFNODE_experiments/scheme_A_review_extension/`；`paper/`指 `manuscript/clean_pv_benchmark/`。这是本轮状态，不把旧M3通过数冒充本轮结果。

|编号|原文件位置|问题及影响|拟采取措施|实际修改|验证证据|状态|剩余限制|
|---|---|---|---|---|---|---|---|
|R01|paper/main.tex 摘要、方法及旧图3/4；Supplement|H12等容易被理解为孤立lead误差|区分累计窗口与固定lead|方法给出1/4/8/12h前缀映射；主图轴改小时；新增144个固定lead误差|review/results/lead_specific.csv；Fig S4；逐点对照6720项|完成|不同原始窗口合法origin不同，正文明确不能全归因窗口长度|
|R02|原primary_horizon_specific及Qcells短窗|样本组成随窗口改变|保留主分析并做common-H144|五系统、四模型、两种支持、两种参考交集、三功率范围均重算；Fig S2分完整参考范围与神经细节|metrics_decomposition.csv 的support/origins/points；compare_frozen.py|完成|Qcells1h active由36504降至42，Last-value在共同子集领先；完整H144不能代表被排除时段|
|R03|main.tex、supplementary.tex、图注表头|daylight实为真实功率阈值|命名准确且不改阈值|全文当前图表改power-active；补集low-power；保留冻结CSV键daylight并解释|阈值源自Train；三范围可加和检查2400项|完成|不能推断夜晚、积雪、停机；未伪造太阳高度角|
|R04|原模型介绍autoregressive/direct表述|与真实forward不一致|逐层核对，不改权重|正文给出固定context的GRUCell状态递推；三直接多输出头分开；Supplement说明dropout/normalization|原run_corrected_benchmark.py 297–402行；当前Supplement结构和参数表|完成|本轮8个17/7通道synthetic forward及模块参与检查通过；不加载checkpoint|
|R05|固定预算架构排名解释|相同预算不代表充分优化|查全部训练记录并收窄结论|60份历史日志最优/停止轮核对；NIST TCN三seed最佳均25；Fig S9|learning_histories.csv、training_budget.csv、REVIEW_VERIFICATION.json|完成|预算边界仍存在；未延长或重训神经模型|
|R06|仅三seed SD|未表达时间支持不确定性|预写方案后配对块重采样|48h主分析，24/72h敏感性，2000次，共用方法和seed索引，SSE/count重算；正文报告关键区间|ANALYSIS_PLAN.md；paired_block_intervals.csv；Fig S5|完成|区间条件于已观察日期与已拟合模型；跨块残余相关，Alice时间短；无未经计划p值|
|R07|NIST H144/full与active反转|缺少直接数值解释|逐seed恒等式和逐点分解|五系统全部模型窗口分解RMSE/MAE/bias/比例及加权MSE；主文Table IV、Fig5|2400项SSE恒等式；NIST净MSE增加234.744 kW²|完成|低功率为统计子集，不是操作事件标签|
|R08|仅两个确定性参考|简单学习与日周期输入作用未区分|五系统Ridge A/B、固定有限alpha、Validation选择|完成10个确定性fit；50候选Validation分数全部保留；Train-only变换；同目标评价；不伪造seed|ridge_metrics.csv、ridge_validation_selection.csv；480逐点独立指标行检查；Table V、Fig S8|完成|事后新增；Yulara线性参考较差仍保留；不证明给神经网络增加同信息后的效果|
|R09|外部结果缺少时序案例|不易理解成功与失败|按origin月份和透明规则挑案例|外部三月汇总；每站三固定日历起报及明确posthoc最好/最坏seed42；每曲线单一起报|monthly.csv、trajectory_cases.csv；Fig S6/S7|完成|极端例子不是代表频率；无天气故事|
|R10|旧标题与重复审计叙事|主线过分强调流程、用词混淆|围绕参考信息与功率误差组成重构|新标题、196词摘要、独立拟合与事后分析分层；正文约2895词；旧正文4245词（同口径）|WORD_COUNTS.json；main.tex；引用/术语及PDF检查|完成|作者声明与公开URL仍需最终确认；本轮不猜测|
|R11|旧8幅图、极值压缩、latency线性轴|读图困难，排名放大细小差异|重绘当前所有正式图并实际看PDF|5主图＋9补图，全部PDF/SVG/320dpiPNG及CSV/caption/alt；旧新预览；NIST加权MSE和固定轨迹|review_figures/；FIGURE_QA.md；FIGURE_OVERVIEW.png；PDF逐页渲染记录|完成|这是经过视觉检查的科学初版；S2/S7多面板标签可在定刊后再做美术微调，数据与整图可独立交接|
|R12|Windows字体、私人数据查找、git show|离开原环境不易复现|自包含当前入口和显式路径|当前build_figures入口用Matplotlib字体；portable_entry；prepare_training_copy保留历史源码、独立新目录显式路径，已导入检查|ENVIRONMENT_AND_REPRODUCTION.md；CSV绘图与保存预测指标实际执行；训练副本inspect成功|部分完成|本轮禁止神经重训，未测试官方数据→60次完整训练；CAS/JCR/作者费用路线需机构/官方核实|

## 科学结论变化

原冻结神经数字不变。新增分析使解释更具体并限制推广：小幅Yulara长窗Daily收益的时间区间跨零；NIST低功率额外误差抵消active节省；Ridge＋日周期在NIST的12h/full优于现有主模型和Daily，但在Yulara仍较差。论文不再把神经模型普遍占优或某结构普遍第一作为贡献。

## 未执行项目

本轮没有神经训练、微调、checkpoint更新或额外神经模型；没有原60次训练全复现；没有自动完成作者声明、许可证、OA选择、公开发布或实际投稿。CAS最新版大类/小类、JCR分区及大部分当前JIF尚未获得权威条目，见JOURNAL_STRATEGY_CN.md。

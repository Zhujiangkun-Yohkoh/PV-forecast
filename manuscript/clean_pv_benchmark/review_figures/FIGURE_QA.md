# Figure QA — 2026-09-08

正式图14组：5主图、9补图。PDF/SVG矢量、PNG320dpi，源CSV/caption/alt逐图同名。最终主文约7.1英寸宽；实际查看全部主文10页、Supplement36页渲染，图件没有可见裁切或文字遮挡。主图5图与补图9图均已查看。

热图初次检测到imshow栅格化后已改pcolormesh矢量色块；不是仅把位图放进PDF。Daily统一深灰虚线，Inverted蓝、recurrent橙、patch绿、TCN紫，以线型/标记辅助灰度。历史旧图仅放figures中，不与当前编号混用。

|当前图|源数据|科学职责与检查|剩余事项|
|---|---|---|---|
|fig1_design|fig1_design_data.csv|三场址独立拟合和单一起报；示意轴明确非等比例|未发现需要阻止审阅的版式问题|
|fig2_alice_references|fig2_alice_references_data.csv|双参考同Daily交集；ratio=1、对数轴、全部不利点保留|未发现需要阻止审阅的版式问题|
|fig3_external_skills|fig3_external_skills_data.csv|外部seed点及sample SD；Daily负值保留|未发现需要阻止审阅的版式问题|
|fig4_neural_gaps|fig4_neural_gaps_data.csv|相对最佳神经误差差距，不以平均rank放大微差|未发现需要阻止审阅的版式问题|
|fig5_decomposition_case|fig5_decomposition_case_data.csv|加权MSE可加和、low占比；固定单一起报轨迹|未发现需要阻止审阅的版式问题|
|figS1_alice_ranks|figS1_alice_ranks_data.csv|三共址阵列分面、离散灰度、矢量色块|未发现需要阻止审阅的版式问题|
|figS2_common_origins|figS2_common_origins_data.csv|HS/CO曲线分开，左完整参考范围，右神经细节|多面板标签可在定刊尺寸后再微调；当前可读|
|figS3_latency|figS3_latency_data.csv|对数latency轴和参数量，未声称0.535与0.558稳定差异|未发现需要阻止审阅的版式问题|
|figS4_fixed_leads|figS4_fixed_leads_data.csv|固定lead与累计前缀区别，共同H144 origins|未发现需要阻止审阅的版式问题|
|figS5_block_intervals|figS5_block_intervals_data.csv|24/48/72h区间完整保留，不按跨零与否选择|未发现需要阻止审阅的版式问题|
|figS6_monthly|figS6_monthly_data.csv|按origin月归组，完整12h轨迹不拆成不同月份|未发现需要阻止审阅的版式问题|
|figS7_cases|figS7_cases_data.csv|固定日历和posthoc极端都保留；seed42，单一origin|多面板标签可在定刊尺寸后再微调；当前可读|
|figS8_ridge|figS8_ridge_data.csv|五系统所有Ridge结果保留，Yulara不利值未剪裁|未发现需要阻止审阅的版式问题|
|figS9_learning_budget|figS9_learning_budget_data.csv|历史60日志核对；NIST TCN预算边界三seed展示|未发现需要阻止审阅的版式问题|

本轮没有对PNG预览声称可编辑统计图。SVG保留文字，PDF嵌入字体；真正的可重建源为Python与CSV。BEFORE_AFTER.png供旧新对照，FIGURE_OVERVIEW.png供导航。Supplement表格密集但保留逐seed证据，未为压页缩小正文字号；单图页留白为独立阅读排版。

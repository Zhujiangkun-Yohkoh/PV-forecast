# 从这里开始：完整导师审核版

本轮已完成R01–R11的科学修改、十个Ridge参考、14组重绘图及论文整合；R12三条复现路径中完整神经重训未重跑，机构分区/费用信息仍待核实。原60个神经实验和数据证据不改写。

建议阅读顺序：
1. PROJECT_REPORT_CN.md：当前新发现与完整历史流程。
2. manuscript/clean_pv_benchmark/main.pdf、supplementary.pdf：本轮稿件。
3. REVIEW_RESPONSE_MATRIX.md：逐项实际处理及限制。
4. manuscript/clean_pv_benchmark/review_figures/FIGURE_OVERVIEW.png、FIGURE_QA.md：图件总览与视觉记录。
5. JOURNAL_STRATEGY_CN.md：Renewable Energy冲刺、Solar Energy下一顺位、JRSE备选；CAS正式分区待机构核验。
6. REVIEW_VALIDATION.md、ENVIRONMENT_AND_REPRODUCTION.md：本轮验证与未重跑路径。

本轮关键变化：Qcells共同H144子集的1h active点数从36504降为42，Last-value反而优于四神经模型（full RMSE0.00398 kW、Inverted0.12494）；NIST低功率误差解释总体反转；小幅长窗Daily收益的时间区间跨零；日周期输入改善五系统Ridge但不能产生统一最佳模型。不把事后结果写成原预设确认。

完整私下ZIP包含project、evidence和data目录；具体文件由PACKAGE_MANIFEST.csv逐项列出，并附PACKAGE_VERIFICATION.md和Git交付信息。此前仅含论文/代码/汇总CSV的小包是历史轻量包，不能替代本轮完整实验审核包。私人路径配置、虚拟环境、凭证、缓存和旧ZIP排除。

包用于导师审核，不是投稿授权、公开发布或数据再分发许可。最终作者声明、许可、公开URL和费用决定未代替作者完成。

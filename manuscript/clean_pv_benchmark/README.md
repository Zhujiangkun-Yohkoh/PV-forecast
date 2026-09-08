# Scheme A：当前投稿前审核版

**Reference Information and Target-Power Regimes in Multi-Window Photovoltaic Forecasting**。当前首投建议及待核实分区见根目录 JOURNAL_STRATEGY_CN.md，不默认JRSE。

先读根目录START_HERE_REVIEW.md、PROJECT_REPORT_CN.md、REVIEW_RESPONSE_MATRIX.md。当前主文10页、Supplement36页；5主图和9补图，位于review_figures。figures目录为保留的历史图件，不能与当前编号混用。

绘图：从本目录运行 `python -B build_figures.py`；或使用根目录ENVIRONMENT_AND_REPRODUCTION.md中的portable入口。依赖Matplotlib自带DejaVu字体，无Windows字体路径。build_figures_legacy.py是原M3实现的历史副本，不是本轮重建入口。

TeX：`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`，Supplement同理。数据代码/训练/指标路径及本轮未重跑项目见复现指南。原60次神经实验冻结不变，本轮10个Ridge对照与posthoc解释结果在scheme_A_review_extension。

作者审稿、许可证、最终公开URL和正式投稿仍未代替作者完成。私下完整审核ZIP包括可用本地证据，不表示公开再分发授权。

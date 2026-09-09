# 最终图件与PDF QA（2026-09-09）

本轮实际：完整编译main12页、Supplement43页；全部55页85dpi渲染并逐页联系表查看，主文3/8页与补充13页修复后再次按实际页面查看。无overfull、undefined citation/reference、重复label；所有字体嵌入，页面无栅格image对象。REVTeX默认子刊/字体替代及BibTeX样式控制提示不冒充科学失败；无缺失引用。

主文普通图字最小约8.83pt；Supplement约8.16pt。PDF的旋转文字包围盒不能直接当字体点数，检查排除旋转度量；仅latency的log上标0/1约5.98pt，普通刻度与标签8.54pt。所有图为PDF/SVG与320dpi PNG，PDF图未栅格化。最终排版审查对象是整页嵌入图，不只独立放大PNG。

统计图主线已可读，Supplement仍采用单图单页，部分小图页留白较多；定刊后可并排排版。S4共同起报有十面板，是最密集图，已保留可读字体与单独Qcells解释图；可进一步人工调整美观，不改变数据或筛选。历史完整表占较多页，按要求后置，未缩字强行限页。

|当前图号|源文件名|证据与已检查改动|剩余问题|
|---|---|---|---|
|1|fig1_design|起报虚线限定在时间轴，72样本端点−355..0min，previous-day至−12h；无池化暗示。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|5|fig2_alice_references|共同Daily-valid双参考、log轴和ratio=1，三阵列分隔；envelope为事后描述性。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|6|fig3_external_skills|48h时间区间取代微小seed SD；负区间不裁剪。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S6|fig4_neural_gaps|零值下方留空；分母为该窗口范围最佳神经均值RMSE。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|4|fig5_decomposition_case|MSE可加和、斜纹低功率、净差置于柱外；固定EST案例及日期。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S7|figS1_alice_ranks|四级离散灰阶，数字1最好4最差，仅四神经模型。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S4|figS2_common_origins|HS/CO线型、左完整参考/右神经细节，final普通字体约8.16pt；Qcells机制另图，不靠密集图解释。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S8|figS3_latency|log轴、batch1/硬件/计时边界，参数数目；普通标签约8.54pt，log指数0/1约5.98pt属于上标。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S5|figS4_fixed_leads|固定提前量、共同H144及Daily匹配，与累计指标分开。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S9|figS5_block_intervals|24/48/72h同时保留，时间区间非seed SD。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S10|figS6_monthly|按起报月份，新增每月O和目标对N；不按目标日期归月。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S11|figS7_cases_fixed|三个月×两站，seed42，固定日历规则；每条轨迹来自一个起报。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S12|figS7_cases_extremes|独立两行图，事后最好/最差诊断不代表频率，不补写天气故事。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S13|figS8_ridge|保留原网格全部不利结果；不与扩展网格无标注混合。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S14|figS9_learning_budget|三条历史TCN记录保留25轮边界现象，不当作本轮训练。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|2|fig6_ridge_intervals|5系统×full/active；负值B更好，各面板不同kW尺度明确。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|3|fig7_qcells_selection|小时分布和active比例；42是重复target pairs，caption注明25唯一时间。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S3|figS10_qcells_support|Train/Val/Test及剔除子集，显式功率分箱；不得称物理daylight。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S1|figS11_yulara_diagnostic|真实极值/分位数与缺失暴露，Train最大值非裁剪线。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|
|S2|figS12_alpha_sensitivity|所有10个敏感性结果并列，Qcells B恶化保留；五分面尺度明确。 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|

FIGURE_OVERVIEW.png与BEFORE_AFTER_1—5.png提供全图及旧新对照；新增图没有旧版时明确标注。FIGURE_NUMBER_MAP.csv是源文件到新编号映射。每图caption与alt独立提供。

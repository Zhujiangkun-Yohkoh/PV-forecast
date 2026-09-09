"""Record performed PDF checks and per-figure handoff notes after visual inspection."""
from pathlib import Path
import json,re
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=ROOT/'manuscript/clean_pv_benchmark';F=P/'diagnostic_figures'
qa=json.loads((HERE/'results/PDF_QA.json').read_text(encoding='utf8'));records=json.loads((F/'FIGURE_CAPTIONS_AND_ALT.json').read_text(encoding='utf8'));mapping=pd.read_csv(F/'FIGURE_NUMBER_MAP.csv').set_index('source_figure').current_number
for n,r in qa.items():
 assert not r['overfull_lines'] and not r['undefined_references'];assert all(p['raster_images']==0 for p in r['pages']);assert all(not p['small_character_text'] or set(p['small_character_text'])<=set('01') for p in r['pages'])
 for line in r['font_table'].splitlines()[2:]:
  if line.strip():assert 'yes' in line,line
notes={
'fig1_design':'起报虚线限定在时间轴，72样本端点−355..0min，previous-day至−12h；无池化暗示。',
'fig2_alice_references':'共同Daily-valid双参考、log轴和ratio=1，三阵列分隔；envelope为事后描述性。',
'fig3_external_skills':'48h时间区间取代微小seed SD；负区间不裁剪。',
'fig4_neural_gaps':'零值下方留空；分母为该窗口范围最佳神经均值RMSE。',
'fig5_decomposition_case':'MSE可加和、斜纹低功率、净差置于柱外；固定EST案例及日期。',
'figS1_alice_ranks':'四级离散灰阶，数字1最好4最差，仅四神经模型。',
'figS2_common_origins':'HS/CO线型、左完整参考/右神经细节，final普通字体约8.16pt；Qcells机制另图，不靠密集图解释。',
'figS3_latency':'log轴、batch1/硬件/计时边界，参数数目；普通标签约8.54pt，log指数0/1约5.98pt属于上标。',
'figS4_fixed_leads':'固定提前量、共同H144及Daily匹配，与累计指标分开。',
'figS5_block_intervals':'24/48/72h同时保留，时间区间非seed SD。',
'figS6_monthly':'按起报月份，新增每月O和目标对N；不按目标日期归月。',
'figS7_cases_fixed':'三个月×两站，seed42，固定日历规则；每条轨迹来自一个起报。',
'figS7_cases_extremes':'独立两行图，事后最好/最差诊断不代表频率，不补写天气故事。',
'figS8_ridge':'保留原网格全部不利结果；不与扩展网格无标注混合。',
'figS9_learning_budget':'三条历史TCN记录保留25轮边界现象，不当作本轮训练。',
'fig6_ridge_intervals':'5系统×full/active；负值B更好，各面板不同kW尺度明确。',
'fig7_qcells_selection':'小时分布和active比例；42是重复target pairs，caption注明25唯一时间。',
'figS10_qcells_support':'Train/Val/Test及剔除子集，显式功率分箱；不得称物理daylight。',
'figS11_yulara_diagnostic':'真实极值/分位数与缺失暴露，Train最大值非裁剪线。',
'figS12_alpha_sensitivity':'所有10个敏感性结果并列，Qcells B恶化保留；五分面尺度明确。'}
text='''# 最终图件与PDF QA（2026-09-09）

本轮实际：完整编译main12页、Supplement43页；全部55页85dpi渲染并逐页联系表查看，主文3/8页与补充13页修复后再次按实际页面查看。无overfull、undefined citation/reference、重复label；所有字体嵌入，页面无栅格image对象。REVTeX默认子刊/字体替代及BibTeX样式控制提示不冒充科学失败；无缺失引用。

主文普通图字最小约8.83pt；Supplement约8.16pt。PDF的旋转文字包围盒不能直接当字体点数，检查排除旋转度量；仅latency的log上标0/1约5.98pt，普通刻度与标签8.54pt。所有图为PDF/SVG与320dpi PNG，PDF图未栅格化。最终排版审查对象是整页嵌入图，不只独立放大PNG。

统计图主线已可读，Supplement仍采用单图单页，部分小图页留白较多；定刊后可并排排版。S4共同起报有十面板，是最密集图，已保留可读字体与单独Qcells解释图；可进一步人工调整美观，不改变数据或筛选。历史完整表占较多页，按要求后置，未缩字强行限页。

|当前图号|源文件名|证据与已检查改动|剩余问题|
|---|---|---|---|
'''
for r in records:
 for suffix in ['.pdf','.svg','.png','_data.csv','_caption.txt','_alt.txt']:assert (F/(r['name']+suffix)).stat().st_size>0
 text+=f"|{mapping[r['name']]}|{r['name']}|{notes[r['name']]} 对应同名前缀_data.csv；build_figures.py/additional_figures.py生成。|定刊后可微调间距，未发现裁切/遮挡；科学限制见报告|\n"
text+='\nFIGURE_OVERVIEW.png与BEFORE_AFTER_1—5.png提供全图及旧新对照；新增图没有旧版时明确标注。FIGURE_NUMBER_MAP.csv是源文件到新编号映射。每图caption与alt独立提供。\n'
(ROOT/'FIGURE_QA.md').write_text(text,encoding='utf8');(F/'FIGURE_QA.md').write_text(text,encoding='utf8')
(P/'REVIEW.md').write_text('''# 当前诊断修订审核（2026-09-09）

请以根目录PROJECT_REPORT_CN.md、REVIEW_RESPONSE_MATRIX.md、新scheme_A_diagnostic_revision三个诊断报告为当前事实。P01实际矩阵诊断和统一13点alpha；P02三split负标签/窗口选择；P03五系统原Ridge配对区间均已执行。764冻结文件SHA/size/mtime不变。旧43/17/10和10432审计是历史，未冒称本轮重跑。

主文12页、Supplement43页，6/14图、5/18表；摘要203个文档化词法token。详细分项计数WORD_COUNTS.json不使用PDF全字数替代正文。全部55页渲染查看，图字/引用/字体/越界见FIGURE_QA.md。原始神经训练未发生，原checkpoint/预测未改写。

剩余科学问题：Alice有限负标签规则的提供方依据、额外时间/季节支持、扩展网格效果尚未补同类时间区间。原网格时间区间已完整，不能套给新网格。CAS官方分区需机构核实。作者声明/发布/投稿未代办。
''',encoding='utf8')
print('PDF and 20-figure QA records written')

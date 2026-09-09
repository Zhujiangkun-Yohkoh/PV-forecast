"""Build traceable summary tables and reports from the new formal results."""
from pathlib import Path
import json,sys
import pandas as pd
import numpy as np
H=Path(__file__).resolve().parent;R=H/'results'
def table(d):
 def f(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
 return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
def write(name,t):(H/name).write_text(t,encoding='utf8')
metrics=pd.concat([pd.read_csv(R/(s+'_metrics.csv')) for s in ['Sanyo','Hanwha','Qcells']]);metrics[metrics.status=='accepted'].to_csv(R/'Alice_accepted_metrics.csv',index=False);metrics[metrics.status!='accepted'].to_csv(R/'Alice_diagnostic_metrics.csv',index=False)
intervals=pd.concat([pd.read_csv(R/(s+'_intervals.csv')) for s in ['Sanyo','Hanwha','Qcells']]);intervals.to_csv(R/'Alice_accepted_intervals.csv',index=False)
# Summary means are means of actual per-seed metrics, not errors of ensemble predictions.
allm=[metrics];alle=[intervals]
for s in ['YULARA_COMBINED','NIST_GROUND']:
 for year in ['2017','2018']:
  allm.append(pd.read_csv(R/f'{s}_{year}_metrics.csv'));alle.append(pd.read_csv(R/f'{s}_{year}_intervals.csv'))
allm=pd.concat(allm);alle=pd.concat(alle);allm['period']=allm['period'].astype(str);alle['period']=alle['period'].astype(str);summary=[]
keys=['site','period','support','horizon','scope']
for key,g in allm[allm.status=='accepted'].groupby(keys):
 for name in ['Original A','Original B','Expanded A','Expanded B','Daily','Last-value','Inverted mean']:
  q=g[g.method.isin(['Inverted42','Inverted43','Inverted44'])] if name=='Inverted mean' else g[g.method==name]
  if len(q)!=(3 if name=='Inverted mean' else 1):continue
  summary.append(dict(zip(keys,key),method=name,RMSE=q.RMSE.mean(),RMSE_seed_SD=q.RMSE.std(ddof=1) if len(q)==3 else np.nan,MAE=q.MAE.mean(),bias=q.bias.mean(),SSE=q.SSE.mean(),points=int(q.points.iloc[0]),seed_count=len(q) if name=='Inverted mean' else 0))
summary=pd.DataFrame(summary);summary.to_csv(R/'accepted_summary.csv',index=False)
external=summary[summary.period.isin(['2017','2018'])];external.to_csv(R/'four_cell_summary.csv',index=False);alle[alle.period.isin(['2017','2018'])].to_csv(R/'four_cell_intervals.csv',index=False)
decomp=[]
for key,g in external.groupby(['site','period','support','horizon']):
 fullN=int(g[g.scope=='full'].points.iloc[0]);daily=g[g.method=='Daily'].set_index('scope')
 for name in ['Inverted mean','Original B','Expanded B']:
  for scope in ['full','power-active','low-power']:
   q=g[(g.method==name)&(g.scope==scope)].iloc[0];delta=(q.SSE-daily.loc[scope,'SSE'])/fullN;decomp.append(dict(zip(['site','period','support','horizon'],key),method=name,scope=scope,points=int(q.points),full_points=fullN,weighted_MSE_delta=delta))
decomp=pd.DataFrame(decomp);decomp.to_csv(R/'four_cell_MSE_decomposition.csv',index=False)
status=pd.read_csv(R/'Alice_method_status.csv');diff=pd.read_csv(R/'Alice_prediction_differences.csv');impact=pd.read_csv(R/'Alice_metric_impact.csv');ranks=pd.read_csv(R/'Alice_rank_impact.csv');probe=pd.read_csv(R/'batch_probe.csv');full=diff[diff.scope=='all_outputs'];maxrow=full.loc[full.max_W.idxmax()];direction_changes=int((np.sign(impact.old_effect_vs_Daily)!=np.sign(impact.new_effect_vs_Daily)).sum());rankchanges=int((ranks.old_rank!=ranks.new_rank).sum())
validation=dict(original_tolerance_pass=int(status.passed_original_tolerance.sum()),original_tolerance_unresolved=int((~status.passed_original_tolerance).sum()),maximum_point_difference_W=float(full.max_W.max()),maximum_prediction_difference_RMS_W=float(full.difference_RMS_W.max()),maximum_RMSE_change_W=float(impact.delta_RMSE_W.abs().max()),maximum_MAE_change_W=float(impact.delta_MAE_W.abs().max()),maximum_bias_change_W=float(impact.delta_bias_W.abs().max()),historical_support_rank_changes=rankchanges,Daily_effect_sign_changes=direction_changes,interpretation='metric sensitivity on checked support, not proof of historical input identity or a bound on new origins')
(R/'REPLAY_IMPACT_SUMMARY.json').write_text(json.dumps(validation,indent=2),encoding='utf8')
write('ALICE_REPLAY_IMPACT_REPORT.md',f'''# Alice回放差异与指标影响

基线89e494f；36组原起报与上一轮保存的新前向数组逐元素核对。原rtol=atol=2e-5不变：{validation['original_tolerance_pass']}通过、{validation['original_tolerance_unresolved']}未通过。TCN与Recurrent各9组未通过，Sanyo Inverted43/44未通过；其余16组通过。不是checkpoint损坏或模型失效判定。

最大单点差为{maxrow.max_W:.9f} W（{maxrow.max_kW:.12f} kW），Qcells TCN44，{maxrow.max_origin}，提前{maxrow.max_lead_min}分钟。其全输出预测差RMS为{maxrow.difference_RMS_W:.9f} W。逐组输出包含完全相同比例、超过容差点数/比例/起报、median/P95/P99/max、预测差RMS，以及full/active/low三个目标范围。CSV同时列kW与W。数值全输出统计包含原不可评分目标位置；指标影响仅在真实有效且Daily匹配的同一标签/掩码上计算。

在1h/12h、full/active/low各原支持中，RMSE变化最大{validation['maximum_RMSE_change_W']:.9f} W，MAE变化最大{validation['maximum_MAE_change_W']:.9f} W，bias变化最大{validation['maximum_bias_change_W']:.9f} W。已比较的四模型均值排名改变{rankchanges}项，相对Daily效应符号改变{direction_changes}项。未重新定义或替换旧指标。

界限：同一支持的 |RMSE(new)-RMSE(old)| <= RMS(new-old)，|MAE(new)-MAE(old)| <= mean|new-old|；所有216个指标影响行均满足。界限不覆盖新增起报，也不证明历史输入状态相同。参考RMSE接近零时，skill的相对变化可被放大；CSV保留old/new skill与Daily分母，不使用epsilon。

## 代表诊断（固定5组，未重复CPU/parser搜索）
'''+table(probe[['site','model','seed','mode','batch_size','max_difference_W','passes_original_tolerance']])+'''

原/混合256批中被选中的Alice极值轨迹数值相同，换回原批次未解决这些Alice差异；singleton可变化，不能将其视为真值。最后不完整批次的选定轨迹则可能通过，这说明不能由单个批次外推全部起报。

五组checkpoint均strict state_dict加载，无缺失/额外键，eval模式、float32、72×17输入和目标逆变换已核对。输入通道、KNN拟合矩阵维度、scaler已拟合数值、权重键与参数量见REPRESENTATIVE_IDENTITY.json。处理器是后来保存的Train-only Ridge处理器；没有单独原神经KNN/IF/scaler对象或原输入数组可直接比对。已保存IF在冻结增强路径中调用，但无法与未保存的历史IF对象逐树比较。checkpoint本身只含epoch/state_dict/validation_global_mse；run_summary为运行记录列表，不提供完整历史cuDNN/backend快照。

当前Python/PyTorch/CUDA/cuDNN/GPU及TF32配置见CURRENT_FORWARD_ENVIRONMENT.json。实际模型源码为基线89e494f中的冻结run_corrected_benchmark.py，本次未修改；保护清单提供内容校验。旧parser/CPU结果保留为历史诊断，本次不重复。当前证据支持“已评分指标稳定，但历史数值状态未完全恢复”，不支持把GPU后端或输入重建断言为唯一原因。

## 按比较推进
Hanwha和Qcells完整三个Inverted种子可用于接受的主比较。Sanyo仅Inverted42可单独报告；不补齐假想三seed均值。三个系统的Ridge及参考均接受；Joint-patch为接受的次要结果。TCN/Recurrent和Sanyo43/44只在独立diagnostic指标CSV保留，不进入正式新支持均值或排名。
''')
primary=alle[(alle.horizon==144)&(alle.scope=='full')&(alle.block_hours==48)&(alle.reference=='Daily')&alle.method.isin(['Original B','Expanded B','Inverted mean'])]
al=primary[primary.site.isin(['Hanwha','Qcells'])&primary.support.isin(['Original','S1','S2'])];ex=primary[primary.period.isin(['2017','2018'])]
write('REPORT_CN.md','''# Scheme A有限收尾报告

## 最终状态：B——收窄新增核心范围后，可进入投稿准备

核心证据链已完成：参考历史信息、Alice已验证模型的目标支持敏感性、两外部场址相同支持规则下的跨时期四格对照。Sanyo新增支持不报告完整三seed Inverted均值；TCN/Recurrent不作为新支持核心排名。历史四模型结果保持原证据身份。该收窄不破坏论文完整性，也不要求次要模型逐点一致才推进准备。作者签核、期刊格式与公开范围仍待对应决策；没有实际投稿或PR描述对外更新。

## R1：容差与科学影响分开

'''+f'''36组仍为16通过/20未通过原容差。最大单点差{validation['maximum_point_difference_W']:.6f} W，最大预测差RMS {validation['maximum_prediction_difference_RMS_W']:.6f} W；已评分RMSE变化最大{validation['maximum_RMSE_change_W']:.9f} W，没有改变这些原支持下的排名或Daily效应方向。全部逐组、范围、单位和界限在ALICE_REPLAY_IMPACT_REPORT.md及CSV；有限代表GPU探针未恢复历史数值状态，不把CPU当真值，也未放宽容差。

'''+'''## R2：已验证Alice支持比较

主表为12h/full，效应=方法RMSE−Daily RMSE，负数有利于方法。48h块、2000次、同抽样；1h、active/low、24/72h与新增起报分别保存。

'''+table(al[['site','support','method','rmse','reference_rmse','effect_kW','ci_low_kW','ci_high_kW']])+'''

Qcells的Inverted误差在S1增大，Hanwha略降，模型权重没有变化。二者相对Daily和Ridge B更高、相对Last-value更低的方向保留；新增起报子集独立报告。Ridge原/扩展可在某些支持上换序，不能将原支持上的Qcells扩展恶化推广到全部恢复支持。S1与S2只改变可评分目标，未改历史输入和Daily/Last-value的非负规则。完整目标对、唯一目标时间戳数（不是独立观测数）及active比例见support CSV。新增起报按每个horizon自己的原支持补集定义。

训练支持仍有选择偏移。本次前向回答“冻结模型在更广评分总体上如何比较”，不回答“重新训练后会怎样”。当前主张不需要启动重训；若未来要研究训练选择效应，最小另立实验可限于Qcells Inverted三个seed、同输入/日期/预算、逐目标masked训练与历史整窗规则的对照。这不是本轮完成事项，也不启动全部60次重训。

## R3：相同规则的四格对照

'''+table(ex[['site','period','support','method','rmse','reference_rmse','effect_kW','ci_low_kW','ci_high_kW']])+'''

同一时期：Yulara2017全候选未来标签均有限，因此两种支持数值完全相同；2018少量缺失使两格略有差异。NIST完整/逐目标的样本数和绝对误差差异更大，但Inverted相对Daily在2017正、2018负的样本均值方向在两种支持下均保留。2017区间跨零，不能声称已证明稳定劣势；2018区间为负也只条件于观察时期与冻结预测。

Yulara原B由2017高误差转为2018低于Daily；扩展B由2017接近Daily转为2018高于Daily。两网格不根据2018成绩重新选择。跨期不做日期一一配对，不解释为纯季节因果效应。

统一候选2017-10-01 00:00至12-31 23:55、2018-04-01 00:00至06-30 23:55，均有历史/目标缓冲。2017的Jan1只是12月末目标缓冲，来源为已有官方NIST年ZIP/已有Yulara数据，不新增起报月份。归档从06:00开始并截断期末，原数字保留；boundary_audit与archive_metric_replay明确对照。NIST固定−05:00，无DST；Yulara保持提供方坐标/+5min可用规则。

2017扩展候选初次混合批次中Yulara Inverted43最大差0.03624 W未过原容差，原批次通过。正式四格采用明确的推理分组：原起报按原批次重新计算，新增起报单独前向，同一权重和处理器；没有把旧保存预测拼进新预测。6个原2017神经回放、8个Ridge回放通过。新2017全候选数组单独保存，2018数组直接复用。

## 实际验证与未验证边界

本次：36组保存新旧数组差异及216行指标影响；5组代表GPU批次/strict加载检查；Alice1836总指标行中的诊断行独立标识，全部新分析共有1836行逐点指标和114个支持组独立复算；6120行时间块效应/区间由不导入生产interval函数的块计数矩阵实现复算。检查数量不是独立实验数量。最终保护清单、PDF和归档结果另附实际报告。

原M1-R/M2整套测试与完整神经训练复现未重跑。原Alice神经处理器未单独保存的限制仍存在，不能由很小指标差恢复它的身份。无神经训练、处理器fit、alpha搜索、新模型或起报月份。新脚本不写死失败20项作为必须满足的新条件；旧verify_light仍是历史快照。

## 投稿与交付

可进入缩窄后的投稿准备，先导师审核新的支持与四格主图。Renewable Energy保持有条件冲刺、Solar Energy为直接主题匹配选项；既有CAS年份/大类/小类未获得权威条目，本次未重新查询，继续待机构核实。不是按JCR推断CAS，也不保证录用。PR_DESCRIPTION_DRAFT.md仅本地，不对外写入描述。

三个包保留全量本地证据/轻量独立复核/当前图件；新旧结果目录分开。打包CRC/试解压只验证归档和轻量算术，不等于神经训练复现。最终提交与包校验信息由交付sidecar记录。
''')
print(validation)

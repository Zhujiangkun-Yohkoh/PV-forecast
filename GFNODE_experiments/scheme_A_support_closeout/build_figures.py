"""Only redraw accepted Alice sensitivity and cross-period/support effects."""
from pathlib import Path
import shutil,json,csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;P=H.parents[1]/'manuscript/clean_pv_benchmark';O=P/'closeout_figures';O.mkdir(exist_ok=True)
for f in (P/'fixed_period_figures').glob('*'):
 if f.is_file() and not (O/f.name).exists():shutil.copy2(f,O/f.name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def save(fig,stem,data,caption,alt):
 for ext in ['pdf','svg','png']:fig.savefig(O/(stem+'.'+ext),dpi=320)
 plt.close(fig);data.to_csv(O/(stem+'_data.csv'),index=False);(O/(stem+'_caption.txt')).write_text(caption,encoding='utf8');(O/(stem+'_alt.txt')).write_text(alt,encoding='utf8')
q=pd.read_csv(H/'results/Alice_accepted_intervals.csv');q=q[(q.block_hours==48)&(q.method=='Inverted mean')&q.reference.isin(['Daily','Expanded B'])&q.scope.eq('full')&q.support.isin(['Original','S1','S2'])]
fig,axs=plt.subplots(2,2,figsize=(7.1,5.0),layout='constrained')
for i,site in enumerate(['Hanwha','Qcells']):
 for j,h in enumerate([12,144]):
  ax=axs[i,j]
  for k,support in enumerate(['Original','S1','S2']):
   for ref,dy,col,marker in [('Daily',-.12,'#0072B2','o'),('Expanded B',.12,'#D55E00','s')]:
    r=q[(q.site==site)&(q.horizon==h)&(q.support==support)&(q.reference==ref)].iloc[0];ax.plot([r.ci_low_kW,r.ci_high_kW],[k+dy,k+dy],c=col);ax.scatter(r.effect_kW,k+dy,c=col,marker=marker,s=20)
  ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(3),['Original','S1: nonnegative','S2: finite']);ax.set_ylim(2.5,-.5);ax.set_title(f'({chr(97+i*2+j)}) {site} / {h/12:g} h / full');ax.set_xlabel('Inverted RMSE − reference (kW)');ax.grid(axis='x',alpha=.2)
fig.legend(handles=[Line2D([],[],c='#0072B2',marker='o',label='Daily'),Line2D([],[],c='#D55E00',marker='s',label='Expanded Ridge B')],loc='outside upper center',ncol=2)
save(fig,'fig9_alice_support',q,'Accepted three-seed Inverted comparisons at Hanwha and Qcells on full targets. Original is prefix-specific complete nonnegative support; S1 scores individual nonnegative targets and S2 finite raw targets. Inputs and Daily/Last-value rules remain frozen. Points average seed-specific RMSE differences; bars are 95% paired 48 h block intervals, not seed SD. Negative favors Inverted. No Sanyo three-seed mean is included.','The 12 h Inverted-minus-Daily and Inverted-minus-expanded-B differences remain positive after restoring origins; one-hour comparisons and their intervals are retained without assuming the same selected target population.')
q=pd.read_csv(H/'results/four_cell_intervals.csv');q=q[q.block_hours.eq(48)&q.reference.eq('Daily')&q.method.isin(['Original B','Expanded B','Inverted mean'])&q.scope.isin(['full','power-active','low-power'])]
fig,axs=plt.subplots(2,3,figsize=(7.1,5.7),layout='constrained')
style={'Original B':('#009E73','v',-.18),'Expanded B':('#D55E00','s',0),'Inverted mean':('#0072B2','o',.18)}
for i,site in enumerate(['YULARA_COMBINED','NIST_GROUND']):
 for j,scope in enumerate(['full','power-active','low-power']):
  ax=axs[i,j]
  for k,(year,support) in enumerate([(2017,'complete_H144'),(2017,'pointwise_finite'),(2018,'complete_H144'),(2018,'pointwise_finite')]):
   for method,(col,mark,dy) in style.items():
    r=q[(q.site==site)&(q.period==year)&(q.support==support)&(q.scope==scope)&(q.method==method)].iloc[0];ax.plot([r.ci_low_kW,r.ci_high_kW],[k+dy,k+dy],c=col,lw=1);ax.scatter(r.effect_kW,k+dy,c=col,marker=mark,s=18)
  ax.axvline(0,c='black',lw=.7);ax.axhline(1.5,c='#999',lw=.5,ls='--');ax.set_yticks(range(4),['2017 complete','2017 pointwise','2018 complete','2018 pointwise']);ax.set_ylim(3.6,-.6);ax.set_xlabel('RMSE − Daily (kW)');ax.set_title(f'({chr(97+i*3+j)}) '+('Yulara' if i==0 else 'NIST')+' /\n'+scope);ax.grid(axis='x',alpha=.2)
fig.legend(handles=[Line2D([],[],c=v[0],marker=v[1],label=k.replace(' mean','')) for k,v in style.items()],loc='outside upper center',ncol=3)
save(fig,'fig8_fixed_period',q,'Four-cell same-site comparison of cumulative 12 h forecasts: October–December 2017 and April–June 2018, each on complete-H144 and pointwise-finite target support. Candidates use unified historical and target buffers; Daily and origin validity are common within a cell. Points and 95% 48 h block intervals are paired within periods, not between different dates. Inverted is the mean of three seed-specific effects. Both saved Ridge grids remain labeled. Negative favors the named method; panel scales differ.','NIST Inverted-minus-Daily has positive 2017 point estimates with intervals crossing zero and negative 2018 intervals under both supports. Yulara original and expanded Ridge B change ordering between periods. Support affects magnitudes but does not explain away these cross-period directions.')
records={}
for f in O.glob('*_caption.txt'):
 stem=f.name[:-12];records[stem]={'caption':f.read_text(encoding='utf8'),'alt':(O/(stem+'_alt.txt')).read_text(encoding='utf8')}
(O/'FIGURE_CAPTIONS_AND_ALT.json').write_text(json.dumps(records,indent=2),encoding='utf8')
old=pd.read_csv(O/'FIGURE_NUMBER_MAP.csv');mapping={'fig9_alice_support':'3','fig7_qcells_selection':'S15','fig2_alice_references':'S16','fig3_external_skills':'S17','fig8_fixed_period':'5'}
for k,v in mapping.items():
 if (old.source_figure==k).any():old.loc[old.source_figure==k,'current_number']=v
 else:old=pd.concat([old,pd.DataFrame([{'source_figure':k,'current_number':v}])])
old.to_csv(O/'FIGURE_NUMBER_MAP.csv',index=False)
print('22 groups; two plots redrawn, twenty retained')

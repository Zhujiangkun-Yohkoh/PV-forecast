"""Targeted revision of existing vector figures; frozen CSVs are the inputs."""
from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PAPER=ROOT/'manuscript/clean_pv_benchmark';OUT=PAPER/'fixed_period_figures';OUT.mkdir(exist_ok=True)
for p in (PAPER/'diagnostic_figures').glob('*'):
 if p.is_file() and not (OUT/p.name).exists():shutil.copy2(p,OUT/p.name)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
records=json.loads((OUT/'FIGURE_CAPTIONS_AND_ALT.json').read_text(encoding='utf8'))
def save(fig,stem,data,caption,alt):
 for suffix in ['pdf','svg','png']:fig.savefig(OUT/(stem+'.'+suffix),dpi=320)
 data.to_csv(OUT/(stem+'_data.csv'),index=False);(OUT/(stem+'_caption.txt')).write_text(caption,encoding='utf8');(OUT/(stem+'_alt.txt')).write_text(alt,encoding='utf8');plt.close(fig)
 # Keep independent per-figure text authoritative; inherited JSON formats vary.
records_new={}
q=pd.read_csv(HERE/'results/expanded_paired_intervals.csv');q=q[(q.block_hours==48)&q.reference.isin(['Expanded A','Daily','Inverted mean'])&q.scope.isin(['full','power-active'])&(q.method=='Expanded B')]
fig,axs=plt.subplots(5,2,figsize=(7.1,8.2),layout='constrained')
for i,site in enumerate(['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):
 for j,scope in enumerate(['full','power-active']):
  ax=axs[i,j];z=q[(q.site==site)&(q.scope==scope)].set_index('reference').loc[['Expanded A','Daily','Inverted mean']]
  for k,(ref,r) in enumerate(z.iterrows()):ax.plot([r.ci_low_kW,r.ci_high_kW],[k,k],c=['#009E73','#777777','#0072B2'][k]);ax.scatter(r.effect_kW,k,marker=['v','x','o'][k],c=['#009E73','#777777','#0072B2'][k])
  ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(3),['B − A','B − Daily','B − Inverted']);ax.set_ylim(2.6,-.6);ax.set_title(f'({chr(97+i*2+j)}) '+{'YULARA_COMBINED':'Yulara','NIST_GROUND':'NIST'}.get(site,site)+' / '+scope);ax.set_xlabel('RMSE difference (kW)');ax.grid(axis='x',alpha=.2)
save(fig,'fig6_ridge_intervals',q,'Expanded-grid Ridge B minus each reference over cumulative 12 h windows. Negative differences favor B. Bars are 95% paired 48 h origin-block intervals, 2000 draws; Inverted contrasts average three seed-specific effects, not predictions. Identical complete-H144 and Daily-finite target support is used within each system/scope. Scales differ between panels. Original-grid intervals remain in the supplementary evidence.','All five systems are shown. Yulara expanded B has intervals crossing zero against Daily and Inverted; Qcells deterioration relative to original B is preserved in the grid sensitivity data.')
q=pd.read_csv(HERE/'results/fixed_2018_primary_intervals.csv');q=q[(q.horizon==144)&(q.block_hours==48)&(q.reference=='Daily')]
fig,axs=plt.subplots(2,3,figsize=(7.1,5.4),layout='constrained')
for i,site in enumerate(['YULARA_COMBINED','NIST_GROUND']):
 for j,scope in enumerate(['full','power-active','low-power']):
  ax=axs[i,j];z=q[(q.site==site)&(q.scope==scope)].set_index('method')
  for k,(name,col,mark) in enumerate([('Original B','#009E73','v'),('Expanded B','#CC79A7','s'),('Inverted mean','#0072B2','o')]):
   r=z.loc[name];ax.plot([r.ci_low_kW,r.ci_high_kW],[k,k],color=col);ax.scatter(r.effect_kW,k,c=col,marker=mark)
  ax.axvline(0,c='black',lw=.8);ax.set_yticks(range(3),['Original B','Expanded B','Inverted']);ax.set_ylim(2.6,-.6);ax.set_title(f'({chr(97+i*3+j)}) '+('Yulara' if i==0 else 'NIST')+'\n'+scope);ax.set_xlabel('RMSE − Daily (kW)');ax.grid(axis='x',alpha=.2)
save(fig,'fig8_fixed_period',q,'Same-site next-year evaluation, origins fixed to 1 April–30 June 2018; cumulative 12 h windows. All methods use the same finite target/Last-value/Daily intersection in each panel. Original 2017 transformations, weights, and both saved Ridge grids are unchanged. Points and 95% paired 48 h block intervals condition on frozen predictions. Negative differences favor the named method over Daily. Panel scales differ.','The next-year period changes comparisons: original and expanded Ridge grids remain distinct and all adverse effects are shown. Neither seasons nor geographic sites are treated as exchangeable repeats.')
# S1 retains ranges but states the rare-mask denominator visibly.
old=ROOT/'GFNODE_experiments/scheme_A_diagnostic_revision/results';dist=pd.read_csv(old/'ridge_distributions.csv');feat=pd.read_csv(old/'ridge_features.csv')
fig,axs=plt.subplots(1,2,figsize=(7.1,3.8),layout='constrained')
for k,model in enumerate(['Ridge','Ridge+day']):
 z=dist[(dist.model==model)&(dist.kind=='prediction')].set_index('split').loc[['train','validation','test']];xx=np.arange(3)+(k-.5)*.16;col=['#009E73','#CC79A7'][k];axs[0].vlines(xx,z['min'],z['max'],color=col);axs[0].vlines(xx,z.p01,z.p99,color=col,lw=5);axs[0].scatter(xx,z['median'],c=col,marker=['v','s'][k],label=['Original A','Original B'][k])
axs[0].axhline(109.903154,c='black',ls=':');axs[0].set_xticks(range(3),['Train','Val','Test']);axs[0].set_ylabel('Predicted power (kW)');axs[0].set_title('(a) Full ranges and 1–99%');axs[0].legend()
z=feat[(feat.model=='Ridge')&(feat.feature=='lag0:temperature_missing')].set_index('split').loc[['train','validation','test']];axs[1].bar(range(3),z.raw_mean*100,color=['#009E73','#CC79A7','#0072B2'],hatch=['','//','..']);axs[1].set_xticks(range(3),['Train','Val','Test']);axs[1].set_ylabel('Missing temperature at origin (%)');axs[1].set_title('(b) 4 / 69,333 Train origins\n0.0058%; mask scale 0.007595')
save(fig,'figS11_yulara_diagnostic',pd.concat([dist,z.reset_index()],ignore_index=True),'Original-grid Yulara prediction ranges (thin), 1st–99th percentiles (thick) and medians (markers), over saved candidate trajectories. The dotted line is the Train target maximum, not clipping. Missing temperature at origin occurs in 4/69333 Train origins (0.0058%). These diagnostics support a missing-pattern extrapolation explanation, not an identified unique cause.','Original A and B adverse ranges remain visible. The different denominators for candidate prediction distributions and headline matched complete windows are explicit.')
# Unequal-width power bins are categories, with no connecting interpolation.
hours=pd.read_csv(old/'qcells_hours.csv');hist=pd.read_csv(old/'qcells_power_histogram.csv');fig,axs=plt.subplots(3,2,figsize=(7.1,7.2),layout='constrained')
for i,split in enumerate(['train','validation','test']):
 for k,(label,color,mark) in enumerate([('H12_support','#0072B2','o'),('common_H144','#009E73','s'),('H12_removed_by_H144','#D55E00','^')]):
  z=hours[(hours.split==split)&(hours.support==label)];axs[i,0].plot(z.hour,z.origins/z.origins.sum()*100,color=color,ls=['-','--',':'][k]);z=hist[(hist.split==split)&(hist.h==12)&(hist.support==label)];z=z[(z.bin_right>0)&(z.bin_left<6)];axs[i,1].scatter(np.arange(len(z))+(k-1)*.15,z.target_pairs/z.target_pairs.sum()*100,c=color,marker=mark,s=20)
 axs[i,0].set_title(f'({chr(97+i*2)}) '+split+' origin hours');axs[i,0].set_xlabel('Origin hour');axs[i,0].set_ylabel('Origins (%)');axs[i,1].set_title(f'({chr(98+i*2)}) '+split+' / 1 h');axs[i,1].set_ylabel('Target pairs (%)');axs[i,1].set_xticks(range(len(z)),['0–τ','τ–0.1','0.1–0.5','0.5–1','1–2','2–3','3–4','4–5','5–6'],rotation=45);axs[i,1].set_xlabel('Power categories (kW; τ = Train threshold)')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],c=c,marker=m,ls=l,label=n) for n,c,m,l in [('1 h support','#0072B2','o','-'),('Complete 12 h','#009E73','s','--'),('Excluded','#D55E00','^',':')]],loc='outside upper center',ncol=3)
save(fig,'figS10_qcells_support',pd.concat([hours,hist],ignore_index=True),'Historical Qcells support. Unequal-width power bins are categorical points, not a continuous density. Each power series is normalized by its target-pair count in the displayed bins. Origin-hour series use all origins in the named support. Repeated target times remain overlapping forecast pairs.','Three split rows compare one-hour, complete-twelve-hour and removed support without interpreting unequal bin widths as density.')
# A visible label for the inherited Original-grid S13; preserve its data.
data=pd.read_csv(OUT/'figS8_ridge_data.csv');fig,axs=plt.subplots(5,1,figsize=(7.1,8.8),layout='constrained')
colors={'Inverted':'#0072B2','Daily':'#777777','Ridge':'#009E73','Ridge+day':'#CC79A7','Recurrent':'#D55E00','Joint-patch':'#E69F00','TCN':'#009E73','Last-value':'#777777'}
for i,(ax,site) in enumerate(zip(axs,['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND'])):
 for name,mark,ls in [('Inverted','o','-'),('Daily','x',':'),('Ridge','v','--'),('Ridge+day','s','-.')]:
  z=data[(data.site==site)&(data.model==name)].groupby('h').RMSE.agg(['mean','std']);ax.errorbar(z.index/12,z['mean'],yerr=z['std'].fillna(0),color=colors[name],marker=mark,ls=ls,label=name)
 ax.set_title(f'({chr(97+i)}) '+{'YULARA_COMBINED':'Yulara','NIST_GROUND':'NIST'}.get(site,site));ax.set_xticks([1,4,8,12]);ax.set_ylabel('RMSE (kW)');ax.set_ylim(bottom=0)
axs[-1].set_xlabel('Cumulative window (h)');fig.suptitle('ORIGINAL GRID — preserved comparison');axs[0].legend(ncol=4,fontsize=10)
save(fig,'figS8_ridge',data,'ORIGINAL GRID. Recent-history Ridge and Ridge plus available previous-day trajectory, with original Validation-selected alphas. Daily-matched full targets use the historical horizon-specific support. Ridge is deterministic; neural bars are seed SD, not temporal intervals. Expanded-grid results are separately labeled.','All original-grid outcomes, including large Yulara errors, are retained without presenting them as the final expanded linear reference.')
data=pd.read_csv(OUT/'figS2_common_origins_data.csv');fig,axs=plt.subplots(5,2,figsize=(7.1,9),layout='constrained')
for i,site in enumerate(['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):
 for j in [0,1]:
  ax=axs[i,j]
  for model in (['Inverted','Last-value'] if j==0 else ['Recurrent','Inverted','Joint-patch','TCN']):
   for support,ls in [('horizon_specific','-'),('common_H144','--')]:
    z=data[(data.site==site)&(data.model==model)&(data.support==support)].groupby('h').RMSE.agg(['mean','std']);ax.errorbar(z.index/12,z['mean'],yerr=z['std'].fillna(0),color=colors[model],ls=ls,marker='o',markersize=2,lw=.8)
  ax.set_title(f'({chr(97+i*2+j)}) '+{'YULARA_COMBINED':'Yulara','NIST_GROUND':'NIST'}.get(site,site)+(' / full' if j==0 else ' / neural'));ax.set_ylabel('RMSE (kW)');ax.set_ylim(bottom=0);ax.set_xticks([1,4,8,12]);ax.set_xlabel('Cumulative window (h)')
handles=[Line2D([],[],color=colors[k],label=k) for k in ['Recurrent','Inverted','Joint-patch','TCN','Last-value']]+[Line2D([],[],color='black',ls='-',label='Horizon-specific support'),Line2D([],[],color='black',ls='--',label='Common 12 h origins')]
fig.legend(handles=handles,loc='outside upper center',ncol=3,fontsize=10)
save(fig,'figS2_common_origins',data,'Historical horizon-specific versus common complete-twelve-hour origin support, full target range. Bars show seed SD. Separate reference and neural panels retain full error ranges and readable details. Common-origin selection does not isolate window length from the selected population.','All internal support abbreviations are expanded. Qcells common origins select a different power distribution; this is not the new S1/S2 restored-origin error comparison.')
for p in OUT.glob('*_caption.txt'):
 stem=p.name[:-12];records_new[stem]={'caption':p.read_text(encoding='utf8'),'alt':(OUT/(stem+'_alt.txt')).read_text(encoding='utf8')}
(OUT/'FIGURE_CAPTIONS_AND_ALT.json').write_text(json.dumps(records_new,indent=2),encoding='utf8')
print('Current complete figure directory:',len(list(OUT.glob('fig*.pdf'))),'groups; targeted changes only')

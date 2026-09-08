"""Portable statistical figures from review CSVs. Matplotlib bundled DejaVu fonts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'manuscript/clean_pv_benchmark/review_figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':320,'axes.spines.top':False,'axes.spines.right':False})
C={'Inverted':'#0072B2','Recurrent':'#D55E00','Joint-patch':'#009E73','TCN':'#CC79A7','Daily':'#333333','Last-value':'#888888','Ridge':'#E69F00','Ridge+day':'#56B4E9'}
MARK={'Inverted':'o','Recurrent':'s','Joint-patch':'^','TCN':'D','Daily':'x','Last-value':'+','Ridge':'v','Ridge+day':'P'}
D=pd.read_csv(HERE/'results/metrics_decomposition.csv');P=D[(D.support=='horizon_specific')&(D.scope!='low-power')];G=P.groupby(['site','model','h','analysis','scope']).agg(RMSE=('RMSE','mean'),sd=('RMSE','std')).reset_index();records=[]
def save(fig,name,data,caption,alt):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=320,bbox_inches='tight');plt.close(fig)
 data.to_csv(OUT/(name+'_data.csv'),index=False);(OUT/(name+'_caption.txt')).write_text(caption,encoding='utf8');(OUT/(name+'_alt.txt')).write_text(alt,encoding='utf8');records.append(dict(name=name,caption=caption,alt=alt))
def meanplot(ax,df,models,xcol='h',ycol='RMSE',sd=True):
 for model in models:
  z=df[df.model==model].groupby(xcol)[ycol].agg(['mean','std']).sort_index()
  xx=z.index.to_numpy()/12 if xcol=='h' else z.index
  ax.errorbar(xx,z['mean'],yerr=z['std'].fillna(0) if sd else None,color=C[model],marker=MARK[model],markersize=4,capsize=2,lw=1,label=model)
 ax.grid(alpha=.2);ax.set_xlabel('Cumulative forecast window (h)' if xcol=='h' else 'Lead (h)')

# Figure 1: no repeated processing columns or pooled fit.
fig,ax=plt.subplots(figsize=(7.1,3.6));ax.axis('off')
ax.text(.5,.96,'Three sites, five independently fitted targets',ha='center',weight='bold',transform=ax.transAxes)
for x,text in [(.17,'Alice Springs\n3 co-located arrays\n17 channels / 36 runs'),(.50,'Yulara\n1 combined system\n7 channels / 12 runs'),(.83,'NIST Ground\n1 array\n7 channels / 12 runs')]:ax.text(x,.74,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#edf2f5',ec='#566573'),transform=ax.transAxes)
ax.text(.5,.49,'Separate Train fits → Validation selection → matched target evaluation',ha='center',transform=ax.transAxes)
ax.plot([.08,.95],[.25,.25],color='black',transform=ax.transAxes)
for x,t in [(.08,'−24 h'),(.38,'−6 h'),(.56,'origin'),(.95,'+12 h')]:ax.plot([x,x],[.23,.27],color='black',transform=ax.transAxes);ax.text(x,.17,t,ha='center',transform=ax.transAxes)
ax.annotate('',xy=(.56,.32),xytext=(.38,.32),xycoords='axes fraction',arrowprops=dict(arrowstyle='<->',color=C['Inverted']));ax.text(.47,.37,'72 history steps',ha='center',transform=ax.transAxes)
ax.annotate('',xy=(.95,.32),xytext=(.56,.32),xycoords='axes fraction',arrowprops=dict(arrowstyle='<->',color=C['Recurrent']));ax.text(.755,.37,'144 future steps',ha='center',transform=ax.transAxes)
ax.text(.5,.02,'Schematic, not to scale. Daily at τ uses τ − 24 h; available by origin.',ha='center',transform=ax.transAxes)
save(fig,'fig1_design',pd.DataFrame({'site':['Alice','Yulara','NIST'],'targets':[3,1,1],'channels':[17,7,7],'runs':[36,12,12]}),'Independent fitting and forecast-time information. Every step is five minutes; cumulative windows use the first 12, 48, 96 or 144 future steps.','Three site boxes feed a common evaluation protocol, not a pooled model. A time axis separates six-hour inputs, origin and twelve-hour targets; Daily uses target minus twenty-four hours.')

# Figure 2 both references on same Daily-valid points.
dat=[]
for site in ['Sanyo','Hanwha','Qcells']:
 for h in [12,48,96,144]:
  for scope in ['full','power-active']:
   q=G[(G.site==site)&(G.h==h)&(G.scope==scope)&(G.analysis=='daily_matched')];env=q[q.model.isin(['Inverted','Recurrent','Joint-patch','TCN'])].RMSE.min()
   for ref in ['Last-value','Daily']:dat.append(dict(site=site,h=h,scope=scope,reference=ref,ratio=env/float(q[q.model==ref].RMSE.iloc[0])))
dat=pd.DataFrame(dat);fig,axs=plt.subplots(1,2,figsize=(7.1,6.3),sharey=True,layout='constrained');labels=[]
for j,ref in enumerate(['Last-value','Daily']):
 q=dat[dat.reference==ref];yy=np.arange(len(q));axs[j].hlines(yy,1,q.ratio,color='#aaaaaa',lw=.8);axs[j].scatter(q.ratio,yy,c=[C['Inverted'] if x<1 else C['Recurrent'] for x in q.ratio],s=19);axs[j].axvline(1,color='black',ls='--',lw=1);axs[j].set_xscale('log');axs[j].set_xlim(.15,12);axs[j].set_xticks([.2,.5,1,2,5,10],labels=['0.2','0.5','1','2','5','10']);axs[j].set_title('Envelope / '+ref);axs[j].set_xlabel('RMSE ratio (log scale)');axs[j].grid(axis='x',alpha=.2)
 labels=[f'{z.site}  {z.h/12:g} h  {"A" if z.scope=="power-active" else "F"}' for z in q.itertuples()]
axs[0].set_yticks(np.arange(24),labels);axs[0].invert_yaxis();fig.suptitle('Post hoc neural envelope — identical Daily-valid targets',fontsize=11)
save(fig,'fig2_alice_references',dat,'Post hoc minimum three-seed mean neural RMSE relative to both references on the same Daily-valid targets. F: full; A: power-active. The envelope is descriptive, not deployable.','Twenty-four Alice comparisons on shared logarithmic axes: both references use the Daily intersection. Ratio one is the comparison boundary; unfavorable ratios remain visible.')

# Figure 3 seed-level skills avoids compressed Alice extreme scale.
dat=[]
for site in ['YULARA_COMBINED','NIST_GROUND']:
 for analysis,ref in [('primary','Last-value'),('daily_matched','Daily')]:
  for scope in ['full','power-active']:
   q=P[(P.site==site)&(P.analysis==analysis)&(P.scope==scope)&(P.model=='Inverted')]
   for z in q.itertuples():
    rr=float(G[(G.site==site)&(G.h==z.h)&(G.analysis==analysis)&(G.scope==scope)&(G.model==ref)].RMSE.iloc[0]);dat.append(dict(site=site,reference=ref,scope=scope,h=z.h,seed=z.seed,skill=100*(1-z.RMSE/rr)))
dat=pd.DataFrame(dat);fig,axs=plt.subplots(2,2,figsize=(7.1,4.7),sharex=True,layout='constrained')
for i,site in enumerate(['YULARA_COMBINED','NIST_GROUND']):
 for j,ref in enumerate(['Last-value','Daily']):
  ax=axs[i,j]
  for scope,ls,marker in [('full','-','o'),('power-active','--','s')]:
   z=dat[(dat.site==site)&(dat.reference==ref)&(dat.scope==scope)];g=z.groupby('h').skill.agg(['mean','std']);ax.errorbar(g.index/12,g['mean'],yerr=g['std'],marker=marker,ls=ls,capsize=3,label=scope,color=C['Inverted'] if scope=='full' else C['Recurrent'])
   ax.scatter(z.h/12,z.skill,s=9,alpha=.4,color=C['Inverted'] if scope=='full' else C['Recurrent'])
  ax.axhline(0,c='black',lw=.7);ax.set_title(f'{"Yulara" if i==0 else "NIST"} vs {ref}');ax.set_ylabel('RMSE skill (%)');ax.set_xticks([1,4,8,12]);ax.grid(alpha=.2)
  if i==1:ax.set_xlabel('Cumulative forecast window (h)')
axs[0,0].legend();save(fig,'fig3_external_skills',dat,'Prespecified external Inverted-variate skill. Dots are individual seeds; bars show sample SD of three seed metrics, not temporal confidence intervals. Each reference uses its own matched support.','Four panels show two external sites and two references, with full and power-active curves. NIST twelve-hour full crosses below zero for Daily; no Alice extreme compresses this scale.')

# Figure 4 relative gaps, not rank magnification.
dat=[]
for z in G[(G.analysis=='primary')&G.site.isin(['YULARA_COMBINED','NIST_GROUND'])&G.model.isin(['Inverted','Recurrent','Joint-patch','TCN'])].itertuples():
 q=G[(G.site==z.site)&(G.h==z.h)&(G.scope==z.scope)&(G.analysis=='primary')&G.model.isin(['Inverted','Recurrent','Joint-patch','TCN'])];dat.append(dict(site=z.site,h=z.h,scope=z.scope,model=z.model,gap=100*(z.RMSE/q.RMSE.min()-1)))
dat=pd.DataFrame(dat);fig,axs=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
for i,site in enumerate(['YULARA_COMBINED','NIST_GROUND']):
 for j,scope in enumerate(['full','power-active']):
  ax=axs[i,j];meanplot(ax,dat[(dat.site==site)&(dat.scope==scope)],['Recurrent','Inverted','Joint-patch','TCN'],ycol='gap',sd=False);ax.set_title(f'{"Yulara" if i==0 else "NIST"}: {scope}');ax.set_ylabel('RMSE above best neural (%)');ax.set_xticks([1,4,8,12]);ax.set_ylim(bottom=0)
axs[0,0].legend(ncol=2);save(fig,'fig4_neural_gaps',dat,'Relative RMSE gaps from the best mean neural implementation in each cumulative window and power scope. Zero identifies the lowest error; small gaps are shown as small gaps rather than enlarged rank differences.','Four model curves show NIST full and power-active winners differing across windows, while Yulara favors Inverted-variate. Axes begin at zero and retain all relative gaps.')

# Figure 5 additive MSE plus transparent fixed-origin trajectory.
q=D[(D.site=='NIST_GROUND')&(D.h==144)&(D.support=='horizon_specific')&(D.analysis=='daily_matched')&D.model.isin(['Inverted','Daily'])]
agg=q.groupby(['model','scope']).mean(numeric_only=True);case=pd.read_csv(HERE/'results/trajectory_cases.csv');case=case[(case.site=='NIST_GROUND')&(case.case=='fixed_month_10')]
fig,axs=plt.subplots(1,2,figsize=(7.1,3.4),layout='constrained')
bottom=np.zeros(2)
for scope,col,hatch in [('power-active',C['Inverted'],''),('low-power',C['Recurrent'],'///')]:
 vals=[agg.loc[(m,scope),'weighted_MSE_contribution'] for m in ['Inverted','Daily']];axs[0].bar(['Inverted','Daily'],vals,bottom=bottom,color=col,hatch=hatch,label=scope);bottom+=vals
axs[0].set_ylabel('Contribution to full MSE (kW²)');axs[0].legend(loc='upper right',fontsize=8);axs[0].set_ylim(0,2600);low=agg.loc[('Daily','low-power'),'points']/agg.loc[('Daily','full'),'points'];axs[0].set_title(f'NIST 12 h; low-power share {low:.2%}')
for field,label,col in [('truth','Observed','black'),('Inverted','Inverted seed42',C['Inverted']),('Daily','Daily',C['Daily'])]:axs[1].plot(case.hours,case[field],label=label,color=col,lw=1,ls='--' if field=='Daily' else '-')
axs[1].set_title('Fixed calendar case: '+str(case.origin.iloc[0])[:16]);axs[1].set_xlabel('Lead from one origin (h)');axs[1].set_ylabel('AC power (kW)');axs[1].legend(fontsize=8)
save(fig,'fig5_decomposition_case',pd.concat([q.assign(panel='decomposition'),case.assign(panel='trajectory')],ignore_index=True),'Left: additive active/low-power contributions to full MSE on NIST twelve-hour Daily-matched targets; neural contributions are averaged over seeds. Right: one seed42 trajectory from the fixed calendar selection, not a rolling reconstruction.','Stacked squared-error contributions show active error savings offset by low-power excess. A fixed October origin compares observed, neural and Daily trajectories without claiming a weather event.')

# Supplement S1 discrete Alice rank facets.
fig,axs=plt.subplots(1,3,figsize=(7.1,3.5),layout='constrained');rankdata=[]
for ax,site in zip(axs,['Sanyo','Hanwha','Qcells']):
 q=G[(G.site==site)&(G.analysis=='primary')&G.model.isin(['Recurrent','Inverted','Joint-patch','TCN'])].copy();q['rank']=q.groupby(['h','scope']).RMSE.rank();rankdata.append(q);pv=q.pivot(index=['h','scope'],columns='model',values='rank')[['Recurrent','Inverted','Joint-patch','TCN']];ax.pcolormesh(np.arange(5)-.5,np.arange(9)-.5,pv.values,vmin=.5,vmax=4.5,cmap=plt.get_cmap('Greys',4),shading='flat',rasterized=False);ax.set_ylim(7.5,-.5);ax.set_xticks(range(4),['Rec','Inv','Patch','TCN']);ax.set_yticks(range(8),[f'{h/12:g}h {"F" if scope=="full" else "A"}' for h,scope in pv.index]);ax.set_title(site)
 for (i,j),v in np.ndenumerate(pv.values):ax.text(j,i,f'{v:g}',ha='center',va='center',color='white' if v>=3 else 'black',fontsize=8)
save(fig,'figS1_alice_ranks',pd.concat(rankdata),'Alice neural ranks by cumulative window and target-power scope. F: full; A: power-active; four discrete rank levels.','Three facets retain co-located arrays separately and show discrete numerical ranks for four models.')

# S2 common-origin sensitivity with full baseline panel and neural-detail panel.
fig,axs=plt.subplots(5,2,figsize=(7.1,10),layout='constrained')
q=D[(D.analysis=='primary')&(D.scope=='full')]
for i,site in enumerate(['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):
 for j in [0,1]:
  ax=axs[i,j];models=['Inverted','Last-value'] if j==0 else ['Recurrent','Inverted','Joint-patch','TCN']
  for model in models:
   for support,ls in [('horizon_specific','-'),('common_H144','--')]:
    z=q[(q.site==site)&(q.model==model)&(q.support==support)].groupby('h').RMSE.agg(['mean','std']);ax.errorbar(z.index/12,z['mean'],yerr=z['std'].fillna(0),color=C[model],ls=ls,marker=MARK[model],markersize=3,lw=.8,label=model+' '+('HS' if support=='horizon_specific' else 'CO'))
  ax.set_title(site+(' / full reference range' if j==0 else ' / neural detail'));ax.set_ylabel('RMSE (kW)');ax.set_ylim(bottom=0);ax.set_xticks([1,4,8,12]);ax.set_xlabel('Cumulative window (h)')
from matplotlib.lines import Line2D
handles=[Line2D([],[],color=C[m],marker=MARK[m],label=m) for m in ['Recurrent','Inverted','Joint-patch','TCN','Last-value']]+[Line2D([],[],color='black',ls='-',label='HS'),Line2D([],[],color='black',ls='--',label='Common H144')]
fig.legend(handles=handles,loc='outside upper center',ncol=4,fontsize=8);save(fig,'figS2_common_origins',q,'Horizon-specific (HS, solid) versus common-H144-origin (CO, dashed) cumulative errors. Bars are seed SD. Left panels preserve reference error ranges; right panels show neural detail on separately labelled axes.','Five target rows compare original support with a fixed complete twelve-hour origin set; changes cannot be attributed solely to window length.')

# S3 preserved hardware-specific latency, log axis.
meta=pd.read_csv(HERE.parent/'scheme_A_submission_correction/corrected_metrics.csv');meta=meta[meta.analysis=='run_metadata'];mapping={'Discrete recurrent decoder':'Recurrent','Inverted-variate Transformer':'Inverted','Joint-patch Transformer':'Joint-patch','Depthwise convolutional TCN':'TCN'}
eff=meta[meta.metric.isin(['latency_mean_ms','parameter_count'])].pivot_table(index='model',columns='metric',values='value',aggfunc='mean').reset_index().rename(columns={'latency_mean_ms':'latency_ms','parameter_count':'parameters'});eff['model']=eff.model.map(mapping);eff['parameters']=eff.parameters.astype(int);eff=eff.set_index('model').loc[['Recurrent','Inverted','Joint-patch','TCN']].reset_index();fig,ax=plt.subplots(figsize=(7.1,2.4));ax.scatter(eff.latency_ms,range(4),c=[C[m] for m in eff.model],s=40);ax.set_xscale('log');ax.set_xlim(.3,60);ax.set_yticks(range(4),[f'{z.model} ({z.parameters:,} parameters)' for z in eff.itertuples()]);ax.set_xlabel('Batch-one latency (ms, log scale)');ax.grid(axis='x',alpha=.3)
for i,z in enumerate(eff.itertuples()):ax.annotate(f'{z.latency_ms:.3f}',(z.latency_ms,i),xytext=(5,5),textcoords='offset points',fontsize=8)
save(fig,'figS3_latency',eff,'Historical 17-channel inference timings on RTX 3060 Laptop GPU, excluding loading/preprocessing. Point summaries do not establish a stable difference between 0.535 and 0.558 ms; no new timing experiment.','A logarithmic latency axis displays all four models and their parameter counts without compressing sub-millisecond values into zero.')

# S4 fixed leads vs cumulative windows.
q=pd.read_csv(HERE/'results/lead_specific.csv');q=q[q.scope=='full'];fig,axs=plt.subplots(5,1,figsize=(7.1,9),layout='constrained')
for ax,site in zip(axs,['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):
 for model in ['Inverted','Daily','Last-value']:
  z=q[(q.site==site)&(q.model==model)].groupby('lead').RMSE.mean();ax.plot(z.index/12,z,color=C[model],label=model,lw=1)
 ax.set_title(site);ax.set_xlabel('Fixed forecast lead (h), common H144 origins');ax.set_ylabel('RMSE (kW)');ax.set_ylim(bottom=0)
axs[0].legend(ncol=3);save(fig,'figS4_fixed_leads',q,'Lead-specific error at each five-minute lead on common complete-H144 origins and Daily-valid targets. These are not cumulative-prefix metrics. Neural curves average per-seed RMSE.','Five system panels show fixed-lead curves on one origin set, distinguishing them from cumulative windows.')

# S5 block intervals.
q=pd.read_csv(HERE/'results/paired_block_intervals.csv');q=q[(q.analysis=='daily_matched')&q.site.isin(['YULARA_COMBINED','NIST_GROUND'])&(q.scope!='low-power')];fig,axs=plt.subplots(2,2,figsize=(7.1,4.5),layout='constrained')
for i,site in enumerate(['YULARA_COMBINED','NIST_GROUND']):
 for j,scope in enumerate(['full','power-active']):
  ax=axs[i,j]
  for bh,shift,col in [(24,-.17,'#999999'),(48,0,C['Inverted']),(72,.17,C['Recurrent'])]:
   z=q[(q.site==site)&(q.scope==scope)&(q.block_hours==bh)];ax.errorbar(z.h/12+shift,z.skill*100,yerr=np.vstack([(z.skill-z.ci_low)*100,(z.ci_high-z.skill)*100]),fmt='o',color=col,ms=3,capsize=2,label=f'{bh}h blocks')
  ax.axhline(0,color='black',lw=.7);ax.set_title(site+' / '+scope);ax.set_ylabel('Daily RMSE skill (%)');ax.set_xlabel('Cumulative window (h)');ax.set_xticks([1,4,8,12])
axs[0,0].legend(fontsize=8);save(fig,'figS5_block_intervals',q,'Paired temporal-block percentile intervals (2000 resamples; 24/48/72 h). SSE/count is aggregated before each seed RMSE, then seed skills are averaged. These intervals are conditional on observed dates and fitted models, not prediction intervals.','Temporal intervals expose uncertainty around small long-window Daily improvements. All three prechosen block sizes are shown, including intervals crossing zero.')

# S6 monthly summaries by origin, common H144.
q=pd.read_csv(HERE/'results/monthly.csv');q=q[q.site.isin(['YULARA_COMBINED','NIST_GROUND'])];fig,axs=plt.subplots(1,2,figsize=(7.1,3),layout='constrained')
for ax,site in zip(axs,['YULARA_COMBINED','NIST_GROUND']):
 for model in ['Inverted','Daily','Last-value']:
  z=q[(q.site==site)&(q.model==model)].groupby('origin_month').RMSE.agg(['mean','std']);ax.errorbar(z.index,z['mean'],yerr=z['std'].fillna(0),marker=MARK[model],color=C[model],label=model,capsize=3)
 ax.set_title(site);ax.set_ylabel('12 h cumulative RMSE (kW)');ax.set_xlabel('Forecast-origin month');ax.set_ylim(bottom=0)
axs[0].legend();save(fig,'figS6_monthly',q,'Monthly errors assign each full twelve-hour trajectory to its forecast-origin month on common H144 Daily-valid support. Bars show seed SD.','October, November and December are compared without assigning months by target time or treating them as independent climate replications.')

q=pd.read_csv(HERE/'results/trajectory_cases.csv');fig,axs=plt.subplots(5,2,figsize=(7.1,10),layout='constrained')
for j,site in enumerate(['YULARA_COMBINED','NIST_GROUND']):
 for i,label in enumerate(['fixed_month_10','fixed_month_11','fixed_month_12','posthoc_best','posthoc_worst']):
  ax=axs[i,j];z=q[(q.site==site)&(q.case==label)]
  for field,col in [('truth','black'),('Inverted',C['Inverted']),('Daily',C['Daily'])]:ax.plot(z.hours,z[field],label=field,color=col,lw=.8,ls='--' if field=='Daily' else '-')
  ax.set_title(site+' '+label+'\n'+str(z.origin.iloc[0])[:16],fontsize=8);ax.set_ylabel('AC power (kW)');ax.set_xlabel('Lead (h)')
axs[0,0].legend(fontsize=8);save(fig,'figS7_cases',q,'Fixed calendar and explicitly post hoc best/worst seed42 H144 Daily-improvement cases. Each curve is one forecast origin; extremes are diagnostic, not representative frequencies. No weather or outage labels are inferred.','Ten trajectory panels include predetermined dates plus both favorable and unfavorable error extremes for the same fixed seed.')

if (HERE/'results/ridge_metrics.csv').exists():
 q=pd.read_csv(HERE/'results/ridge_metrics.csv');q=q[(q.support=='horizon_specific')&(q.analysis=='daily_matched')&(q.scope=='full')];nn=D[(D.support=='horizon_specific')&(D.analysis=='daily_matched')&(D.scope=='full')&D.model.isin(['Inverted','Daily'])];q=pd.concat([q,nn],ignore_index=True);fig,axs=plt.subplots(5,1,figsize=(7.1,9),layout='constrained')
 for ax,site in zip(axs,['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):meanplot(ax,q[q.site==site],['Inverted','Daily','Ridge','Ridge+day']);ax.set_title(site);ax.set_ylabel('RMSE (kW)');ax.set_ylim(bottom=0);ax.set_xticks([1,4,8,12])
 axs[0].legend(ncol=4);save(fig,'figS8_ridge',q,'Post hoc Ridge with recent history and Ridge plus available previous-day trajectory; alphas selected on Validation only. Daily-matched full targets are identical to the neural comparison. Ridge is deterministic; neural bars are three-seed SD.','Five systems compare simple learned references against Inverted-variate and Daily; all unfavorable results are retained.')
q=pd.read_csv(HERE/'results/learning_histories.csv');q=q[(q.site=='NIST_GROUND')&(q.model=='DEPTHWISE_TCN_TRAJECTORY')];fig,axs=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained')
for ax,seed in zip(axs,[42,43,44]):
 z=q[q.seed==seed];ax.plot(z.epoch,z.train_MSE,label='Train',color=C['Inverted']);ax.plot(z.epoch,z.validation_MSE,label='Validation',color=C['Recurrent'],ls='--');ax.set_title(f'NIST TCN seed {seed}');ax.set_xlabel('Historical epoch');ax.set_ylabel('Scaled global MSE');ax.set_ylim(bottom=0)
axs[0].legend(fontsize=8);save(fig,'figS9_learning_budget',q,'Historical NIST TCN learning records. All three selected checkpoints occur at the fixed maximum budget of 25 epochs. These curves document the budget boundary, not convergence or a new fit.','Three historical training and validation curves expose continued improvement at the budget boundary; no extended training was performed.')
(OUT/'FIGURE_CAPTIONS_AND_ALT.json').write_text(json.dumps(records,indent=2),encoding='utf8')
(OUT/'FIGURE_QA.md').write_text('# Figure QA\n\nAll figures: 7.1-inch intended two-column width; PDF/SVG vector and PNG 320 dpi. DejaVu bundled fonts; source CSV, caption and alt text per figure. Generated from review CSVs; old frozen CSVs unchanged. Visual review must be recorded separately after rendering.\n\n'+'\n'.join('- '+x['name']+': '+x['caption'] for x in records),encoding='utf8')
print('Figures generated:',len(records))

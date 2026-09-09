"""Executed with the parent figure builder's plotting/data namespace."""
q=pd.read_csv(HERE/'results/ridge_paired_intervals.csv');q=q[(q.block_hours==48)&q.reference.isin(['Ridge A','Daily','Inverted mean'])&q.scope.isin(['full','power-active'])]
fig,axs=plt.subplots(5,2,figsize=(7.1,8.2),layout='constrained')
for i,site in enumerate(['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):
 for j,scope in enumerate(['full','power-active']):
  ax=axs[i,j];z=q[(q.site==site)&(q.scope==scope)].set_index('reference').loc[['Ridge A','Daily','Inverted mean']]
  for k,(ref,row) in enumerate(z.iterrows()):ax.errorbar(row.effect_kW,k,xerr=np.array([[row.effect_kW-row.ci_low_kW],[row.ci_high_kW-row.effect_kW]]),fmt=['v','x','o'][k],color=[C['Ridge'],C['Daily'],C['Inverted']][k],capsize=3)
  ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(3),['B − A','B − Daily','B − Inverted']);ax.set_ylim(2.6,-.6);ax.set_title({'YULARA_COMBINED':'Yulara','NIST_GROUND':'NIST'}.get(site,site)+' / '+scope);ax.set_xlabel('RMSE difference (kW)');ax.grid(axis='x',alpha=.2)
save(fig,'fig6_ridge_intervals',q,'Original-grid Ridge B minus each reference: negative RMSE differences favor B. Points and 95% paired 48-hour temporal-block intervals use identical complete-H144, Daily-finite targets within each system and scope. Inverted effects average three seed-specific contrasts under common draws. Panel scales differ; kW errors are not pooled.','Conditional temporal intervals distinguish observed error reductions from stability in unobserved periods. Ridge is deterministic; intervals do not include refitting.')

support=pd.read_csv(HERE/'results/qcells_support.csv');hours=pd.read_csv(HERE/'results/qcells_hours.csv');hists=pd.read_csv(HERE/'results/qcells_power_histogram.csv')
fig,axs=plt.subplots(1,2,figsize=(7.1,3.7),layout='constrained')
for label,col,ls in [('H12_support',C['Inverted'],'-'),('common_H144',C['Ridge+day'],'--'),('H12_removed_by_H144',C['Recurrent'],':')]:
 z=hours[(hours.split=='test')&(hours.support==label)];axs[0].plot(z.hour,z.origins/z.origins.sum()*100,label={'H12_support':'1 h support','common_H144':'Complete 12 h','H12_removed_by_H144':'Excluded'}[label],color=col,ls=ls,marker='o',ms=3)
axs[0].set_xlabel('Origin hour (frozen Alice coordinate)');axs[0].set_ylabel('Share of origins (%)');axs[0].set_xticks([0,6,12,18,23]);axs[0].legend()
z=support[(support.split=='test')&(support.h==12)&support.support.isin(['H12_support','common_H144','H12_removed_by_H144'])].set_index('support').loc[['H12_support','common_H144','H12_removed_by_H144']]
axs[1].bar(range(3),z.power_active_fraction*100,color=[C['Inverted'],C['Ridge+day'],C['Recurrent']],hatch=['','//','..']);axs[1].set_xticks(range(3),['1 h','Complete\n12 h','Excluded']);axs[1].set_ylabel('Active share of 1 h target pairs (%)');axs[1].set_ylim(0,105)
for i,row in enumerate(z.itertuples()):axs[1].text(i,row.power_active_fraction*100+2,f'{100*row.power_active_fraction:.3f}%',ha='center')
save(fig,'fig7_qcells_selection',pd.concat([hours.assign(panel='origin_hours'),support.assign(panel='power_support')],ignore_index=True),'Qcells Test: 6463 one-hour origins become 2996 complete-twelve-hour origins. Exclusive attribution assigns 3350 removals to future negative labels and 117 to the tail boundary. Active one-hour pairs decrease from 36504 to 42, involving 25 unique active timestamps and 25 origins.','The frozen negative-label rule and continuous-window requirement select a strongly different hourly and power distribution; complete-window support is not inherently representative.')

fig,axs=plt.subplots(3,2,figsize=(7.1,7.2),layout='constrained')
for i,split in enumerate(['train','validation','test']):
 for label,col,ls in [('H12_support',C['Inverted'],'-'),('common_H144',C['Ridge+day'],'--'),('H12_removed_by_H144',C['Recurrent'],':')]:
  z=hours[(hours.split==split)&(hours.support==label)];axs[i,0].plot(z.hour,z.origins/z.origins.sum()*100,color=col,ls=ls)
  z=hists[(hists.split==split)&(hists.h==12)&(hists.support==label)];finite=z[(z.bin_right>0)&(z.bin_left<6)].copy();axs[i,1].plot(range(len(finite)),finite.target_pairs/finite.target_pairs.sum()*100,color=col,ls=ls,marker='o',ms=3)
 axs[i,0].set_title(split+' origins');axs[i,0].set_xlabel('Origin hour');axs[i,0].set_ylabel('Origins (%)');axs[i,0].set_xticks([0,6,12,18,23]);axs[i,1].set_title(split+' / 1 h power pairs');axs[i,1].set_xlabel('Power bin upper edge (kW)');axs[i,1].set_ylabel('Target pairs (%)');axs[i,1].set_xticks(range(len(finite)),[f'{v:.2g}' for v in finite.bin_right],rotation=45)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([],[],color=c,ls=l,label=n) for n,c,l in [('1 h support',C['Inverted'],'-'),('Complete 12 h',C['Ridge+day'],'--'),('Excluded',C['Recurrent'],':')]],loc='outside upper center',ncol=3)
save(fig,'figS10_qcells_support',pd.concat([hours.assign(panel='hours'),hists.assign(panel='power_histogram')],ignore_index=True),'Qcells support composition in every split. Left: origin-hour proportions. Right: finite one-hour power-pair distributions in explicit bins. Complete-H144 conditioning selects different hours and power ranges in Train and Validation as well as Test.','All three split rows show the complete-window effect. Power-pair counts include overlapping forecasts and are not independent physical measurements.')

dist=pd.read_csv(HERE/'results/ridge_distributions.csv');features=pd.read_csv(HERE/'results/ridge_features.csv')
fig,axs=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
for k,variant in enumerate(['Ridge','Ridge+day']):
 z=dist[(dist.model==variant)&(dist.kind=='prediction')].set_index('split').loc[['train','validation','test']];x=np.arange(3)+(k-.5)*.16;axs[0].vlines(x,z['min'],z['max'],color=C[variant],lw=1);axs[0].vlines(x,z.p01,z.p99,color=C[variant],lw=5);axs[0].scatter(x,z['median'],color=C[variant],marker=MARK[variant],label=variant)
axs[0].axhline(109.903154,color='black',ls=':',lw=1);axs[0].set_xticks(range(3),['Train','Validation','Test']);axs[0].set_ylabel('Predicted power (kW)');axs[0].legend();axs[0].set_title('Range; thick line: 1–99%')
z=features[(features.model=='Ridge')&(features.feature=='lag0:temperature_missing')].set_index('split').loc[['train','validation','test']];axs[1].bar(range(3),z.raw_mean*100,color=[C['Inverted'],C['Recurrent'],C['Ridge']],hatch=['','//','..']);axs[1].set_xticks(range(3),['Train','Validation','Test']);axs[1].set_ylabel('Missing temperature at origin (%)');axs[1].set_title('Train mask scale: 0.007595')
save(fig,'figS11_yulara_diagnostic',pd.concat([dist.assign(panel='distribution'),z.reset_index().assign(panel='missing_frequency')],ignore_index=True),'Yulara original Ridge distributions and weather-missing exposure. Thin lines show extrema, thick lines 1st–99th percentiles, markers medians. The dotted line is the Train target maximum, not a clipping threshold.','Rare Train missing-mask exposure and frequent later missingness yield extrapolating linear contributions; real-matrix solvers agree numerically. Both adverse prediction ranges are retained.')

expanded=HERE/'results/expanded_ridge_metrics.csv'
if expanded.exists() and len(pd.read_csv(expanded))==30:
 q=pd.read_csv(expanded);q=q[q.scope=='full'];fig,axs=plt.subplots(1,5,figsize=(7.1,3.5),layout='constrained')
 for ax,site in zip(axs,['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']):
  z=q[q.site==site].set_index('model').loc[['Ridge','Ridge+day']]
  for i,r in enumerate(z.itertuples()):ax.plot([0,1],[r.old_RMSE,r.new_RMSE],marker=['v','P'][i],color=C[r.Index],label=r.Index)
  ax.set_xticks([0,1],['Original','Expanded'],rotation=35);ax.set_title({'YULARA_COMBINED':'Yulara','NIST_GROUND':'NIST'}.get(site,site));ax.set_ylim(bottom=0);ax.set_ylabel('12 h RMSE (kW)')
 axs[0].legend();save(fig,'figS12_alpha_sensitivity',q,'Uniform expanded alpha grid 10^-4 through 10^8, selected solely by full-H144 Validation MSE for all five systems and both information sets. Original and expanded Test errors use identical Daily-matched full targets; all ten comparisons are shown.','Separate scales retain the unfavorable original Yulara errors and every sensitivity result. The added grid is post hoc, not a silent replacement of the original fit.')
(OUT/'FIGURE_CAPTIONS_AND_ALT.json').write_text(json.dumps(records,indent=2),encoding='utf8')
print('Diagnostic figures generated:',len(records))

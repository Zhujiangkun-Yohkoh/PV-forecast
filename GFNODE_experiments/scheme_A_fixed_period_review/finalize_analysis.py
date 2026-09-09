from pathlib import Path
import json
import pandas as pd
from paired import intervals
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
rows=[]
for site in ['YULARA_COMBINED','NIST_GROUND']:
 b=pd.read_csv(OUT/(site+'_2018_Apr_Jun_origins_blocks.csv'))
 for (scope,h,hours),g in b.groupby(['scope','horizon','block_hours']):
  for r in intervals(g,[('Inverted mean','Daily'),('Original B','Daily'),('Expanded B','Daily'),('Expanded B','Inverted mean')]):rows.append(dict(site=site,period='2018_Apr_Jun_origins',support='fixed_period',scope=scope,horizon=h,block_hours=hours,**r))
pd.DataFrame(rows).to_csv(OUT/'fixed_2018_primary_intervals.csv',index=False)
for site in ['YULARA_COMBINED','NIST_GROUND']:
 q=pd.DataFrame(rows);print(q[(q.site==site)&(q.horizon==144)&(q.block_hours==48)&(q.scope=='full')][['method','reference','rmse','reference_rmse','effect_kW','ci_low_kW','ci_high_kW']].to_string(index=False))
replay=[]
for site in ['Sanyo','Hanwha','Qcells']:
 for r in json.loads((OUT/(site+'_REPLAY.json')).read_text(encoding='utf8')):replay.append(dict(site=site,**r))
pd.DataFrame(replay).to_csv(OUT/'Alice_replay_summary.csv',index=False)
print('Alice replay:',pd.DataFrame(replay).groupby('passed').size().to_dict())
summ=[];dec=[]
for site in ['YULARA_COMBINED','NIST_GROUND']:
 d=pd.read_csv(OUT/(site+'_2018_Apr_Jun_origins_metrics.csv'))
 for (h,scope),g in d.groupby(['horizon','scope']):
  for name,keys in [('Inverted mean',['Inverted42','Inverted43','Inverted44'])]+[(k,[k]) for k in ['Original A','Original B','Expanded A','Expanded B','Daily','Last-value']]:
   q=g[g.method.isin(keys)];summ.append(dict(site=site,horizon=h,scope=scope,method=name,RMSE=q.RMSE.mean(),MAE=q.MAE.mean(),bias=q.bias.mean(),points=int(q.points.iloc[0]),origins=int(q.origins.iloc[0])))
 for h,g in d.groupby('horizon'):
  q=g.set_index(['method','scope']);n=q.loc[('Daily','full'),'points']
  for name in ['Inverted42','Inverted43','Inverted44','Original B','Expanded B']:
   parts=[]
   for scope in ['power-active','low-power']:
    v=(q.loc[(name,scope),'SSE']-q.loc[('Daily',scope),'SSE'])/n;parts.append(v);dec.append(dict(site=site,horizon=h,method=name,scope=scope,delta_MSE=v,points=int(q.loc[(name,scope),'points']),full_points=int(n)))
   assert abs(sum(parts)-(q.loc[(name,'full'),'SSE']-q.loc[('Daily','full'),'SSE'])/n)<1e-8
pd.DataFrame(summ).to_csv(OUT/'period_table_summary.csv',index=False);pd.DataFrame(dec).to_csv(OUT/'period_table_decomposition.csv',index=False)

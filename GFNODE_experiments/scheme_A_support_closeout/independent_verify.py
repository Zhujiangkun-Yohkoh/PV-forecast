"""Independent percentile bootstrap from block CSVs; never imports production intervals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent/'results'

def verify():
 total=0;groups=0
 for bf in sorted(R.glob('*_blocks.csv')):
  ef=bf.with_name(bf.name.replace('_blocks.csv','_intervals.csv'));b=pd.read_csv(bf);e=pd.read_csv(ef);keys=['site','period','support','horizon','scope','block_hours']
  for key,g in b.groupby(keys,sort=False):
   expected=e
   for k,v in zip(keys,key):expected=expected[expected[k]==v]
   counts=g['count'].to_numpy(float);B=len(counts);rng=np.random.default_rng(20260908);sample=rng.integers(B,size=(2000,B));multiplicity=np.zeros((2000,B),dtype=np.int32)
   # Count selected blocks, then use matrix multiplication rather than production indexed sums.
   np.add.at(multiplicity,(np.repeat(np.arange(2000),B),sample.ravel()),1);N=multiplicity@counts;valid=N>0
   sse={k[:-4]:g[k].to_numpy(float) for k in g if k.endswith('_SSE')};rr={k:np.sqrt((multiplicity@v)[valid]/N[valid]) for k,v in sse.items()};point={k:np.sqrt(v.sum()/counts.sum()) if counts.sum() else np.nan for k,v in sse.items()}
   for row in expected.itertuples():
    methods=['Inverted42','Inverted43','Inverted44'] if row.method=='Inverted mean' else [row.method];refs=['Inverted42','Inverted43','Inverted44'] if row.reference=='Inverted mean' else [row.reference]
    ds=np.stack([rr[a]-rr[z] for a in methods for z in refs]).mean(0);d0=np.mean([point[a]-point[z] for a in methods for z in refs]);ci=np.percentile(ds,[2.5,97.5]) if len(ds) else [np.nan,np.nan]
    assert np.allclose([d0,*ci],[row.effect_kW,row.ci_low_kW,row.ci_high_kW],rtol=1e-8,atol=1e-10,equal_nan=True),(key,row.method,row.reference)
    assert row.valid_replicates==valid.sum() and row.nonempty_blocks==(counts>0).sum() and row.points==counts.sum();total+=1
   groups+=1
 states=pd.read_csv(R/'Alice_method_status.csv');assert len(states)==36;status=states.set_index(['site','method']).status.to_dict()
 for site in states.site.unique():
  d=pd.read_csv(R/(site+'_metrics.csv'));e=pd.read_csv(R/(site+'_intervals.csv'));accepted=set(d[d.status=='accepted'].method)
  for row in e.itertuples():
   for name in [row.method,row.reference]:
    if name=='Inverted mean':assert all('Inverted'+str(s) in accepted for s in [42,43,44])
    else:assert name in accepted
  for row in d.itertuples():
   if (site,row.method) in status:assert row.status==status[site,row.method]
 impact=pd.read_csv(R/'Alice_metric_impact.csv');assert (impact.delta_RMSE_kW.abs()<=impact.difference_RMS_kW+1e-12).all();assert (impact.delta_MAE_kW.abs()<=impact.difference_MAE_kW+1e-12).all()
 for mf in R.glob('*_metrics.csv'):
  d=pd.read_csv(mf)
  if 'support' not in d:continue
  for key,g in d.groupby(['site','period','support','horizon','method']):
   q=g.set_index('scope');assert q.loc['full','points']==q.loc['power-active','points']+q.loc['low-power','points'];assert np.isclose(q.loc['full','SSE'],q.loc['power-active','SSE']+q.loc['low-power','SSE'],rtol=1e-12,atol=1e-9)
 out=dict(independent_interval_rows=total,block_groups=groups,Alice_original_tolerance_pass=int(states.passed_original_tolerance.sum()),Alice_original_tolerance_unresolved=int((~states.passed_original_tolerance).sum()),failed=0,skipped=0,production_interval_import=False,neural_forward_in_this_check=False,meaning='independent block-weight arithmetic and acceptance/seed isolation; not independent experiments')
 print(json.dumps(out,indent=2));return out
if __name__=='__main__':verify()

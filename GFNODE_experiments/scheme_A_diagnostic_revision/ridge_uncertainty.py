"""Paired temporal uncertainty for frozen linear and neural predictions."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def read(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def effects(blocks,seed=20260908):
 n=blocks['n'];nb=len(n);rng=np.random.default_rng(seed);ix=rng.integers(0,nb,size=(2000,nb));nn=n[ix].sum(1);ok=nn>0
 rm={k:np.sqrt(v[ix][ok].sum(1)/nn[ok]) for k,v in blocks.items() if k!='n'};point={k:np.sqrt(v.sum()/n.sum()) for k,v in blocks.items() if k!='n'};rows=[]
 for ref in ['Ridge A','Daily','Inverted42','Inverted43','Inverted44','Inverted mean']:
  r=np.mean([rm['Inverted'+str(s)] for s in [42,43,44]],axis=0) if ref.endswith('mean') else rm[ref];rp=np.mean([point['Inverted'+str(s)] for s in [42,43,44]]) if ref.endswith('mean') else point[ref]
  delta=rm['Ridge B']-r;skill=np.mean([1-rm['Ridge B']/rm['Inverted'+str(s)] for s in [42,43,44]],axis=0) if ref.endswith('mean') else 1-rm['Ridge B']/r;lo,hi=np.quantile(delta,[.025,.975]);sl,sh=np.quantile(skill,[.025,.975]);rows.append(dict(reference=ref,rmse_B=point['Ridge B'],rmse_reference=rp,effect_kW=point['Ridge B']-rp,ci_low_kW=lo,ci_high_kW=hi,skill=float(np.mean([1-point['Ridge B']/point['Inverted'+str(s)] for s in [42,43,44]])) if ref.endswith('mean') else 1-point['Ridge B']/rp,skill_ci_low=sl,skill_ci_high=sh,replicates=2000,valid_replicates=int(ok.sum())))
 return rows
def run(paths):
 c=json.loads(Path(paths).read_text());out=HERE/'results';out.mkdir(exist_ok=True);rows=[];blockrows=[];support=[]
 for site in ['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']:
  f=Path(c['new_results'])/site;a=read(f/'Ridge_predictions.npz');b=read(f/'Ridge_day_predictions.npz');y=a['labels'];v=a['target_valid'];o=pd.DatetimeIndex(pd.to_datetime(a['forecast_origin'])).as_unit('ns');preds={'Ridge A':a['predictions'],'Ridge B':b['predictions'],'Daily':a['daily']}
  for name in ['forecast_origin','target_start','target_valid']:assert np.array_equal(a[name],b[name])
  assert np.allclose(a['labels'],b['labels'],equal_nan=True);assert np.allclose(a['daily'],b['daily'],equal_nan=True)
  external=site in ['YULARA_COMBINED','NIST_GROUND'];root=Path(c['external_results' if external else 'alice_results']);loaded=[]
  for p in root.glob('*/completed.json'):
   info=json.loads(p.read_text());ss=info.get('site',info.get('dataset'));model=info['model']
   if ss!=site or 'inverted' not in model.lower():continue
   z=read(p.parent/('test_predictions.npz' if external else 'test_H144.npz'));assert np.array_equal(z['forecast_origin'],a['forecast_origin']);assert np.array_equal(z['target_start'],a['target_start']);assert np.array_equal(z['target_valid'],v);assert np.allclose(z['labels'],y,equal_nan=True);preds['Inverted'+str(info['seed'])]=z['predictions'];loaded.append(info['seed'])
  assert sorted(loaded)==[42,43,44]
  common=v.all(1)&np.isfinite(a['last_power']);base=common[:,None]&v&np.isfinite(a['daily']);assert all(np.isfinite(p[base]).all() for p in preds.values())
  anchor=pd.Timestamp('2017-10-01',tz=o.tz) if external else pd.Timestamp('2018-08-08');delta=(o-anchor).total_seconds()/3600
  targets=o.asi8[:,None]+np.arange(1,145)*300_000_000_000
  for scope,mask in [('full',base),('power-active',base&(y>a['daylight_threshold'])),('low-power',base&(y<=a['daylight_threshold']))]:
   support.append(dict(site=site,scope=scope,origins=int(mask.any(1).sum()),points=int(mask.sum()),unique_target_timestamps=len(np.unique(targets[mask])),support='common_H144_and_Daily_finite_all_methods',anchor=str(anchor),coordinate='FIXED_EST_LST' if site=='NIST_GROUND' else ('provider_local' if external else 'frozen_Alice'),target_index_sha256=hashlib.sha256(targets[mask].tobytes()).hexdigest()))
   for hours in [24,48,72]:
    ids=np.asarray(delta//hours,int);nb=ids.max()+1;n=np.bincount(ids,weights=mask.sum(1),minlength=nb);blocks={'n':n}
    for model,p in preds.items():blocks[model]=np.bincount(ids,weights=np.where(mask,(p.astype(float)-y.astype(float))**2,0).sum(1),minlength=nb)
    for j in range(nb):blockrows.append(dict(site=site,scope=scope,block_hours=hours,block=j,start=str(anchor+pd.Timedelta(hours=j*hours)),count=n[j],**{k+'_SSE':z[j] for k,z in blocks.items() if k!='n'}))
    for r in effects(blocks):rows.append(dict(site=site,scope=scope,block_hours=hours,blocks=nb,nonempty_blocks=int((n>0).sum()),points=int(n.sum()),**r))
  print(site,'paired target indices and three neural seeds aligned',flush=True)
 pd.DataFrame(rows).to_csv(out/'ridge_paired_intervals.csv',index=False);pd.DataFrame(blockrows).to_csv(out/'ridge_block_sse.csv',index=False);pd.DataFrame(support).to_csv(out/'ridge_paired_support.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);a=p.parse_args();run(a.paths)

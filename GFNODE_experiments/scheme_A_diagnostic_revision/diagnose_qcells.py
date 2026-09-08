"""Raw-timestamp support attribution; no neural loader or fit."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def run(paths):
 c=json.loads(Path(paths).read_text());cfg=json.loads((HERE.parent/'scheme_A_submission_correction/config.json').read_text());raw=pd.read_csv(c['alice_raw']['Qcells']);raw.timestamp=pd.to_datetime(raw.timestamp);raw=raw.sort_values('timestamp').drop_duplicates('timestamp',keep='last').set_index('timestamp');grid=pd.date_range(raw.index.min(),raw.index.max(),freq='5min');power=pd.to_numeric(raw.Active_Power,errors='coerce').reindex(grid);threshold=.01*power.loc[slice(*cfg['splits']['train'])].where(lambda v:v>=0).max();tables=[];hours=[];months=[];reasons=[];flags=[];hist=[];checks=[]
 for split,bounds in cfg['splits'].items():
  p=power.loc[slice(*bounds)];idx=p.index;y=p.to_numpy(float);n=len(y);pos=np.arange(n);valid=np.isfinite(y)&(y>=0);t=pos[:,None]+np.arange(1,145);inside=t<n;safe=np.minimum(t,n-1);future=np.where(inside,y[safe],np.nan);missing=inside&~np.isfinite(future);negative=inside&np.isfinite(future)&(future<0);states={}
  for h in [12,144]:
   boundary=(pos<71)|(pos+h>=n);nonfinite=missing[:,:h].any(1);neg=negative[:,:h].any(1);origin_invalid=~valid;keep=~(boundary|nonfinite|neg|origin_invalid);states[h]=keep
   lag=idx.to_numpy()[:,None]+np.arange(1,h+1)*np.timedelta64(5,'m')-np.timedelta64(24,'h');day=power.reindex(pd.DatetimeIndex(lag.ravel())).to_numpy().reshape(n,h);dailybad=~np.isfinite(day)|(day<0)
   cause=np.select([boundary,nonfinite,neg,origin_invalid],['boundary','future_missing_nonfinite','future_negative','origin_invalid'],default='retained')
   f=pd.DataFrame(dict(split=split,h=h,origin=idx.astype(str),boundary=boundary,future_missing_nonfinite=nonfinite,future_negative=neg,origin_invalid=origin_invalid,daily_any_unavailable=dailybad.any(1),primary_keep=keep,exclusive_cause=cause));flags.append(f)
   for name,z in [('boundary',boundary),('future_missing_nonfinite',nonfinite),('future_negative',neg),('origin_invalid',origin_invalid),('daily_any_unavailable',dailybad.any(1))]:reasons.append(dict(split=split,h=h,population='all_grid_origins',mode='overlap',cause=name,count=int(z.sum())))
   for name in np.unique(cause):reasons.append(dict(split=split,h=h,population='all_grid_origins',mode='exclusive',cause=name,count=int((cause==name).sum())))
   assert len(cause)==sum(int((cause==v).sum()) for v in np.unique(cause));checks.append(dict(split=split,h=h,candidates=n,retained=int(keep.sum()),exclusive_reconciled=True))
  subsets={'all_grid_origins':np.ones(n,bool),'H12_support':states[12],'common_H144':states[144],'H12_removed_by_H144':states[12]&~states[144],'historical_training_fullH144':(pos>=71)&(pos+144<n)&~missing.any(1)&~negative.any(1)}
  f144=flags[-1];sel=states[12]
  for name in np.unique(f144.exclusive_cause):reasons.append(dict(split=split,h=144,population='H12_support',mode='exclusive',cause=name,count=int(((f144.exclusive_cause==name)&sel).sum())))
  for name in ['boundary','future_missing_nonfinite','future_negative','origin_invalid','daily_any_unavailable']:reasons.append(dict(split=split,h=144,population='H12_support',mode='overlap',cause=name,count=int((f144[name]&sel).sum())))
  for label,keep in subsets.items():
   for hour in range(24):hours.append(dict(split=split,support=label,hour=hour,origins=int((keep&(idx.hour==hour)).sum())))
   for mo in sorted(set(idx.strftime('%Y-%m'))):months.append(dict(split=split,support=label,month=mo,origins=int((keep&(idx.strftime('%Y-%m')==mo)).sum())))
   for h in [12,144]:
    mask=keep[:,None]&inside[:,:h];finite=mask&np.isfinite(future[:,:h]);v=finite&(future[:,:h]>=0);active=v&(future[:,:h]>threshold);vals=future[:,:h][finite];targetid=t[:,:h]
    tables.append(dict(split=split,support=label,h=h,origins=int(keep.sum()),in_split_target_pairs=int(mask.sum()),finite_target_pairs=int(finite.sum()),valid_target_pairs=int(v.sum()),power_active_pairs=int(active.sum()),power_active_fraction=float(active.sum()/v.sum()) if v.sum() else np.nan,unique_physical_targets=len(np.unique(targetid[mask])),unique_valid_physical_targets=len(np.unique(targetid[v])),unique_active_physical_targets=len(np.unique(targetid[active])),active_origins=int(active.any(1).sum()),negative_target_pairs=int((finite&(future[:,:h]<0)).sum()),missing_target_pairs=int((mask&~np.isfinite(future[:,:h])).sum()),power_min=float(np.min(vals)) if len(vals) else np.nan,power_median=float(np.median(vals)) if len(vals) else np.nan,power_p95=float(np.quantile(vals,.95)) if len(vals) else np.nan,power_max=float(np.max(vals)) if len(vals) else np.nan))
    counts,edges=np.histogram(vals,bins=np.r_[-np.inf,-.01,0,threshold,.1,.5,1,2,3,4,5,6,np.inf])
    for j,count in enumerate(counts):hist.append(dict(split=split,support=label,h=h,bin_left=edges[j],bin_right=edges[j+1],target_pairs=count))
  # Verify exact frozen Test origins/labels/point masks, not count alone.
  if split=='test':
   pth=next(Path(c['alice_results']).glob('*_Qcells_42/test_H144.npz'))
   with np.load(pth) as z:
    o=pd.DatetimeIndex(pd.to_datetime(z['forecast_origin']));assert o.equals(idx[states[12]]);rows=np.flatnonzero(states[12]);expected=future[rows];ev=np.isfinite(expected)&(expected>=0);assert np.array_equal(ev,z['target_valid']);assert np.allclose(expected,z['labels'],equal_nan=True,atol=1e-6);assert pd.DatetimeIndex(pd.to_datetime(z['target_start'])).equals(o+pd.Timedelta(minutes=5))
    assert np.array_equal(z['target_valid'].all(1)&np.isfinite(z['last_power']),states[144][rows]);checks.append(dict(split=split,h=144,exact_frozen_timestamp_labels_mask=True))
 out=HERE/'results';out.mkdir(exist_ok=True)
 negative_records=[]
 for split,bounds in cfg['splits'].items():
  for timestamp,value in power.loc[slice(*bounds)].items():
   if np.isfinite(value) and value<0:negative_records.append(dict(split=split,timestamp=str(timestamp),hour=timestamp.hour,power=value))
 pd.DataFrame(negative_records).to_csv(out/'qcells_negative_records.csv',index=False)
 for name,rows in [('qcells_support',tables),('qcells_hours',hours),('qcells_months',months),('qcells_reasons',reasons),('qcells_power_histogram',hist)]:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
 pd.concat(flags).to_csv(out/'qcells_origin_flags.csv',index=False);(out/'QCELLS_ALIGNMENT.json').write_text(json.dumps(dict(threshold=threshold,checks=checks),indent=2));print(pd.DataFrame(tables).query("split=='test' and h==12").to_string(index=False));print(pd.DataFrame(reasons).query("split=='test' and population=='H12_support'").to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);a=p.parse_args();run(a.paths)

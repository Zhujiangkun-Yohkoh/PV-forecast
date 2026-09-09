"""Four-cell reductions only; never reruns a forward or fits a processor."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from analysis_saved import read,score,metrics,R

def reduce_site(c,site):
 old=read(Path(c['new_results'])/site/'Ridge_predictions.npz');oo=pd.DatetimeIndex(pd.to_datetime(old['forecast_origin'])).as_unit('ns');bound=[];archive_metrics=[]
 for period,path in [('2017',Path(c['closeout_destination'])/site/'unified_2017_predictions.npz'),('2018',Path(c['destination'])/site/'fixed_2018_predictions.npz')]:
  z=read(path);y=z['labels'];o=pd.DatetimeIndex(pd.to_datetime(z['forecast_origin'])).as_unit('ns');pp={k.replace('_',' '):z[k] for k in ['Original_A','Original_B','Expanded_A','Expanded_B','Inverted42','Inverted43','Inverted44','Daily','Last-value']};last=pp['Last-value'][:,0];daily=pp['Daily'];valid=np.isfinite(y);complete=valid.all(1)
  definitions=lambda h:{'complete_H144':valid[:,:h]&complete[:,None],'pointwise_finite':valid[:,:h]}
  score(site,period,o,y,pp,last,daily,float(z['threshold']),z['target_timestamp_ns'],definitions,{k:'accepted' for k in pp},site+'_'+period,'2017-10-01' if period=='2017' else '2018-04-01',horizons=(144,))
  assert np.array_equal(z['target_timestamp_ns'],o.asi8[:,None]+np.arange(1,145)*300_000_000_000)
  if period=='2017':
   oi=o.get_indexer(oo);assert (oi>=0).all();archive=np.zeros(len(o),bool);archive[oi]=True;ac=np.zeros(len(o),bool);ac[oi]=old['target_valid'].all(1)
   for label,om in [('archive_candidates',archive),('archive_original_H144',ac),('unified_candidates',np.ones(len(o),bool)),('unified_complete_H144',complete),('new_before_archive',o<oo[0]),('new_after_archive',o>oo[-1]),('other_nonarchive',~archive&(o>=oo[0])&(o<=oo[-1]))]:
    mask=valid&om[:,None]&np.isfinite(last)[:,None]&np.isfinite(daily);bound.append(dict(site=site,period=period,category=label,candidate_origins=int(om.sum()),scored_origins=int(mask.any(1).sum()),points=int(mask.sum()),first=str(o[om][0]) if om.any() else '',last=str(o[om][-1]) if om.any() else ''))
   mask=valid&ac[:,None]&np.isfinite(last)[:,None]&np.isfinite(daily)
   for k,v in pp.items():archive_metrics.append(dict(site=site,method=k,**metrics(v,y,mask)))
  else:
   for label,om in [('unified_candidates',np.ones(len(o),bool)),('unified_complete_H144',complete)]:
    mask=valid&om[:,None]&np.isfinite(last)[:,None]&np.isfinite(daily);bound.append(dict(site=site,period=period,category=label,candidate_origins=int(om.sum()),scored_origins=int(mask.any(1).sum()),points=int(mask.sum()),first=str(o[om][0]),last=str(o[om][-1])))
 pd.DataFrame(bound).to_csv(R/(site+'_boundary_audit.csv'),index=False);pd.DataFrame(archive_metrics).to_csv(R/(site+'_archive_metric_replay.csv'),index=False);print(site,'four cells completed',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--sites',nargs='+',default=['YULARA_COMBINED','NIST_GROUND']);a=p.parse_args();c=json.loads(Path(a.paths).read_text(encoding='utf8'))
 for site in a.sites:reduce_site(c,site)

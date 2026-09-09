"""Saved-array replay impact and method-specific support evaluation; no fits."""
from pathlib import Path
import json,argparse,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];R=HERE/'results';R.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/'scheme_A_fixed_period_review'))
from paired import reduce_blocks,intervals

def read(p):
 with np.load(p,allow_pickle=True) as z:return {k:z[k] for k in z.files} # trusted project objects only

def metrics(p,y,mask):
 e=p[mask].astype(float)-y[mask].astype(float);n=len(e)
 return dict(RMSE=np.sqrt(e@e/n) if n else np.nan,MAE=np.mean(abs(e)) if n else np.nan,bias=e.mean() if n else np.nan,SSE=e@e,points=n)

def model_key(info):return ('Inverted' if 'inverted' in info['model'].lower() else info['model'])+str(info['seed'])

def score(site,period,o,y,preds,last,daily,threshold,targets,definitions,status,stem,anchor,horizons=(12,144)):
 mr=[];sr=[];bf=[];ir=[]
 accepted={k for k,v in status.items() if v=='accepted'}
 for h in horizons:
  yy=y[:,:h]
  for support,base in definitions(h).items():
   for scope,sc in [('full',np.ones_like(base)),('power-active',yy>threshold),('low-power',yy<=threshold)]:
    mask=base&sc&np.isfinite(last)[:,None]&np.isfinite(daily[:,:h]);meta=dict(site=site,period=period,support=support,horizon=h,scope=scope)
    sr.append(dict(**meta,candidate_origins=len(o),origins=int(mask.any(1).sum()),points=int(mask.sum()),unique_targets=len(np.unique(targets[:,:h][mask])),active_points=int((mask&(yy>threshold)).sum()),active_fraction=float((mask&(yy>threshold)).sum()/mask.sum()) if mask.sum() else np.nan))
    pp={k:v[:,:h] for k,v in preds.items()}
    for k,v in pp.items():mr.append(dict(**meta,method=k,status=status[k],**metrics(v,yy,mask)))
    comps=[]
    for k in ['Original A','Expanded A','Original B','Expanded B','Inverted42','Inverted43','Inverted44']:
     for ref in ['Daily','Last-value']:
      if k in accepted:comps.append((k,ref))
    if all('Inverted'+str(s) in accepted for s in [42,43,44]):
     comps += [('Inverted mean',ref) for ref in ['Daily','Last-value','Original B','Expanded B']]
    if 'Expanded B' in accepted:comps += [('Expanded B',r) for r in ['Original B','Expanded A'] if r in accepted]
    for hours in [24,48,72]:
     block=reduce_blocks(yy,{k:v for k,v in pp.items() if k in accepted},mask,o,anchor,hours)
     bf.append(block.assign(**meta,block_hours=hours));ir += [dict(**meta,block_hours=hours,**r) for r in intervals(block,comps)]
 for suffix,d in [('metrics',pd.DataFrame(mr)),('support',pd.DataFrame(sr)),('blocks',pd.concat(bf)),('intervals',pd.DataFrame(ir))]:d.to_csv(R/(stem+'_'+suffix+'.csv'),index=False)
 return pd.DataFrame(mr)

def alice(c):
 diff=[];impact=[];states=[];ranks=[]
 previous=pd.read_csv(HERE.parent/'scheme_A_fixed_period_review/results/Alice_replay_summary.csv')
 for site in ['Sanyo','Hanwha','Qcells']:
  a=read(Path(c['destination'])/site/'support_and_ridge.npz');old=read(Path(c['new_results'])/site/'Ridge_predictions.npz');o=pd.DatetimeIndex(pd.to_datetime(a['forecast_origin'])).as_unit('ns');oo=pd.DatetimeIndex(pd.to_datetime(old['forecast_origin'])).as_unit('ns');ix=o.get_indexer(oo);assert (ix>=0).all()
  assert np.array_equal(a['target_valid'][ix],old['target_valid']);assert np.allclose(a['labels'][ix],old['labels'],equal_nan=True);assert np.array_equal(a['target_timestamp_ns'][ix],oo.asi8[:,None]+np.arange(1,145)*300_000_000_000)
  y=a['labels'];threshold=float(old['daylight_threshold']);preds={k.replace('_',' '):a[k] for k in ['Original_A','Original_B','Expanded_A','Expanded_B']};preds['Daily']=a['daily'];preds['Last-value']=np.broadcast_to(a['last_power'][:,None],y.shape);status={k:'accepted' for k in preds}
  for grid,root in [('Original',Path(c['new_results'])/site),('Expanded',Path(c['expanded_results'][site]))]:
   for label,stem in [('A','Ridge'),('B','Ridge_day')]:
    ref=read(root/(stem+'_predictions.npz'));assert np.array_equal(ref['forecast_origin'],old['forecast_origin']);assert np.allclose(preds[grid+' '+label][ix],ref['predictions'],rtol=2e-5,atol=2e-5)
  for f in Path(c['alice_results']).glob('*/completed.json'):
   info=json.loads(f.read_text(encoding='utf8'))
   if info['dataset']!=site:continue
   key=model_key(info);z=read(f.parent/'test_H144.npz');new=read(Path(c['destination'])/site/(key.replace(' ','_')+'.npz'));assert np.array_equal(new['forecast_origin'],a['forecast_origin']);assert np.array_equal(z['forecast_origin'],old['forecast_origin']);p0=z['predictions'].astype(float);p1=new['predictions'][ix].astype(float);delta=p1-p0;bad=~np.isclose(p1,p0,rtol=2e-5,atol=2e-5);passed=not bad.any();status[key]='accepted' if passed else 'diagnostic';preds[key]=new['predictions'];prev=previous[(previous.site==site)&(previous['check']==key)].iloc[0];assert bool(prev.passed)==passed
   states.append(dict(site=site,method=key,model=info['model'],seed=info['seed'],passed_original_tolerance=passed,status=status[key],checkpoint=f.parent.name,rtol=2e-5,atol=2e-5))
   for scope,mask in [('all_outputs',np.ones_like(delta,dtype=bool)),('full',np.isfinite(z['labels'])&(z['labels']>=0)),('power-active',np.isfinite(z['labels'])&(z['labels']>threshold)),('low-power',np.isfinite(z['labels'])&(z['labels']>=0)&(z['labels']<=threshold))]:
    v=abs(delta[mask]);n=len(v);masked=np.where(mask,abs(delta),-1);i,j=np.unravel_index(np.argmax(masked),masked.shape)
    row=dict(site=site,method=key,scope=scope,compared_pairs=n,identical_points=int((delta[mask]==0).sum()),identical_fraction=float((delta[mask]==0).mean()) if n else np.nan,over_tolerance_points=int(bad[mask].sum()),over_tolerance_fraction=float(bad[mask].mean()) if n else np.nan,affected_origins=int((bad&mask).any(1).sum()),max_origin=str(oo[i]),max_lead_min=(j+1)*5,old_kW=p0[i,j],new_kW=p1[i,j],original_tolerance_passed=passed)
    for label,value in [('median',np.median(v)),('p95',np.quantile(v,.95)),('p99',np.quantile(v,.99)),('max',v.max()),('difference_RMS',np.sqrt(np.mean(v*v))),('difference_MAE',v.mean())]:row[label+'_kW']=value;row[label+'_W']=1000*value
    row['old_W']=row['old_kW']*1000;row['new_W']=row['new_kW']*1000;diff.append(row)
   for h in [12,144]:
    yy=z['labels'][:,:h];valid=z['target_valid'][:,:h];base=valid&valid.all(1)[:,None]&np.isfinite(a['daily'][ix,:h])&np.isfinite(a['last_power'][ix])[:,None]
    for scope,sc in [('full',np.ones_like(base)),('power-active',yy>threshold),('low-power',yy<=threshold)]:
     mask=base&sc;m0=metrics(p0[:,:h],yy,mask);m1=metrics(p1[:,:h],yy,mask);daily=metrics(a['daily'][ix,:h],yy,mask);v=delta[:,:h][mask];rms=np.sqrt(np.mean(v*v));ma=np.mean(abs(v));assert abs(m1['RMSE']-m0['RMSE'])<=rms+1e-12;assert abs(m1['MAE']-m0['MAE'])<=ma+1e-12
     row=dict(site=site,method=key,model=info['model'],seed=info['seed'],horizon=h,scope=scope,status=status[key],points=m0['points'],Daily_RMSE=daily['RMSE'],difference_RMS_kW=rms,difference_MAE_kW=ma,RMSE_bound_passed=True,MAE_bound_passed=True)
     for k in ['RMSE','MAE','bias']:
      row['old_'+k]=m0[k];row['new_'+k]=m1[k];row['delta_'+k+'_kW']=m1[k]-m0[k];row['delta_'+k+'_W']=1000*(m1[k]-m0[k])
     row['old_effect_vs_Daily']=m0['RMSE']-daily['RMSE'];row['new_effect_vs_Daily']=m1['RMSE']-daily['RMSE'];row['old_skill']=1-m0['RMSE']/daily['RMSE'] if daily['RMSE']>0 else np.nan;row['new_skill']=1-m1['RMSE']/daily['RMSE'] if daily['RMSE']>0 else np.nan;impact.append(row)
  def definitions(h):
   v=a['target_valid'][:,:h];oldmask=v.all(1);finite=np.isfinite(y[:,:h]);return {'Original':v&oldmask[:,None],'S1':v,'S2':finite,'S1_new_origins':v&~oldmask[:,None],'S2_new_origins':finite&~oldmask[:,None]}
  score(site,'Alice2018Test',o,y,preds,a['last_power'],a['daily'],threshold,a['target_timestamp_ns'],definitions,status,site,'2018-08-08')
  print(site,'impact quantified; accepted methods',sorted(k for k,v in status.items() if v=='accepted'),flush=True)
 pd.DataFrame(diff).to_csv(R/'Alice_prediction_differences.csv',index=False);im=pd.DataFrame(impact);im.to_csv(R/'Alice_metric_impact.csv',index=False);pd.DataFrame(states).to_csv(R/'Alice_method_status.csv',index=False)
 for key,g in im.groupby(['site','horizon','scope']):
  q=g.groupby('model')[['old_RMSE','new_RMSE']].mean();q['old_rank']=q.old_RMSE.rank(method='min');q['new_rank']=q.new_RMSE.rank(method='min');ranks.append(q.reset_index().assign(site=key[0],horizon=key[1],scope=key[2],meaning='historical-support numerical sensitivity, not accepted new-support ranking'))
 pd.concat(ranks).to_csv(R/'Alice_rank_impact.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);alice(json.loads(Path(p.parse_args().paths).read_text(encoding='utf8')))

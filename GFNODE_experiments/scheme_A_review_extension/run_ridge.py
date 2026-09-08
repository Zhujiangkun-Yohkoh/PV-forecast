"""Post hoc deterministic linear references; never calls a neural training function."""
from pathlib import Path
import argparse,json,sys,time,pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'GFNODE_experiments/scheme_A_multisite_extension'))
sys.path.insert(0,str(ROOT/'GFNODE_experiments/scheme_A_submission_correction'))
import run_multisite_benchmark as m
import run_corrected_benchmark as b
from analyze_predictions import stats

def run(path):
 cfg=json.loads(Path(path).read_text(encoding='utf8'));out=Path(cfg['new_results']);out.mkdir(exist_ok=True);scores=[];choices=[]
 for site in ['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']:
  start=time.perf_counter();external=site in ['YULARA_COMBINED','NIST_GROUND'];folder=out/site;folder.mkdir(exist_ok=True)
  if external:
   c=m.a.config();frame=m.load_site(cfg['external_paths'],site,c);bounds=c['splits'];splits={s:m.split_frame(frame,c,s) for s in bounds}
   proc=m.Processor(c).fit(splits['train'],'train');xs={s:proc.transform(f) for s,f in splits.items()};target=proc.target;threshold=proc.daylight;power=frame.power
   ref=next(Path(cfg['external_results']).glob(site+'__*seed42/test_predictions.npz'))
  else:
   c=b.load_config();b.resolve_data_path=lambda config,dataset:Path(cfg['alice_raw'][dataset]);proc=b.CorrectedProtocol(c,site);proc.load();proc.fit_preprocessors();bounds=c['splits'];splits={s:proc.split_raw(s) for s in bounds}
   xs={s:proc.feature_scaler.transform(proc._augment(f,proc.knn.transform(f[proc.feature_columns].astype(float)))).astype(np.float32) for s,f in splits.items()};target=proc.target_scaler;threshold=proc.daylight_threshold;power=pd.to_numeric(proc.raw.Active_Power,errors='coerce').where(lambda x:x>=0)
   for s in splits:splits[s]=splits[s].assign(power=pd.to_numeric(splits[s].Active_Power,errors='coerce').where(lambda x:x>=0))
   ref=next(Path(cfg['alice_results']).glob('*_'+site+'_42/test_H144.npz'))
  with np.load(ref) as z:archive={k:z[k] for k in z.files}
  # Refit preprocessing only on Train for the added references, never alter neural processors.
  with open(folder/'ridge_preprocessor.pkl','wb') as f:pickle.dump(proc,f)
  designs={};ys={};os={};daily={}
  for split,framepart in splits.items():
   yy=framepart.power.to_numpy(float);idx=framepart.index
   if split=='test':
    oi=pd.DatetimeIndex(pd.to_datetime(archive['forecast_origin']));pos=idx.get_indexer(oi);assert (pos>=71).all()
    y=archive['labels'];expected=yy[np.minimum(pos[:,None]+np.arange(1,145),len(yy)-1)]
    inside=pos[:,None]+np.arange(1,145)<len(yy);check=inside & archive["target_valid"];assert np.allclose(y[check],expected[check],equal_nan=True,rtol=1e-6,atol=1e-6)
   else:
    pos=np.arange(72 if external else 71,len(yy)-144);future=yy[pos[:,None]+np.arange(1,145)]
    keep=np.isfinite(future).all(1)&np.isfinite(yy[pos]);pos=pos[keep];y=future[keep];oi=idx[pos]
   x=xs[split][pos[:,None]+np.arange(-71,1)].reshape(len(pos),-1).astype(float)
   lag=oi.to_numpy()[:,None]+np.arange(1,145)*pd.Timedelta(minutes=5)-pd.Timedelta(hours=24) if oi.tz is None else None
   # pandas timestamp operations preserve fixed offsets.
   lagtimes=pd.DatetimeIndex([t+pd.Timedelta(minutes=int(j)*5)-pd.Timedelta(hours=24) for t in oi for j in range(1,145)])
   day=power.reindex(lagtimes).to_numpy().reshape(len(pos),144)
   assert all(t+pd.Timedelta(hours=12)-pd.Timedelta(hours=24)<t for t in oi[:1])
   designs[split]=x;ys[split]=y;os[split]=oi;daily[split]=day
  med=np.nanmedian(daily['train'],axis=0);assert np.isfinite(med).all()
  for variant in ['Ridge','Ridge+day']:
   X={s:np.concatenate([x,np.where(np.isfinite(daily[s]),daily[s],med),~np.isfinite(daily[s])],axis=1) if variant.endswith('day') else x for s,x in designs.items()}
   scale=StandardScaler().fit(X['train']);X={s:scale.transform(x) for s,x in X.items()}
   Y=target.transform(ys['train'].reshape(-1,1)).reshape(ys['train'].shape);mean=Y.mean(0);Y-=mean
   gram=X['train'].T@X['train'];rhs=X['train'].T@Y;values,vectors=np.linalg.eigh(gram);values=np.maximum(values,0);project=vectors.T@rhs
   best=None
   for alpha in [0.1,1,10,100,1000]:
    coef=vectors@(project/(values[:,None]+alpha));pv=target.inverse_transform((X['validation']@coef+mean).reshape(-1,1)).reshape(ys['validation'].shape)
    mse=float(np.mean((pv-ys['validation'])**2));choices.append(dict(site=site,model=variant,alpha=alpha,validation_global_MSE_kW2=mse,train_origins=len(Y),validation_origins=len(pv)))
    if best is None or mse<=best[0]:best=(mse,alpha,coef.copy())
   # Test predictions only after this variant's alpha and preprocessing are fixed.
   pred=target.inverse_transform((X['test']@best[2]+mean).reshape(-1,1)).reshape(ys['test'].shape);assert np.isfinite(pred).all()
   np.savez_compressed(folder/(variant.replace('+','_')+'_predictions.npz'),predictions=pred,labels=ys['test'],forecast_origin=archive['forecast_origin'],target_start=archive['target_start'],target_valid=archive['target_valid'],daily=daily['test'],last_power=archive['last_power'],daylight_threshold=threshold)
   np.savez_compressed(folder/(variant.replace('+','_')+'_coefficients.npz'),coef=best[2],intercept=mean,feature_mean=scale.mean_,feature_scale=scale.scale_,daily_train_median=med,target_scale=target.scale_,target_min=target.min_)
   (folder/(variant.replace('+','_')+'_completed.json')).write_text(json.dumps(dict(site=site,model=variant,selected_alpha=best[1],validation_MSE=best[0],post_hoc=True,seed=None,seconds=time.perf_counter()-start),indent=2))
   y=ys['test'];valid=archive['target_valid'];day=daily['test']
   for h in [12,48,96,144]:
    for support in ['horizon_specific','common_H144']:
     mask=valid[:,:h]&valid[:,:h if support=='horizon_specific' else 144].all(1)[:,None]&np.isfinite(archive['last_power'])[:,None]
     for analysis in ['primary','daily_matched']:
      mm=mask if analysis=='primary' else mask&np.isfinite(day[:,:h])
      for scope,sm in [('full',mm),('power-active',mm&(y[:,:h]>threshold)),('low-power',mm&(y[:,:h]<=threshold))]:scores.append(dict(site=site,model=variant,h=h,support=support,analysis=analysis,scope=scope,selected_alpha=best[1],**stats(y[:,:h],pred[:,:h],sm)))
   print(site,variant,'alpha',best[1],'seconds',time.perf_counter()-start,flush=True)
  pd.DataFrame(scores).to_csv(HERE/'results/ridge_metrics.csv',index=False);pd.DataFrame(choices).to_csv(HERE/'results/ridge_validation_selection.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True)
 with threadpool_limits(limits=4):run(p.parse_args().paths)

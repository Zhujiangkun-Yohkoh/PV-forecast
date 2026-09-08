"""Reconstruct original Ridge designs from saved Train-only transformations."""
from pathlib import Path
import json,sys,pickle
import numpy as np
import pandas as pd
from sklearn import set_config
set_config(working_memory=64)  # Distance chunking only; no changed neighbors.
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE.parent/'scheme_A_multisite_extension'),str(HERE.parent/'scheme_A_submission_correction')]
import run_multisite_benchmark as m
import run_corrected_benchmark as b
SITES=['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']
def load_npz(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def reconstruct(paths,site):
 cfg=json.loads(Path(paths).read_text());folder=Path(cfg['new_results'])/site;external=site in SITES[3:]
 with (folder/'ridge_preprocessor.pkl').open('rb') as f:proc=pickle.load(f)
 if external:
  config=m.a.config();frame=m.load_site(cfg['external_paths'],site,config);splits={s:m.split_frame(frame,config,s) for s in config['splits']};xs={s:proc.transform(f) for s,f in splits.items()};target=proc.target;power=frame.power;channels=config['input_channels']
 else:
  proc.data_path=Path(cfg['alice_raw'][site]);proc.load();config=proc.cfg;splits={s:proc.split_raw(s) for s in config['splits']};xs={s:proc.feature_scaler.transform(proc._augment(f,proc.knn.transform(f[proc.feature_columns].astype(float)))).astype(np.float32) for s,f in splits.items()};target=proc.target_scaler;power=pd.to_numeric(proc.raw.Active_Power,errors='coerce').where(lambda x:x>=0);channels=proc.feature_columns+[x+'_missing' for x in proc.feature_columns]+['isolation_forest_flag'];splits={s:f.assign(power=pd.to_numeric(f.Active_Power,errors='coerce').where(lambda x:x>=0)) for s,f in splits.items()}
 print(site,'saved transforms reconstructed',flush=True)
 archive=load_npz(folder/'Ridge_predictions.npz');design={};labels={};origins={};daily={};align=[]
 for split,part in splits.items():
  idx=part.index.as_unit('ns');yy=part.power.to_numpy(float);assert (np.diff(idx.asi8)==300_000_000_000).all()
  if split=='test':
   oi=pd.DatetimeIndex(pd.to_datetime(archive['forecast_origin'])).as_unit('ns');pos=idx.get_indexer(oi);assert (pos>=71).all();y=archive['labels'];t=pos[:,None]+np.arange(1,145);inside=t<len(yy);expected=np.where(inside,yy[np.minimum(t,len(yy)-1)],np.nan);valid=archive['target_valid'];assert np.allclose(y[valid],expected[valid],rtol=1e-6,atol=1e-6,equal_nan=True);assert np.array_equal(valid,np.isfinite(expected));assert pd.DatetimeIndex(pd.to_datetime(archive['target_start'])).as_unit('ns').equals(oi+pd.Timedelta(minutes=5))
  else:
   pos=np.arange(72 if external else 71,len(yy)-144);future=yy[pos[:,None]+np.arange(1,145)];keep=np.isfinite(future).all(1)&np.isfinite(yy[pos]);pos=pos[keep];y=future[keep];oi=idx[pos]
  design[split]=xs[split][pos[:,None]+np.arange(-71,1)].reshape(len(pos),-1).astype(float);labels[split]=y;origins[split]=oi
  lag_ns=(oi.asi8[:,None]+np.arange(1,145)*300_000_000_000-86_400_000_000_000).ravel()
  lagtimes=pd.DatetimeIndex(pd.to_datetime(lag_ns,utc=True)).tz_convert(oi.tz) if oi.tz is not None else pd.DatetimeIndex(pd.to_datetime(lag_ns))
  explicit=pd.DatetimeIndex([t+pd.Timedelta(minutes=j*5)-pd.Timedelta(hours=24) for t in oi[:2] for j in range(1,145)]);assert lagtimes[:288].equals(explicit)
  daily[split]=power.reindex(lagtimes).to_numpy().reshape(len(pos),144);assert np.all(lagtimes.asi8.reshape(-1,144)<=oi.asi8[:,None]);assert np.all(idx[pos-71].asi8==oi.asi8-71*300_000_000_000)
  align.append(dict(site=site,split=split,origins=len(pos),first=str(oi[0]),last=str(oi[-1]),first_target=str(oi[0]+pd.Timedelta(minutes=5)),first_lag=str(lagtimes[0]),last_lag_at_first_origin=str(lagtimes[143]),history_start=str(idx[pos[0]-71]),timezone=str(idx.tz),label_and_timestamp_alignment=True))
 names=[f'lag{71-i}:{ch}' for i in range(72) for ch in channels]
 return dict(folder=folder,design=design,labels=labels,origins=origins,daily=daily,target=target,archive=archive,channels=channels,names=names,alignment=align,config=config)
def variant_design(d,variant):
 c=load_npz(d['folder']/(variant.replace('+','_')+'_coefficients.npz'));names=d['names'];med=c['daily_train_median'];raw={s:np.concatenate([x,np.where(np.isfinite(d['daily'][s]),d['daily'][s],med),~np.isfinite(d['daily'][s])],axis=1) if variant.endswith('day') else x for s,x in d['design'].items()}
 if variant.endswith('day'):names=names+[f'day_power_lead{j}' for j in range(1,145)]+[f'day_missing_lead{j}' for j in range(1,145)]
 X={s:(x-c['feature_mean'])/c['feature_scale'] for s,x in raw.items()};return raw,X,c,names

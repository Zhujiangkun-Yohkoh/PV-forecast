"""Unified-period candidates with frozen processors; existing 2018 arrays reused."""
from pathlib import Path
import sys,json,argparse,zipfile,io
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scheme_A_fixed_period_review'))
import frozen_evaluation as f
from frozen_evaluation import np,pd,torch,m,b,load,deny,ExitStack,patch,KNNImputer,IsolationForest,MinMaxScaler,threadpool_limits,set_config
from analysis_saved import read,score,R

def frame2017(c,site):
 cfg=m.a.config();paths=json.loads(Path(c['external_paths']).read_text(encoding='utf8'))
 if site=='YULARA_COMBINED':
  raw=pd.read_csv(paths['YULARA_RAW_FILE'],usecols=['timestamp',*cfg['yulara']['fields']],dtype=str,keep_default_na=False);idx=pd.DatetimeIndex(pd.to_datetime(raw.timestamp));keep=(idx>=pd.Timestamp('2017-09-29'))&(idx<pd.Timestamp('2018-01-02'));fr=m.a.common_frame(raw.loc[keep],idx[keep],cfg['yulara']['fields']);fr,excluded=m.a.regular_yulara(fr)
 else:
  # Official January 1 is only the target buffer for December 31 origins.
  files=sorted(Path(paths['NIST_GROUND_2017_DIRECTORY']).glob('*/*.csv'));raws=[]
  for file in files:
   if file.name >= 'onemin-Ground-2017-09-29.csv':raws.append(pd.read_csv(file,dtype=str,keep_default_na=False))
  assert raws,'Explicit official filenames required'
  with zipfile.ZipFile(Path(c['nist_2018'])/'official_2018_ground.zip') as z:raws.append(pd.read_csv(io.BytesIO(z.read('2018/01/onemin-Ground-2018-01-01.csv')),dtype=str,keep_default_na=False))
  raw=pd.concat(raws,ignore_index=True);fr=m.a.common_frame(raw,m.a.fixed_est(raw.TIMESTAMP),cfg['nist']['fields']);fr=m.a.aggregate_nist(fr)
 return fr.reindex(pd.date_range('2017-09-29','2018-01-01 23:55',freq='5min',tz=fr.index.tz))

def evaluate(c,site):
 dest=Path(c['closeout_destination'])/site;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'unified_2017_predictions.npz').exists():raise FileExistsError('Preserve existing evidence: use reduce_periods.py, or an explicitly new destination for fresh inference')
 cfg=m.a.config();proc=m.Processor(cfg);proc.__dict__.update(load(Path(c['external_results'])/site/'preprocessors.pkl'));rp=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl')
 fr=frame2017(c,site);idx=fr.index.as_unit('ns');pos=np.flatnonzero((idx>=pd.Timestamp('2017-10-01',tz=idx.tz))&(idx<=pd.Timestamp('2017-12-31 23:55',tz=idx.tz)));o=idx[pos];assert pos.min()>=288 and pos.max()+144<len(idx);x=proc.transform(fr);rx=rp.transform(fr)
 yy=fr.power.to_numpy(float)[pos[:,None]+np.arange(1,145)];last=fr.power.to_numpy(float)[pos];daily,tt=f.lag_lookup(fr.power,o);preds=f.ridge_predictions(c,site,rx,pos,daily);audit=[]
 old=read(Path(c['new_results'])/site/'Ridge_predictions.npz');oo=pd.DatetimeIndex(pd.to_datetime(old['forecast_origin'])).as_unit('ns');oi=o.get_indexer(oo);op=idx.get_indexer(oo);assert (oi>=0).all();assert np.allclose(yy[oi][old['target_valid']],old['labels'][old['target_valid']],equal_nan=True);assert np.allclose(daily[oi],old['daily'],equal_nan=True)
 original_neural={}
 for key,p,folder in f.neural(c,site,x,op,proc.target,True):
  audit.append(f.check(p,read(folder/'test_predictions.npz')['predictions'],key+' original_batch'));original_neural[key]=p
 for grid,root in [('Original',Path(c['new_results'])/site),('Expanded',Path(c['expanded_results'][site]))]:
  for letter,stem in [('A','Ridge'),('B','Ridge_day')]:audit.append(f.check(preds[grid+' '+letter][oi],read(root/(stem+'_predictions.npz'))['predictions'],grid+' '+letter))
 new_indices=np.flatnonzero(~np.isin(np.arange(len(pos)),oi))
 # Canonical inference partition: archival origins in their original chronological
 # batches, then newly eligible origins in chronological batches. Both are NEW
 # forwards of the same checkpoint; no historical saved predictions are spliced.
 for key,p,folder in f.neural(c,site,x,pos[new_indices],proc.target,True):
  merged=np.empty_like(yy,dtype=np.float32);merged[oi]=original_neural[key];merged[new_indices]=p;preds[key]=merged
 preds['Daily']=daily;preds['Last-value']=np.broadcast_to(last[:,None],yy.shape)
 np.savez_compressed(dest/'unified_2017_predictions.npz',labels=yy,forecast_origin=np.asarray(o.astype(str),dtype=str),target_timestamp_ns=tt,threshold=proc.daylight,**{k.replace(' ','_'):v for k,v in preds.items()})
 (R/(site+'_REPLAY.json')).write_text(json.dumps(audit,indent=2),encoding='utf8');print(site,'2017 original batching replay passes; separate new-candidate inference; unified candidates',len(o),flush=True)
 from reduce_periods import reduce_site
 reduce_site(c,site)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--sites',nargs='+',default=['YULARA_COMBINED','NIST_GROUND']);a=p.parse_args();c=json.loads(Path(a.paths).read_text(encoding='utf8'));set_config(working_memory=64);torch.set_num_threads(4)
 with ExitStack() as stack,threadpool_limits(limits=4):
  for cls,name in [(KNNImputer,'fit'),(IsolationForest,'fit'),(MinMaxScaler,'fit'),(torch.Tensor,'backward'),(torch,'save')]:stack.enter_context(patch.object(cls,name,deny))
  for site in a.sites:evaluate(c,site)


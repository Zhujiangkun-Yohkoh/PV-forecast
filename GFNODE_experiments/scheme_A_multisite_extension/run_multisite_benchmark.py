"""M2: frozen 24-run training, with Test opened only after all checkpoints are fixed."""
from __future__ import annotations
import argparse
import csv
import json
import math
import pickle
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import audit_multisite_data as a

HERE=a.HERE
# Outside M1 directory, so its unchanged artifact-absence tests remain meaningful.
RESULTS=a.REPO/'.local'/'multisite_results'
SOURCE='e60f3482248cc657397db2b665ea0f8957dee7c5'
BENCH=a.load_model_module()

def write_json(path, value):
    path.write_text(json.dumps(a.json_safe(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def frozen_config():
    cfg=a.config()
    original=json.loads(subprocess.check_output(['git','-c','safe.directory='+a.REPO.as_posix(),'show',
        SOURCE+':GFNODE_experiments/scheme_A_multisite_extension/multisite_config.json'],cwd=a.REPO).decode('utf-8'))
    if cfg!=original:raise RuntimeError('PREFLIGHT: frozen M1-R configuration changed')
    return cfg

def source_snapshot(paths):
    y,_,files=a.validate_paths(paths)
    return {p:a.stats(p) for p in [y,*files]}

def assert_unchanged(snapshot):
    if any(a.stats(p)!=s for p,s in snapshot.items()):raise RuntimeError('RAW_SOURCE_CHANGED')

def load_site(paths,site,cfg):
    y,n,files=a.validate_paths(paths)
    if site=='NIST_GROUND':
        cols=['TIMESTAMP',*cfg['nist']['fields']]
        raw=pd.concat([pd.read_csv(f,usecols=cols,dtype=str,keep_default_na=False) for f in files],ignore_index=True)
        frame=a.common_frame(raw,a.fixed_est(raw.TIMESTAMP),cfg['nist']['fields'])
        frame=a.aggregate_nist(frame)
        return frame.reindex(pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min',tz=a.EST))
    if site!='YULARA_COMBINED':raise ValueError(site)
    raw=pd.read_csv(y,usecols=['timestamp',*cfg['yulara']['fields']],dtype=str,keep_default_na=False)
    idx=pd.DatetimeIndex(pd.to_datetime(raw.timestamp,errors='raise'))
    keep=(idx>=pd.Timestamp('2017-01-01'))&(idx<pd.Timestamp('2018-01-01'))
    frame=a.common_frame(raw.loc[keep],idx[keep],cfg['yulara']['fields'])
    frame,excluded=a.regular_yulara(frame)
    if [str(t) for t in excluded.index]!=['2017-02-22 15:30:02','2017-08-07 05:20:01']:raise RuntimeError('PREFLIGHT: off-grid changed')
    return frame.reindex(pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min'))

class Processor:
    def __init__(self,cfg):self.cfg=cfg;self.fit_log=[]
    def fit(self,frame,split):
        if split!='train':raise ValueError('Train-only fit')
        raw=frame[a.FIELDS].to_numpy(float)
        masks=a.missing_masks(raw)  # record before any imputation
        self.knn=a.train_only_fit(KNNImputer(n_neighbors=5),raw,split)
        filled=self.knn.transform(raw)
        if filled.shape!=raw.shape:raise ValueError('Empty Train feature')
        self.iforest=a.train_only_fit(IsolationForest(n_estimators=100,contamination=.01,random_state=42),filled,split)
        augmented=np.concatenate([filled,masks,(self.iforest.predict(filled)==-1)[:,None]],axis=1)
        self.feature=a.train_only_fit(MinMaxScaler(),augmented,split)
        target=frame.power.to_numpy(float);target=target[np.isfinite(target)]
        self.target=a.train_only_fit(MinMaxScaler(),target[:,None],split)
        self.minimum=float(target.min());self.maximum=float(target.max());self.range=self.maximum-self.minimum
        self.daylight=.01*self.maximum
        self.fit_log=[{'component':v,'split':split,'first':str(frame.index[0]),'last':str(frame.index[-1]),'rows':len(frame)}
                      for v in ['KNNImputer','IsolationForest','feature_MinMaxScaler','target_MinMaxScaler']]
        return self
    def transform(self,frame):
        raw=frame[a.FIELDS].to_numpy(float);filled=self.knn.transform(raw)
        augmented=a.augment_seven(raw,filled,(self.iforest.predict(filled)==-1).astype(int))
        x=self.feature.transform(augmented).astype(np.float32)
        if x.shape[1]!=7 or not np.isfinite(x).all():raise ValueError('Nonfinite/invalid inputs')
        return x
    def metadata(self):
        return {'fit_log':self.fit_log,'channels':self.cfg['input_channels'],'train_min':self.minimum,
                'train_max':self.maximum,'train_range':self.range,'daylight_threshold':self.daylight,
                'feature_min':self.feature.data_min_.tolist(),'feature_max':self.feature.data_max_.tolist(),
                'target_scale':self.target.scale_.tolist(),'target_offset':self.target.min_.tolist(),
                'knn_neighbors':5,'if_parameters':self.cfg['isolation_forest'],'label_imputation':False}

def split_frame(frame,cfg,split):
    return frame.loc[slice(*a.split_coordinates(frame.index,cfg['splits'][split]))]

class Windows(Dataset):
    def __init__(self,frame,x,processor,horizon,bounds):
        self.frame,self.origins,_=a.eligible(frame,horizon,bounds)
        if not self.frame.index.equals(frame.index):raise ValueError('Pass split-local inputs')
        self.x=torch.from_numpy(x);self.processor=processor
        self.raw=frame.power.to_numpy(float)
        self.horizon=horizon
    def __len__(self):return len(self.origins)
    def __getitem__(self,j):
        o=self.origins[j];y=self.raw[o+1:o+145]
        if len(y)!=144 or not np.isfinite(y).all():raise ValueError('Training requires full H144')
        scaled=self.processor.target.transform(y[:,None]).ravel().astype(np.float32)
        return self.x[o-71:o+1],torch.from_numpy(scaled),torch.ones(144,dtype=torch.bool)

def loader(w,shuffle):return DataLoader(w,batch_size=256,shuffle=shuffle,num_workers=0,pin_memory=True)

def finite_prediction(prediction):
    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite neural prediction: whole run fails')

def global_mse(model,validation_loader,device):
    model.eval();sse=0.;count=0
    with torch.inference_mode():
        for x,y,mask in validation_loader:
            p=model(x.to(device));y=y.to(device);mask=mask.to(device)
            if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite Validation prediction')
            e=p[mask]-y[mask];sse+=float(e.square().sum().cpu());count+=int(mask.sum().cpu())
    if not count:raise ValueError('No Validation targets')
    return sse/count

def choose_checkpoint(value,best,min_delta):
    if not math.isfinite(value):raise FloatingPointError('Nonfinite Validation MSE')
    return value<best-min_delta

def train_one(model,train_loader,validation_loader,cfg,device,folder,identity):
    tc=cfg['future_training'];opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    started=time.perf_counter();best=math.inf;best_epoch=0;stale=0;history=[]
    for epoch in range(1,tc['max_epochs']+1):
        model.train();sse=0.;count=0;clock=time.perf_counter()
        for x,y,mask in train_loader:
            opt.zero_grad(set_to_none=True);p=model(x.to(device));y=y.to(device);mask=mask.to(device)
            e=p[mask]-y[mask];loss=e.square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if not all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip_norm']);opt.step()
            sse+=float(e.detach().square().sum().cpu());count+=int(mask.sum().cpu())
        value=global_mse(model,validation_loader,device)
        history.append({'epoch':epoch,'train_mse':sse/count,'validation_global_mse':value,'seconds':time.perf_counter()-clock})
        if choose_checkpoint(value,best,tc['min_delta']):
            best=value;best_epoch=epoch;stale=0
            torch.save({'state_dict':model.state_dict(),'identity':identity,'input_dim':7,'lookback':72,'output_horizon':144,
                        'epoch':epoch,'validation_global_mse':value,'config':cfg},folder/'best_validation.pt')
        else:stale+=1
        write_json(folder/'training_history.json',history)
        print(json.dumps({'run':identity['run_id'],'epoch':epoch,'validation_mse':value,'best_epoch':best_epoch,'seconds':history[-1]['seconds']}),flush=True)
        if stale>=tc['patience']:break
    torch.cuda.synchronize()
    return {'best_epoch':best_epoch,'best_validation_mse':best,'actual_epochs':epoch,'training_seconds':time.perf_counter()-started,
            'parameter_count':BENCH.parameter_count(model),'numeric_failure':False}

def environment():
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'device':'cuda:0','numpy':np.__version__,
            'cpu_threads':torch.get_num_threads(),'precision':'float32; no AMP','cudnn_deterministic':True}

def build_test(frame,processor,cfg,freeze):
    if not freeze.get('all_24_checkpoints_fixed') or freeze.get('config')!=cfg:raise RuntimeError('Test is locked')
    f=split_frame(frame,cfg,'test');x=processor.transform(f)
    _,o,_=a.eligible(f,12,cfg['splits']['test']);t=o[:,None]+np.arange(1,145)
    inside=t<len(f);safe=np.minimum(t,len(f)-1)
    labels=np.where(inside,f.power.to_numpy(float)[safe],np.nan)
    daily_timeline=a.daily_lookup(frame.power,f.index.to_numpy())
    daily=np.where(inside,daily_timeline[safe],np.nan)
    # Strings retain NIST offset and do not invent an offset for Yulara.
    origin=np.asarray([str(v) for v in f.index[o]])
    starts=np.asarray([str(v+a.FIVE) for v in f.index[o]])
    return {'x':x,'positions':o,'labels':labels,'target_valid':np.isfinite(labels),
            'forecast_origin':origin,'target_start':starts,'last_power':f.power.to_numpy(float)[o],
            'daily':daily,'daylight_threshold':np.asarray(processor.daylight),'train_range':np.asarray(processor.range)}

def predict(model,bundle,processor,device):
    outputs=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(bundle['positions']),256):
            origins=bundle['positions'][start:start+256]
            x=bundle['x'][origins[:,None]-np.arange(71,-1,-1)]
            p=model(torch.from_numpy(x).to(device)).cpu().numpy();finite_prediction(p);outputs.append(p)
    scaled=np.concatenate(outputs)
    result=processor.target.inverse_transform(scaled.reshape(-1,1)).reshape(scaled.shape).astype(np.float32)
    finite_prediction(result);return result

def point_masks(bundle,h,scope):
    y=bundle['labels'][:,:h];eligible=np.isfinite(y).all(axis=1)&np.isfinite(bundle['last_power'])
    primary=np.broadcast_to(eligible[:,None],y.shape).copy()
    if scope=='daylight':primary &= y>float(bundle['daylight_threshold'])
    return primary,primary&np.isfinite(bundle['daily'][:,:h])

def metrics(y,p,mask,denominator):
    actual=y[mask].astype(np.float64);pred=p[mask].astype(np.float64);e=pred-actual
    if not len(e):return dict(RMSE=math.nan,MAE=math.nan,bias=math.nan,R2=math.nan,nRMSE=math.nan)
    if not np.isfinite(e).all():raise FloatingPointError('Nonfinite metric points')
    rmse=float(np.sqrt(np.mean(e*e)));ss=float(np.sum((actual-actual.mean())**2))
    return {'RMSE':rmse,'MAE':float(np.mean(np.abs(e))),'bias':float(e.mean()),
            'R2':1-float(np.sum(e*e))/ss if ss else math.nan,'nRMSE':rmse/denominator if denominator else math.nan}

def skill(r,baseline):return 1-r/baseline if baseline>0 else math.nan

def metric_rows(info,bundle,pred,cfg):
    finite_prediction(pred);rows=[]
    for h in cfg['evaluation_horizons']:
        y=bundle['labels'][:,:h];last=np.broadcast_to(bundle['last_power'][:,None],y.shape);daily=bundle['daily'][:,:h]
        for scope in ['full','daylight']:
            primary,matched=point_masks(bundle,h,scope)
            for analysis,mask in [('primary',primary),('supplementary_daily_matched',matched)]:
                lv=metrics(y,last,mask,float(bundle['train_range']));dv=metrics(y,daily,matched,float(bundle['train_range']))
                methods=[(info['model'],pred[:,:h]),('LAST_VALUE_PERSISTENCE',last)]
                if analysis!='primary':methods.append(('DAILY_PERSISTENCE',daily))
                for model,p in methods:
                    v=metrics(y,p,mask,float(bundle['train_range']))
                    v.update(last_value_skill=skill(v['RMSE'],lv['RMSE']),daily_skill=skill(v['RMSE'],dv['RMSE']) if analysis!='primary' else math.nan)
                    rows.append({'site':info['site'],'model':model,'seed':info['seed'],'horizon':h,'scope':scope,'analysis':analysis,
                                 'forecast_origin_count':int(mask.any(axis=1).sum()),'valid_target_count':int(mask.sum()),**v,
                                 **{k:info[k] if model==info['model'] else math.nan for k in ['parameter_count','best_epoch','best_validation_mse','training_seconds']}})
    return rows

def write_tables(rows):
    df=pd.DataFrame(rows).drop_duplicates(['site','model','seed','horizon','scope','analysis'])
    df.to_csv(HERE/'metrics_per_seed.csv',index=False)
    keys=['site','model','horizon','scope','analysis'];values=['RMSE','MAE','bias','R2','nRMSE','last_value_skill','daily_skill']
    groups=df.groupby(keys,sort=False);summary=groups[values].agg(['mean','std'])
    summary.columns=[m+'_'+('sample_sd' if s=='std' else s) for m,s in summary.columns]
    summary=summary.reset_index();summary['seed_count']=groups.size().to_numpy()
    for count in ['forecast_origin_count','valid_target_count']:summary[count]=groups[count].first().to_numpy()
    summary.to_csv(HERE/'metrics_summary_mean_sd.csv',index=False)

def verify_identity(folder,identity,cfg):
    state=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=False)
    if state['identity']!=identity or state['config']!=cfg or [state[k] for k in ['input_dim','lookback','output_horizon']]!=[7,72,144]:raise RuntimeError('STALE_ARTIFACT: identity/config')
    info=json.loads((folder/'training_complete.json').read_text(encoding='utf-8'))
    if info['best_epoch']!=state['epoch'] or info['best_validation_mse']!=state['validation_global_mse']:raise RuntimeError('STALE_ARTIFACT: checkpoint selection')
    if (folder/'completed.json').exists():
        completed=json.loads((folder/'completed.json').read_text(encoding='utf-8'))
        if completed.get('config')!=cfg or any(completed.get(k)!=v for k,v in identity.items()):raise RuntimeError('STALE_ARTIFACT: completed metadata')
    return state,info

def run(paths):
    cfg=frozen_config();before=source_snapshot(paths)
    if not torch.cuda.is_available():raise RuntimeError('GPU_REQUIRED')
    torch.set_num_threads(4);device=torch.device('cuda:0');RESULTS.mkdir(parents=True,exist_ok=True)
    preflight=json.loads((HERE/'M2_PREFLIGHT.json').read_text(encoding='utf-8'))
    if preflight.get('m1_passed')!=43 or preflight.get('m2_passed',0)<17 or preflight.get('skipped')!=0:raise RuntimeError('PREFLIGHT_REQUIRED')
    env=environment();write_json(RESULTS/'environment.json',env);started=time.perf_counter()
    protocols={}
    for site in ['YULARA_COMBINED','NIST_GROUND']:
        print('PREPROCESS '+site,flush=True);frame=load_site(paths,site,cfg);folder=RESULTS/site;folder.mkdir(exist_ok=True)
        train=split_frame(frame,cfg,'train');validation=split_frame(frame,cfg,'validation')
        state_path=folder/'preprocessors.pkl'
        if state_path.exists():
            processor=Processor(cfg)
            with state_path.open('rb') as stream:processor.__dict__.update(pickle.load(stream))
            if processor.cfg!=cfg:raise RuntimeError('STALE_ARTIFACT: preprocessors')
        else:
            processor=Processor(cfg).fit(train,'train')
            with state_path.open('wb') as stream:pickle.dump(processor.__dict__,stream)
            write_json(folder/'preprocessing.json',processor.metadata())
        xtrain=processor.transform(train);xval=processor.transform(validation)
        train_loader=loader(Windows(train,xtrain,processor,144,cfg['splits']['train']),True)
        val_loader=loader(Windows(validation,xval,processor,144,cfg['splits']['validation']),False)
        protocols[site]=(frame,processor)
        for identity in [r for r in cfg['run_matrix'] if r['site']==site]:
            run_dir=RESULTS/identity['run_id']
            if run_dir.exists():
                verify_identity(run_dir,identity,cfg);continue
            run_dir.mkdir();BENCH.set_seed(identity['seed'])
            model=BENCH.make_model(cfg['models'][identity['model']],7,cfg['model_configuration']).to(device)
            info=train_one(model,train_loader,val_loader,cfg,device,run_dir,identity)
            info.update(identity);info.update(environment=env,python_seed=identity['seed'],numpy_seed=identity['seed'],torch_cpu_seed=identity['seed'],torch_cuda_seed=identity['seed'])
            write_json(run_dir/'training_complete.json',info)
            del model;torch.cuda.empty_cache();assert_unchanged(before)
        del train_loader,val_loader,xtrain,xval
    freeze={'all_24_checkpoints_fixed':True,'config':cfg,'run_matrix':cfg['run_matrix'],'baseline':'real origin and exact timestamp minus24h',
            'evaluation':'horizon-specific full/daylight; identical Daily point intersection','test_description':'held-out performance split',
            'checkpoints':[{'run_id':r['run_id'],**{k:verify_identity(RESULTS/r['run_id'],r,cfg)[1][k] for k in ['best_epoch','best_validation_mse']}} for r in cfg['run_matrix']],
            'processors':{site:processor.metadata() for site,(_,processor) in protocols.items()}}
    write_json(RESULTS/'test_release.json',freeze)
    allrows=[]
    for site,(frame,processor) in protocols.items():
        bundle=build_test(frame,processor,cfg,freeze)
        for identity in [r for r in cfg['run_matrix'] if r['site']==site]:
            folder=RESULTS/identity['run_id'];state,info=verify_identity(folder,identity,cfg)
            model=BENCH.make_model(cfg['models'][identity['model']],7,cfg['model_configuration']).to(device);model.load_state_dict(state['state_dict'])
            clock=time.perf_counter();prediction=predict(model,bundle,processor,device)
            artifact=folder/'test_predictions.npz'
            if artifact.exists():
                with np.load(artifact) as saved:
                    for key in ['labels','target_valid','forecast_origin','target_start','last_power','daily']:
                        if not np.array_equal(saved[key],bundle[key],equal_nan= saved[key].dtype.kind=='f'):raise RuntimeError('STALE_ARTIFACT: '+key)
                    if not np.allclose(saved['predictions'],prediction,rtol=2e-5,atol=2e-5):raise RuntimeError('STALE_ARTIFACT: prediction reproduction')
                    for h in cfg['evaluation_horizons']:
                        for scope in ['full','daylight']:
                            for name,mask in zip(['primary','daily'],point_masks(bundle,h,scope)):
                                if not np.array_equal(saved[f'{name}_H{h}_{scope}'],mask):raise RuntimeError('STALE_ARTIFACT: saved point mask')
                prediction=np.load(artifact)['predictions']
            else:
                arrays={k:v for k,v in bundle.items() if k not in ['x','positions']}
                for h in cfg['evaluation_horizons']:
                    for scope in ['full','daylight']:
                        pm,dm=point_masks(bundle,h,scope);arrays[f'primary_H{h}_{scope}']=pm;arrays[f'daily_H{h}_{scope}']=dm
                np.savez_compressed(artifact,predictions=prediction,**arrays)
            info.update(prediction_seconds=time.perf_counter()-clock,config=cfg,test_opened_after_all_24_checkpoints=True)
            write_json(folder/'completed.json',info);allrows.extend(metric_rows(info,bundle,prediction,cfg))
            print('COMPLETED '+identity['run_id'],flush=True);del model;torch.cuda.empty_cache()
    write_tables(allrows);assert_unchanged(before)
    write_json(RESULTS/'execution_summary.json',{'runs':24,'wall_seconds':time.perf_counter()-started,'raw_unchanged':True,'environment':env})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paths',required=True);args=parser.parse_args();run(args.paths)

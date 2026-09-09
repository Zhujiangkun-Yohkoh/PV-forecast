"""Forward-only support/period evaluation. Runtime guards prohibit fitting/saving weights."""
from pathlib import Path
import argparse,json,pickle,sys,time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from sklearn import set_config
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE.parent/'scheme_A_multisite_extension'),str(HERE.parent/'scheme_A_submission_correction')]
import run_multisite_benchmark as m
import run_corrected_benchmark as b
from expanded_intervals import read
from paired import reduce_blocks,intervals
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
def deny(*args,**kwargs):raise RuntimeError('Frozen evaluation forbids fitting, backward, and checkpoint writes')
def load(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def savejson(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def transform_alice(proc,part):return proc.feature_scaler.transform(proc._augment(part,proc.knn.transform(part[proc.feature_columns].astype(float)))).astype(np.float32)
def lag_lookup(power,origins):
    o=pd.DatetimeIndex(origins).as_unit('ns');t=o.asi8[:,None]+np.arange(1,145)*300_000_000_000
    lag=t-86_400_000_000_000;assert (lag<=o.asi8[:,None]).all()
    idx=pd.to_datetime(lag.ravel(),utc=o.tz is not None)
    if o.tz is not None:idx=idx.tz_convert(o.tz)
    return power.reindex(idx).to_numpy(float).reshape(-1,144),t
def ridge_predictions(c,site,x,pos,daily):
    result={};flat=x[pos[:,None]+np.arange(-71,1)].reshape(len(pos),-1).astype(float)
    for grid,root in [('Original',Path(c['new_results'])/site),('Expanded',Path(c['expanded_results'][site]))]:
        for label,stem in [('A','Ridge'),('B','Ridge_day')]:
            z=read(root/(stem+'_coefficients.npz'));raw=flat
            if label=='B':raw=np.concatenate([flat,np.where(np.isfinite(daily),daily,z['daily_train_median']),~np.isfinite(daily)],axis=1)
            result[grid+' '+label]=(((raw-z['feature_mean'])/z['feature_scale'])@z['coef']+z['intercept']-z['target_min'])/z['target_scale']
    return result
def neural(c,site,x,pos,target,external):
    root=Path(c['external_results' if external else 'alice_results']);items=[]
    for f in root.glob('*/completed.json'):
        info=json.loads(f.read_text(encoding='utf8'))
        if info.get('site',info.get('dataset'))==site and (not external or 'inverted' in info['model'].lower()):items.append((f,info))
    assert len(items)==(3 if external else 12)
    for f,info in items:
        b.set_seed(info['seed'])  # reproduce the saved deterministic cuDNN policy
        state=torch.load(f.parent/'best_validation.pt',map_location='cpu',weights_only=False)
        cfg=m.a.config() if external else b.load_config();name=cfg['models'][info['model']] if external else info['model']
        model=b.make_model(name,7 if external else 17,cfg['model_configuration'] if external else cfg).cuda().eval();model.load_state_dict(state['state_dict'],strict=True)
        parts=[]
        with torch.inference_mode():
            for start in range(0,len(pos),256):
                xx=x[pos[start:start+256,None]+np.arange(-71,1)]
                parts.append(model(torch.from_numpy(xx).cuda()).cpu().numpy())
        scaled=np.concatenate(parts);p=target.inverse_transform(scaled.reshape(-1,1)).reshape(scaled.shape).astype(np.float32)
        assert np.isfinite(p).all()
        key=('Inverted' if 'inverted' in info['model'].lower() else info['model'])+str(info['seed'])
        yield key,p,f.parent
        del model,state;torch.cuda.empty_cache()
def check(p,expected,what):
    error=float(np.max(np.abs(p.astype(float)-expected.astype(float))))
    if not np.allclose(p,expected,rtol=2e-5,atol=2e-5,equal_nan=True):raise RuntimeError('Original prediction replay mismatch '+what+' '+str(error))
    return dict(check=what,max_absolute_difference_kW=error,rtol=2e-5,atol=2e-5,passed=True)
def training_support(proc,site):
    rows=[];hours=[]
    for split in ['train','validation']:
        part=proc.split_raw(split);y=pd.to_numeric(part.Active_Power,errors='coerce').to_numpy(float);p=np.arange(71,len(y)-144);future=y[p[:,None]+np.arange(1,145)];valid=np.isfinite(future)&(future>=0)
        for name,keep in [('neural',valid.all(1)),('Ridge',valid.all(1)&np.isfinite(y[p])&(y[p]>=0))]:
            pos=p[keep];o=part.index[pos]
            for h in [12,144]:
                yy=y[pos[:,None]+np.arange(1,h+1)];v=np.isfinite(yy)&(yy>=0)
                rows.append(dict(site=site,split=split,method_support=name,horizon=h,origins=len(pos),points=int(v.sum()),active_points=int((v&(yy>proc.daylight_threshold)).sum()),active_fraction=float((v&(yy>proc.daylight_threshold)).sum()/v.sum()),unique_targets=len(np.unique(pos[:,None]+np.arange(1,h+1)))))
            for hour in range(24):hours.append(dict(site=site,split=split,method_support=name,hour=hour,origins=int((o.hour==hour).sum())))
    pd.DataFrame(rows).to_csv(OUT/(site+'_training_support.csv'),index=False);pd.DataFrame(hours).to_csv(OUT/(site+'_training_hours.csv'),index=False)
def evaluate(site,period,o,y,preds,last,daily,threshold,original_valid,targets,external):
    metrics=[];support=[];blockframes=[];effects=[]
    for h in ([12,48,96,144] if external else [12,144]):
        yy=y[:,:h];finite=np.isfinite(yy);nn=finite&(yy>=0);old=original_valid[:,:h].all(1)
        definitions={'fixed_period':finite} if external else {'Original':nn&old[:,None],'S1':nn,'S2':finite,'S1_new_origins':nn&~old[:,None],'S2_new_origins':finite&~old[:,None]}
        for name,base in definitions.items():
            for scope,sc in [('full',np.ones_like(base)),('power-active',yy>threshold),('low-power',yy<=threshold)]:
                before=base&sc;mask=before&np.isfinite(last)[:,None]&np.isfinite(daily[:,:h]);meta=dict(site=site,period=period,support=name,horizon=h,scope=scope)
                support.append(dict(**meta,candidate_origins=len(o),before_reference_origins=int(before.any(1).sum()),before_reference_points=int(before.sum()),origins=int(mask.any(1).sum()),points=int(mask.sum()),unique_targets=len(np.unique(targets[:,:h][mask])),active_points=int((mask&(yy>threshold)).sum())))
                pp={k:v[:,:h] for k,v in preds.items()}
                for key,p in pp.items():
                    e=p[mask].astype(float)-yy[mask].astype(float)
                    metrics.append(dict(**meta,method=key,origins=int(mask.any(1).sum()),points=len(e),RMSE=float(np.sqrt(np.mean(e*e))) if len(e) else np.nan,MAE=float(np.mean(abs(e))) if len(e) else np.nan,bias=float(e.mean()) if len(e) else np.nan,SSE=float(e@e)))
                comps=[('Expanded B',r) for r in ['Original B','Expanded A','Daily','Inverted42','Inverted43','Inverted44','Inverted mean']]+[(f'Inverted{s}','Daily') for s in [42,43,44]]
                for hours in [24,48,72]:
                    block=reduce_blocks(yy,{k:v for k,v in pp.items() if k in ['Original B','Expanded A','Expanded B','Daily','Inverted42','Inverted43','Inverted44']},mask,o,'2018-04-01' if external else '2018-08-08',hours)
                    blockframes.append(block.assign(**meta,block_hours=hours));effects += [dict(**meta,block_hours=hours,**r) for r in intervals(block,comps)]
    stem=site+'_'+period
    for suffix,data in [('metrics',pd.DataFrame(metrics)),('support',pd.DataFrame(support)),('blocks',pd.concat(blockframes)),('intervals',pd.DataFrame(effects))]:data.to_csv(OUT/(stem+'_'+suffix+'.csv'),index=False)
def alice(c,site):
    dest=Path(c['destination'])/site;dest.mkdir(parents=True,exist_ok=True);proc=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl');proc.data_path=Path(c['alice_raw'][site]);proc.load();training_support(proc,site)
    part=proc.split_raw('test');x=transform_alice(proc,part);idx=part.index.as_unit('ns');pos=np.arange(71,len(part)-1);o=idx[pos]
    target_idx=pos[:,None]+np.arange(1,145);inside=target_idx<len(part);raw=pd.to_numeric(part.Active_Power,errors='coerce').to_numpy(float);y=np.where(inside,raw[np.minimum(target_idx,len(part)-1)],np.nan).astype(np.float32);valid=np.isfinite(y)&(y>=0)
    last=np.where(np.isfinite(raw[pos])&(raw[pos]>=0),raw[pos],np.nan).astype(np.float32)
    # A future label beyond split is unscored, but its previous-day INPUT is
    # still available at origin. Do not mask that input by future label support.
    power=pd.to_numeric(proc.raw.Active_Power,errors='coerce').where(lambda q:q>=0);daily,targets=lag_lookup(power,o)
    preds=ridge_predictions(c,site,x,pos,daily);audit=[]
    old=read(Path(c['new_results'])/site/'Ridge_predictions.npz');oi=pd.DatetimeIndex(pd.to_datetime(old['forecast_origin'])).as_unit('ns');lookup=o.get_indexer(oi);assert (lookup>=0).all()
    assert np.array_equal(valid[lookup],old['target_valid']);assert np.allclose(y[lookup],old['labels'],equal_nan=True);assert np.allclose(daily[lookup],old['daily'],equal_nan=True)
    for grid,root in [('Original',Path(c['new_results'])/site),('Expanded',Path(c['expanded_results'][site]))]:
        for label,stem in [('A','Ridge'),('B','Ridge_day')]:audit.append(check(preds[grid+' '+label][lookup],read(root/(stem+'_predictions.npz'))['predictions'],grid+' '+label))
    for key,p,folder in neural(c,site,x,pos,proc.target_scaler,False):
        z=read(folder/'test_H144.npz');assert np.array_equal(z['forecast_origin'],old['forecast_origin'])
        try:record=check(p[lookup],z['predictions'],key)
        except RuntimeError as exc:record=dict(check=key,passed=False,error=str(exc),max_absolute_difference_kW=float(np.max(abs(p[lookup]-z['predictions']))),rtol=2e-5,atol=2e-5)
        audit.append(record);preds[key]=p
        np.savez_compressed(dest/(key.replace(' ','_')+'.npz'),predictions=p,forecast_origin=np.asarray(o.astype(str)))
        print(site,key,'frozen-origin replay',record['passed'],record['max_absolute_difference_kW'],flush=True)
    preds['Daily']=daily;preds['Last-value']=np.broadcast_to(last[:,None],y.shape)
    np.savez_compressed(dest/'support_and_ridge.npz',labels=y,target_valid=valid,forecast_origin=np.asarray(o.astype(str)),target_timestamp_ns=targets,last_power=last,daily=daily,**{k.replace(' ','_'):v for k,v in preds.items() if k.startswith(('Original','Expanded'))})
    savejson(OUT/(site+'_REPLAY.json'),audit)
    if not all(r['passed'] for r in audit):
        # Keep generated evidence, but do not promote a numerically unmatched
        # forward path into the requested all-model sensitivity comparison.
        support=[]
        for h in [12,144]:
            yy=y[:,:h];vv=valid[:,:h];oldkeep=vv.all(1)
            for name,mask in [('Original',vv&oldkeep[:,None]),('S1',vv),('S2',np.isfinite(yy)),('S1_new_origins',vv&~oldkeep[:,None]),('S2_new_origins',np.isfinite(yy)&~oldkeep[:,None])]:
                mask=mask&np.isfinite(last)[:,None]&np.isfinite(daily[:,:h]);support.append(dict(site=site,horizon=h,support=name,origins=int(mask.any(1).sum()),points=int(mask.sum()),unique_targets=len(np.unique(targets[:,:h][mask])),active_points=int((mask&(yy>proc.daylight_threshold)).sum()),status='support_only_prediction_replay_unresolved'))
        pd.DataFrame(support).to_csv(OUT/(site+'_support_only.csv'),index=False)
        print(site,'ALL-MODEL SCORING BLOCKED: replay mismatch retained',flush=True);return
    evaluate(site,'Alice_2018_Test',o,y,preds,last,daily,proc.daylight_threshold,valid,targets,False)
    print(site,'all support evaluations complete',flush=True)
def new_external_frame(c,site):
    cfg=m.a.config()
    if site=='YULARA_COMBINED':
        paths=json.loads(Path(c['external_paths']).read_text(encoding='utf8'));raw=pd.read_csv(paths['YULARA_RAW_FILE'],usecols=['timestamp',*cfg['yulara']['fields']],dtype=str,keep_default_na=False);idx=pd.DatetimeIndex(pd.to_datetime(raw.timestamp));keep=(idx>=pd.Timestamp('2018-03-30'))&(idx<pd.Timestamp('2018-07-02'));frame=m.a.common_frame(raw.loc[keep],idx[keep],cfg['yulara']['fields']);frame,excluded=m.a.regular_yulara(frame);excluded.to_csv(OUT/'YULARA_2018_off_grid.csv')
    else:
        files=sorted(Path(c['nist_2018']).glob('*.csv'));assert len(files)==94;raw=pd.concat([pd.read_csv(f,dtype=str,keep_default_na=False) for f in files],ignore_index=True);frame=m.a.common_frame(raw,m.a.fixed_est(raw.TIMESTAMP),cfg['nist']['fields']);frame=m.a.aggregate_nist(frame)
    index=pd.date_range('2018-03-30','2018-07-01 23:55',freq='5min',tz=frame.index.tz);return frame.reindex(index)
def external(c,site):
    dest=Path(c['destination'])/site;dest.mkdir(parents=True,exist_ok=True);cfg=m.a.config();proc=m.Processor(cfg);proc.__dict__.update(load(Path(c['external_results'])/site/'preprocessors.pkl'));ridgeproc=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl');audit=[]
    old=read(Path(c['new_results'])/site/'Ridge_predictions.npz');oldframe=m.load_site(c['external_paths'],site,cfg);part=m.split_frame(oldframe,cfg,'test');oo=pd.DatetimeIndex(pd.to_datetime(old['forecast_origin']));sub=np.arange(len(oo));pos=part.index.get_indexer(oo[sub]);x=proc.transform(part);rx=ridgeproc.transform(part)
    for key,p,folder in neural(c,site,x,pos,proc.target,True):audit.append(check(p,read(folder/'test_predictions.npz')['predictions'][sub],key+' original2017'))
    rp=ridge_predictions(c,site,rx,pos,old['daily'][sub])
    for grid,root in [('Original',Path(c['new_results'])/site),('Expanded',Path(c['expanded_results'][site]))]:
        for label,stem in [('A','Ridge'),('B','Ridge_day')]:audit.append(check(rp[grid+' '+label],read(root/(stem+'_predictions.npz'))['predictions'][sub],grid+' '+label+' original2017'))
    savejson(OUT/(site+'_2017_REPLAY.json'),audit);print(site,'2017 frozen replay passed; now opening fixed 2018 period',flush=True)
    frame=new_external_frame(c,site);x=proc.transform(frame);rx=ridgeproc.transform(frame);idx=frame.index.as_unit('ns');pos=np.flatnonzero((idx>=pd.Timestamp('2018-04-01',tz=idx.tz))&(idx<=pd.Timestamp('2018-06-30 23:55',tz=idx.tz)));o=idx[pos];assert pos.min()>=71 and pos.max()+144<len(frame)
    y=frame.power.to_numpy(float)[pos[:,None]+np.arange(1,145)];last=frame.power.to_numpy(float)[pos];daily,targets=lag_lookup(frame.power,o);preds=ridge_predictions(c,site,rx,pos,daily)
    for key,p,folder in neural(c,site,x,pos,proc.target,True):preds[key]=p;print(site,key,'2018 forward complete',flush=True)
    preds['Daily']=daily;preds['Last-value']=np.broadcast_to(last[:,None],y.shape)
    np.savez_compressed(dest/'fixed_2018_predictions.npz',labels=y,forecast_origin=np.asarray(o.astype(str)),target_timestamp_ns=targets,threshold=proc.daylight,**{k.replace(' ','_'):v for k,v in preds.items()})
    dist=[]
    period=frame.loc[o[0]:o[-1]]
    for j,name in enumerate(cfg['input_channels']):dist.append(dict(site=site,channel=name,rows=len(pos),minimum=float(x[pos,j].min()),maximum=float(x[pos,j].max()),below_zero_fraction=float((x[pos,j]<0).mean()),above_one_fraction=float((x[pos,j]>1).mean())))
    pd.DataFrame(dist).to_csv(OUT/(site+'_2018_input_distribution.csv'),index=False)
    coverage={'site':site,'origin_first':str(o[0]),'origin_last':str(o[-1]),'target_last':str(o[-1]+pd.Timedelta(hours=12)),'candidate_origins':len(o),'raw_coordinate':str(idx.tz),'frame_missing':frame.isna().sum().to_dict(),'period_missing_fraction':period.isna().mean().to_dict(),'negative_power':int((period.power<0).sum()),'negative_GHI':int((period.ghi<0).sum()),'threshold_from_2017':proc.daylight,'fit_calls':0}
    savejson(OUT/(site+'_2018_COVERAGE.json'),coverage);evaluate(site,'2018_Apr_Jun_origins',o,y,preds,last,daily,proc.daylight,np.isfinite(y),targets,True)
    print(site,'fixed-period metrics and intervals complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--sites',nargs='+',required=True);v=p.parse_args();c=json.loads(Path(v.paths).read_text(encoding='utf8'));set_config(working_memory=64);torch.set_num_threads(4)
    with ExitStack() as stack,threadpool_limits(limits=4):
        for cls,name in [(KNNImputer,'fit'),(IsolationForest,'fit'),(MinMaxScaler,'fit'),(torch.Tensor,'backward'),(torch,'save')]:stack.enter_context(patch.object(cls,name,deny))
        for site in v.sites:
            if site in ['Sanyo','Hanwha','Qcells']:alice(c,site)
            else:external(c,site)

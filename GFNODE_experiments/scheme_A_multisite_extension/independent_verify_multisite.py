"""Independent NumPy evidence path; never imports production training/metric functions."""
from pathlib import Path
from datetime import timezone,timedelta
import csv
import json
import math
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'.local'/'multisite_results'
METRICS=['RMSE','MAE','bias','R2','nRMSE','last_value_skill','daily_skill']
KEYS=['site','model','seed','horizon','scope','analysis']

def raw_power(site,cfg,paths):
    if site=='NIST_GROUND':
        folder=Path(paths['NIST_GROUND_2017_DIRECTORY']);pieces=[]
        for day in pd.date_range('2017-01-01','2017-12-31'):
            p=folder/f'{day.month:02d}'/f'onemin-Ground-{day:%Y-%m-%d}.csv'
            r=pd.read_csv(p,usecols=['TIMESTAMP','PwrMtrP_kW_Avg'])
            if not r.TIMESTAMP.str.endswith('-05:00').all():raise AssertionError('EST offset')
            idx=pd.DatetimeIndex(pd.to_datetime(r.TIMESTAMP,utc=True)).tz_convert(timezone(timedelta(hours=-5)))
            pieces.append(pd.Series(pd.to_numeric(r.PwrMtrP_kW_Avg,errors='coerce').to_numpy(float),index=idx))
        s=pd.concat(pieces);s=s.where(np.isfinite(s))
        key=s.index.floor('5min')+pd.Timedelta(minutes=5)
        grouped=s.groupby(key);power=grouped.mean().where(grouped.count()==5)
        return power.reindex(pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min',tz=timezone(timedelta(hours=-5))))
    r=pd.read_csv(paths['YULARA_RAW_FILE'],usecols=['timestamp','Active_Power'])
    idx=pd.DatetimeIndex(pd.to_datetime(r.timestamp));v=pd.to_numeric(r.Active_Power,errors='coerce').to_numpy(float)
    select=(idx>=pd.Timestamp('2017-01-01'))&(idx<pd.Timestamp('2018-01-01'))&(idx==idx.floor('5min'))
    s=pd.Series(v[select],index=idx[select]+pd.Timedelta(minutes=5));s=s.where(np.isfinite(s))
    return s.reindex(pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min'))

def calculate(y,p,mask,span):
    observed=np.asarray(y[mask],dtype='float64');estimated=np.asarray(p[mask],dtype='float64')
    residual=estimated-observed;n=len(observed)
    if not n:return dict.fromkeys(['RMSE','MAE','bias','R2','nRMSE'],math.nan)
    square_sum=np.dot(residual,residual);rmse=math.sqrt(square_sum/n)
    centered=observed-observed.mean();variation=np.dot(centered,centered)
    return {'RMSE':rmse,'MAE':np.abs(residual).sum()/n,'bias':residual.sum()/n,
            'R2':1-square_sum/variation if variation>0 else math.nan,'nRMSE':rmse/span if span>0 else math.nan}

def verify(paths_file=None):
    cfg=json.loads((HERE/'multisite_config.json').read_text(encoding='utf-8'))
    paths=json.loads(Path(paths_file or HERE.parents[1]/'.local/multisite_paths.json').read_text(encoding='utf-8'))
    expected={r['run_id'] for r in cfg['run_matrix']};actual={p.parent.name for p in ROOT.glob('*/completed.json')}
    if expected!=actual or len(actual)!=24:raise AssertionError('Exact 24 completed runs required')
    series={site:raw_power(site,cfg,paths) for site in ['YULARA_COMBINED','NIST_GROUND']}
    rows=[];checks=0;max_error=0.
    run_metadata={}
    def equal(value,wanted):
        nonlocal checks,max_error
        checks+=1
        if pd.isna(value) and pd.isna(wanted):return
        if not np.isclose(float(value),float(wanted),rtol=1e-9,atol=1e-9):raise AssertionError(f'Independent mismatch {value} != {wanted}')
        max_error=max(max_error,abs(float(value)-float(wanted)))
    for identity in cfg['run_matrix']:
        folder=ROOT/identity['run_id'];info=json.loads((folder/'completed.json').read_text(encoding='utf-8'))
        for k,v in identity.items():
            if info[k]!=v:raise AssertionError('Identity mismatch')
        run_metadata[(identity['site'],identity['model'],identity['seed'])]=info
        history=json.loads((folder/'training_history.json').read_text(encoding='utf-8'))
        best=math.inf;selected=0
        for record in history:
            if record['validation_global_mse']<best-cfg['future_training']['min_delta']:
                best=record['validation_global_mse'];selected=record['epoch']
        equal(best,info['best_validation_mse']);equal(selected,info['best_epoch'])
        s=series[identity['site']];train=s.loc['2017-01-01':'2017-08-31 23:59:59']
        span=float(train.max()-train.min());threshold=.01*float(train.max())
        with np.load(folder/'test_predictions.npz') as z:
            pred=z['predictions'];y=z['labels'];origins=pd.DatetimeIndex(pd.to_datetime(z['forecast_origin']))
            if not np.isfinite(pred).all() or pred.shape!=y.shape:raise AssertionError('Nonfinite prediction/shape')
            if identity['site']=='NIST_GROUND' and not all(str(t).endswith('-05:00') for t in origins):raise AssertionError('Lost EST')
            np.testing.assert_array_equal(z['target_start'],np.asarray([str(t+pd.Timedelta(minutes=5)) for t in origins]))
            positions=s.index.get_indexer(origins);targets=positions[:,None]+np.arange(1,145)
            validpos=targets<len(s);safe=np.minimum(targets,len(s)-1)
            independently_labeled=np.where(validpos,s.to_numpy()[safe],np.nan)
            np.testing.assert_array_equal(y,independently_labeled)
            np.testing.assert_array_equal(z['last_power'],s.reindex(origins).to_numpy())
            np.testing.assert_array_equal(z['target_valid'],np.isfinite(y))
            # Exact timestamp join on the fixed coordinate, then gather target points.
            lagged=s.reindex(s.index-pd.Timedelta(hours=24)).to_numpy()
            daily=np.where(validpos,lagged[safe],np.nan);np.testing.assert_array_equal(z['daily'],daily)
            equal(z['train_range'],span);equal(z['daylight_threshold'],threshold)
            last=np.broadcast_to(s.reindex(origins).to_numpy()[:,None],y.shape)
            for h in [12,48,96,144]:
                prefix=y[:,:h];eligible=np.isfinite(prefix).all(axis=1)&np.isfinite(last[:,0])
                for scope in ['full','daylight']:
                    base=np.tile(eligible[:,None],(1,h))
                    if scope=='daylight':base &= prefix>threshold
                    dm=base&np.isfinite(daily[:,:h])
                    np.testing.assert_array_equal(z[f'primary_H{h}_{scope}'],base);np.testing.assert_array_equal(z[f'daily_H{h}_{scope}'],dm)
                    for analysis,mask in [('primary',base),('supplementary_daily_matched',dm)]:
                        lv=calculate(prefix,last[:,:h],mask,span);dv=calculate(prefix,daily[:,:h],dm,span)
                        methods=[(identity['model'],pred[:,:h]),('LAST_VALUE_PERSISTENCE',last[:,:h])]
                        if analysis!='primary':methods.append(('DAILY_PERSISTENCE',daily[:,:h]))
                        for name,values in methods:
                            v=calculate(prefix,values,mask,span)
                            v['last_value_skill']=1-v['RMSE']/lv['RMSE'] if lv['RMSE']>0 else math.nan
                            v['daily_skill']=1-v['RMSE']/dv['RMSE'] if analysis!='primary' and dv['RMSE']>0 else math.nan
                            rows.append(dict(site=identity['site'],model=name,seed=identity['seed'],horizon=h,scope=scope,analysis=analysis,
                                             forecast_origin_count=int(mask.any(axis=1).sum()),valid_target_count=int(mask.sum()),**v))
        print('INDEPENDENT '+identity['run_id'],flush=True)
    independent=pd.DataFrame(rows).drop_duplicates(KEYS).set_index(KEYS).sort_index()
    published=pd.read_csv(HERE/'metrics_per_seed.csv').set_index(KEYS).sort_index()
    if not independent.index.equals(published.index):raise AssertionError('CSV row identities')
    for key,row in independent.iterrows():
        for metric in METRICS+['forecast_origin_count','valid_target_count']:equal(row[metric],published.loc[key,metric])
        if key[1] in cfg['models']:
            info=run_metadata[key[:3]]
            for metric in ['parameter_count','best_epoch','best_validation_mse','training_seconds']:equal(info[metric],published.loc[key,metric])
    summary=pd.read_csv(HERE/'metrics_summary_mean_sd.csv').set_index(['site','model','horizon','scope','analysis'])
    calculated=[]
    for key,group in independent.reset_index().groupby(['site','model','horizon','scope','analysis'],sort=False):
        if sorted(group.seed)!=[42,43,44]:raise AssertionError('Three exact seeds')
        record=dict(zip(['site','model','horizon','scope','analysis'],key))
        for metric in METRICS:
            values=group[metric].to_numpy();mean=np.mean(values);sd=np.std(values,ddof=1)
            equal(mean,summary.loc[key,metric+'_mean']);equal(sd,summary.loc[key,metric+'_sample_sd'])
            record[metric+'_mean']=mean;record[metric+'_sample_sd']=sd
        equal(3,summary.loc[key,'seed_count'])
        for metric in ['forecast_origin_count','valid_target_count']:equal(group[metric].iloc[0],summary.loc[key,metric])
        calculated.append(record)
    table=pd.DataFrame(calculated);primary=table[table.model.eq(cfg['primary_model'])]
    wins={}
    for site,group in primary.groupby('site'):
        p=group[group.analysis.eq('primary')];d=group[group.analysis.eq('supplementary_daily_matched')]
        wins[site]={'primary_vs_last_wins':int((p.last_value_skill_mean>0).sum()),'daily_matched_vs_daily_wins':int((d.daily_skill_mean>0).sum()),
                    'daily_wins_over_primary':int((d.daily_skill_mean<0).sum()),'comparisons_per_baseline':8}
    ranks=[];envelopes=[]
    neural=table[table.model.isin(cfg['models'])]
    for key,group in neural.groupby(['site','horizon','scope','analysis']):
        ranking=group.sort_values(['RMSE_mean','model']);rankvalues=group.RMSE_mean.rank(method='average')
        for ix,row in group.iterrows():ranks.append({**dict(zip(['site','horizon','scope','analysis'],key)),'model':row.model,'rank':float(rankvalues.loc[ix]),'RMSE_mean':row.RMSE_mean})
        best=ranking.iloc[0];envelopes.append({**dict(zip(['site','horizon','scope','analysis'],key)),'label':'post hoc descriptive envelope','model':best.model,'RMSE_mean':best.RMSE_mean})
    result={'passed':checks,'failed':0,'skipped':0,'max_absolute_difference':max_error,'runs':24,'training_executed':False,
            'independent_metric_path':True,'primary_model':cfg['primary_model'],'primary_comparisons':wins,
            'primary_results':primary.to_dict('records'),'neural_ranks':ranks,'envelope':envelopes,
            'per_seed_rows':len(independent),'summary_rows':len(summary)}
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    (HERE/'M2_INDEPENDENT_AUDIT.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False),encoding='utf-8')
    return result

if __name__=='__main__':
    result=verify();print(json.dumps({k:result[k] for k in ['passed','failed','skipped','max_absolute_difference','runs','primary_comparisons']}))

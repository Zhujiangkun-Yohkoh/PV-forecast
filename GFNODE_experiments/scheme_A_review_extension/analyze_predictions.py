"""Post hoc, prediction-only analysis. Does not import any training module."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
MODELS={'Discrete recurrent decoder':'Recurrent','Inverted-variate Transformer':'Inverted','Joint-patch Transformer':'Joint-patch','Depthwise convolutional TCN':'TCN','DISCRETE_RECURRENT_TRAJECTORY':'Recurrent','INVERTED_VARIATE_TRAJECTORY':'Inverted','JOINT_PATCH_TRAJECTORY':'Joint-patch','DEPTHWISE_TCN_TRAJECTORY':'TCN'}
def stats(y,p,m):
    e=np.where(m,p-y,0.);n=int(m.sum());sse=float(np.square(e).sum());sae=float(np.abs(e).sum());signed=float(e.sum())
    return dict(points=n,origins=int(m.any(1).sum()),SSE=sse,SAE=sae,signed_error=signed,RMSE=np.sqrt(sse/n) if n else np.nan,MAE=sae/n if n else np.nan,bias=signed/n if n else np.nan)
def run(paths):
    cfg=json.loads(Path(paths).read_text(encoding='utf8'));out=HERE/'results';out.mkdir(exist_ok=True)
    rows=[];lead=[];monthly=[];curves=[];training=[];boot_store={};protected={};checks=0
    for external,key,pattern in [(False,'alice_results','test_H144.npz'),(True,'external_results','test_predictions.npz')]:
      root=Path(cfg[key]);files=sorted(root.rglob(pattern));assert len(files)==(24 if external else 36)
      for file in files:
        for f in file.parent.iterdir():
          if f.is_file():protected[str(f)]=(f.stat().st_size,f.stat().st_mtime_ns)
        info=json.loads((file.parent/'completed.json').read_text());site=info['site'] if external else info['dataset'];model=MODELS[info['model']];seed=int(info['seed'])
        training.append(dict(site=site,model=model,seed=seed,best_epoch=info['best_epoch'],stop_epoch=info['actual_epochs'],at_budget=info['actual_epochs']==25,best_at_budget=info['best_epoch']==25))
        with np.load(file) as z: b={k:z[k] for k in z.files}
        y=b['labels'].astype(float);pred=b['predictions'].astype(float);v=b['target_valid'];origin=pd.DatetimeIndex(pd.to_datetime(b['forecast_origin']));assert np.isfinite(pred).all()
        if external:daily=b['daily'];threshold=float(b['daylight_threshold'])
        else:
          raw=pd.read_csv(cfg['alice_raw'][site]);raw['timestamp']=pd.to_datetime(raw.timestamp);raw=raw.sort_values('timestamp').drop_duplicates('timestamp',keep='last').set_index('timestamp');power=pd.to_numeric(raw.Active_Power,errors='coerce');power=power.where(power>=0)
          threshold=.01*power.loc['2018-04-01':'2018-07-15 23:55'].max()
          target=origin.to_numpy()[:,None]+np.arange(1,145)*np.timedelta64(5,'m')
          daily=power.reindex(pd.DatetimeIndex((target-np.timedelta64(24,'h')).ravel())).to_numpy().reshape(y.shape)
        full=v.all(1)&np.isfinite(b['last_power']);last=np.broadcast_to(b['last_power'][:,None],y.shape)
        active=y>threshold
        for support in ['horizon_specific','common_H144']:
          for h in [12,48,96,144]:
            eligible=(v[:,:h].all(1)&np.isfinite(b['last_power'])) if support=='horizon_specific' else full
            base=eligible[:,None]&v[:,:h]
            for analysis in ['primary','daily_matched']:
              mask=base.copy()
              if analysis=='daily_matched':mask &=np.isfinite(daily[:,:h])
              for method,p in [(model,pred),('Last-value',last),('Daily',daily)]:
                if method=='Daily' and analysis=='primary':continue
                parts={}
                for scope,m in [('full',mask),('power-active',mask&active[:,:h]),('low-power',mask&~active[:,:h])]:
                  s=stats(y[:,:h],p[:,:h],m);parts[scope]=s
                  rows.append(dict(site=site,model=method,seed=seed,neural_source=model,h=h,hours=h/12,support=support,analysis=analysis,scope=scope,**s,weighted_MSE_contribution=s['SSE']/mask.sum()))
                assert np.isclose(parts['full']['SSE'],parts['power-active']['SSE']+parts['low-power']['SSE'],rtol=1e-11,atol=1e-7);checks+=1
              if support=='horizon_specific' and model=='Inverted':
                for scope,sm in [('full',mask),('power-active',mask&active[:,:h]),('low-power',mask&~active[:,:h])]:
                  ref=last if analysis=='primary' else daily
                  for hours in [24,48,72]:
                    delta=(origin-origin[0].normalize()).total_seconds()/3600;ids=np.asarray(delta//hours,int);nb=ids.max()+1
                    n=np.bincount(ids,weights=sm.sum(1),minlength=nb)
                    se=np.bincount(ids,weights=np.where(sm,(pred[:,:h]-y[:,:h])**2,0).sum(1),minlength=nb)
                    sr=np.bincount(ids,weights=np.where(sm,(ref[:,:h]-y[:,:h])**2,0).sum(1),minlength=nb)
                    boot_store.setdefault((site,h,analysis,scope,hours),[]).append((seed,n,se,sr))
        for method,p in [(model,pred),('Daily',daily),('Last-value',last)]:
          m=full[:,None]&v&np.isfinite(daily)
          for j in range(144):
            for scope,sm in [('full',m[:,j:j+1]),('power-active',m[:,j:j+1]&active[:,j:j+1])]:
              lead.append(dict(site=site,model=method,seed=seed,neural_source=model,lead=j+1,lead_hours=(j+1)/12,scope=scope,**stats(y[:,j:j+1],p[:,j:j+1],sm)))
          for month in sorted(set(origin.strftime('%Y-%m'))):
            mm=m&(origin.strftime('%Y-%m')==month)[:,None]
            monthly.append(dict(site=site,model=method,seed=seed,neural_source=model,origin_month=month,**stats(y,p,mm)))
        if external and model=='Inverted' and seed==42:
          ids=np.flatnonzero(full);cases=[]
          for month in [10,11,12]:
            selected=ids[(origin[ids].month==month)&((origin[ids].day>15)|((origin[ids].day==15)&(origin[ids].hour>=10)))]
            if len(selected):cases.append((f'fixed_month_{month}',selected[0]))
          m=full[:,None]&v&np.isfinite(daily);effect=np.where(m,(daily-y)**2-(pred-y)**2,0).sum(1)/np.maximum(m.sum(1),1)
          candidates=ids[m[ids].sum(1)==144]
          if len(candidates):cases += [('posthoc_best',candidates[np.argmax(effect[candidates])]),('posthoc_worst',candidates[np.argmin(effect[candidates])])]
          for label,i in cases:
            for j in range(144):curves.append(dict(site=site,case=label,origin=str(origin[i]),lead=j+1,hours=(j+1)/12,truth=y[i,j],Inverted=pred[i,j],Daily=daily[i,j],last=last[i,j],seed=42))
        print(site,model,seed,flush=True)
    frame=pd.DataFrame(rows)
    # Baselines are deterministic: remove repeated neural-source/seed copies explicitly.
    neural=frame[~frame.model.isin(['Daily','Last-value'])];baselines=frame[frame.model.isin(['Daily','Last-value'])].drop_duplicates(['site','model','h','support','analysis','scope']).copy();baselines['seed']=0;baselines['neural_source']='deterministic'
    frame=pd.concat([neural,baselines]);frame.to_csv(out/'metrics_decomposition.csv',index=False)
    for name,values in [('lead_specific',lead),('monthly',monthly)]:
      df=pd.DataFrame(values);keys=[c for c in df if c not in ['seed','neural_source','points','origins','SSE','SAE','signed_error','RMSE','MAE','bias']];base=df[df.model.isin(['Daily','Last-value'])].drop_duplicates(keys).copy();base['seed']=0;base['neural_source']='deterministic';pd.concat([df[~df.model.isin(['Daily','Last-value'])],base]).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame(curves).to_csv(out/'trajectory_cases.csv',index=False);pd.DataFrame(training).to_csv(out/'training_budget.csv',index=False)
    boots=[]
    for key,items in boot_store.items():
      assert sorted(x[0] for x in items)==[42,43,44];n=np.stack([x[1] for x in items]);se=np.stack([x[2] for x in items]);sr=np.stack([x[3] for x in items]);nb=n.shape[1]
      rng=np.random.default_rng(20260908);indices=rng.integers(0,nb,size=(2000,nb));nn=n[:,indices].sum(2);ss=se[:,indices].sum(2);rr=sr[:,indices].sum(2)
      valid=(nn>0).all(0)&(rr>0).all(0);skills=(1-np.sqrt(ss[:,valid]/nn[:,valid])/np.sqrt(rr[:,valid]/nn[:,valid])).mean(0)
      point=float((1-np.sqrt(se.sum(1)/n.sum(1))/np.sqrt(sr.sum(1)/n.sum(1))).mean());lo,hi=np.quantile(skills,[.025,.975])
      boots.append(dict(zip(['site','h','analysis','scope','block_hours'],key))|dict(blocks=nb,replicates=2000,valid_replicates=int(valid.sum()),skill=point,ci_low=lo,ci_high=hi))
    pd.DataFrame(boots).to_csv(out/'paired_block_intervals.csv',index=False)
    for f,s in protected.items():assert (Path(f).stat().st_size,Path(f).stat().st_mtime_ns)==s
    (out/'VERIFICATION.json').write_text(json.dumps(dict(neural_runs=60,additive_SSE_checks=checks,protected_files=len(protected),source_stats_unchanged=True,training_executed=False,new_analysis='post hoc',tests_not_rerun='Historical M1/M2/M3 suites not rerun in this script'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);run(p.parse_args().paths)

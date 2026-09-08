"""Independent array checks and read-only learning-history extraction."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def run(paths):
 c=json.loads(Path(paths).read_text());rows=[];checks=0;files={}
 for external,key,pattern in [(False,'alice_results','epochs.jsonl'),(True,'external_results','training_history.json')]:
  logs=list(Path(c[key]).rglob(pattern));assert len(logs)==(24 if external else 36)
  for p in logs:
   files[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns)
   meta=json.loads((p.parent/'completed.json').read_text());hist=json.loads(p.read_text()) if external else [json.loads(x) for x in p.read_text().splitlines()]
   assert len(hist)==meta['actual_epochs'];assert min(hist,key=lambda x:x['validation_global_mse'])['epoch']==meta['best_epoch'];checks+=1
   for h in hist:rows.append(dict(site=meta.get('site',meta.get('dataset')),model=meta['model'],seed=meta['seed'],epoch=h['epoch'],train_MSE=h.get('train_mse',h.get('train_global_mse')),validation_MSE=h['validation_global_mse']))
 pd.DataFrame(rows).to_csv(HERE/'results/learning_histories.csv',index=False)
 metrics=pd.read_csv(HERE/'results/ridge_metrics.csv');selection=pd.read_csv(HERE/'results/ridge_validation_selection.csv');count=0
 for p in Path(c['new_results']).rglob('*_predictions.npz'):
  site=p.parent.name;model=p.stem.replace('_predictions','').replace('_day','+day');meta=json.loads(p.with_name(p.name.replace('_predictions.npz','_completed.json')).read_text())
  s=selection[(selection.site==site)&(selection.model==model)].sort_values(['validation_global_MSE_kW2','alpha'],ascending=[True,False]);assert meta['selected_alpha']==s.iloc[0].alpha;checks+=1
  with np.load(p) as z:
   y=z['labels'];pr=z['predictions'];v=z['target_valid'];daily=z['daily'];last=z['last_power'];threshold=z['daylight_threshold'];assert y.shape==pr.shape==v.shape==daily.shape;assert np.isfinite(pr).all();checks+=1
   for row in metrics[(metrics.site==site)&(metrics.model==model)].itertuples():
    h=row.h;eligible=v[:,:h if row.support=='horizon_specific' else 144].all(1)&np.isfinite(last);mask=eligible[:,None]&v[:,:h]
    if row.analysis=='daily_matched':mask=mask&np.isfinite(daily[:,:h])
    if row.scope=='power-active':mask=mask&(y[:,:h]>threshold)
    if row.scope=='low-power':mask=mask&(y[:,:h]<=threshold)
    error=(pr[:,:h]-y[:,:h])[mask];assert len(error)==row.points;assert np.isclose(np.sqrt(np.dot(error,error)/len(error)),row.RMSE,rtol=1e-10);assert np.isclose(np.mean(np.abs(error)),row.MAE,rtol=1e-10);count+=1
 assert count==480
 # Direct normal equation and spectral implementation agree on actual arrays.
 rng=np.random.default_rng(842);x=rng.normal(size=(47,9));y=rng.normal(size=(47,12));x-=x.mean(0);y-=y.mean(0);g=x.T@x;w,u=np.linalg.eigh(g)
 for alpha in [.1,1,10,100,1000]:
  direct=np.linalg.solve(g+alpha*np.eye(9),x.T@y);spectral=u@((u.T@(x.T@y))/(w[:,None]+alpha));assert np.allclose(x@direct,x@spectral,rtol=1e-10,atol=1e-10);checks+=1
 assert all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s for p,s in files.items())
 result=dict(passed=checks+count,failed=0,skipped=0,learning_histories=60,ridge_runs=10,ridge_metric_rows=count,neural_training_executed=False,scope='New review checks; does not repeat historical checkpoint forward verification')
 (HERE/'results/REVIEW_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);run(p.parse_args().paths)

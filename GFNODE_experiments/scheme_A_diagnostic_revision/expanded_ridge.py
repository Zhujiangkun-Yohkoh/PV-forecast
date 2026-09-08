"""Uniform alpha sensitivity after diagnosis; all selections by Validation only."""
from pathlib import Path
import argparse,json,pickle
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve,cho_factor,lstsq
from threadpoolctl import threadpool_limits
from designs import reconstruct,variant_design,HERE,SITES
def run(paths,destination,sites,matrix_cache=None):
 out=HERE/'results';dest=Path(destination);dest.mkdir(exist_ok=True);scores=[];results=[];patterns=[];diagnostic_cases=[]
 if sites!=SITES:
  scores=pd.read_csv(out/'expanded_alpha_validation.csv').query('site not in @sites').to_dict('records');results=pd.read_csv(out/'expanded_ridge_metrics.csv').query('site not in @sites').to_dict('records')
 for site in sites:
  cache=Path(matrix_cache)/(site+'.pkl') if matrix_cache else None
  if cache and cache.exists():
   with cache.open('rb') as f:d=pickle.load(f)
   assert d['folder'].name==site
  else:
   d=reconstruct(paths,site)
   if cache:
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('wb') as f:pickle.dump(d,f)
  folder=dest/site;folder.mkdir(exist_ok=True)
  if site=='YULARA_COMBINED':
   for split,x in d['design'].items():
    missing=x.reshape(-1,72,7)[:,:,4];pcount=missing.sum(1);changes=(np.diff(missing,axis=1)!=0).sum(1)
    for label,mask in [('none',pcount==0),('some',((pcount>0)&(pcount<72))),('all72',pcount==72)]:patterns.append(dict(site=site,split=split,pattern=label,origins=int(mask.sum()),total_origins=len(x),mean_transitions=float(changes[mask].mean()) if mask.any() else np.nan))
  for variant in ['Ridge','Ridge+day']:
   _,X,c,names=variant_design(d,variant);Y=d['labels']['train']*c['target_scale']+c['target_min']-c['intercept'];g=X['train'].T@X['train'];rhs=X['train'].T@Y;e,u=np.linalg.eigh(g);e=np.maximum(e,0);project=u.T@rhs;best=None
   for alpha in [10.**k for k in range(-4,9)]:
    coef=u@(project/(e[:,None]+alpha));pv=(X['validation']@coef+c['intercept']-c['target_min'])/c['target_scale'];mse=float(np.mean((pv-d['labels']['validation'])**2));scores.append(dict(site=site,model=variant,alpha=alpha,validation_MSE_kW2=mse,train_origins=len(Y),validation_origins=len(pv)))
    if best is None or mse<=best[0]:best=(mse,alpha,coef.copy())
   stable=cho_solve(cho_factor(g+best[1]*np.eye(len(g))),rhs);coef=best[2];p=(X['test']@coef+c['intercept']-c['target_min'])/c['target_scale'];pstable=(X['test']@stable+c['intercept']-c['target_min'])/c['target_scale'];maxdiff=float(np.max(np.abs(p-pstable)));print(site,variant,'selected',best[:2],'spectral vs Cholesky maxdiff',maxdiff,flush=True)
   solver_note='spectral_Cholesky_agreement'
   if not np.allclose(p,pstable,atol=1e-5,rtol=1e-7):
    augx=np.vstack([X['train'],np.sqrt(best[1])*np.eye(len(g))]);augy=np.vstack([Y,np.zeros((len(g),Y.shape[1]))]);qr,_,rank,_=lstsq(augx,augy,lapack_driver='gelsy');qrp=(X['test']@qr+c['intercept']-c['target_min'])/c['target_scale'];qrv=(X['validation']@qr+c['intercept']-c['target_min'])/c['target_scale'];record_solver={'site':site,'model':variant,'alpha':best[1],'columns':len(g),'QR_augmented_rank':int(rank),'spectral_QR_maxdiff':float(np.max(np.abs(p-qrp))),'Cholesky_QR_maxdiff':float(np.max(np.abs(pstable-qrp))),'spectral_Cholesky_maxdiff':maxdiff,'spectral_validation_MSE':best[0],'QR_validation_MSE':float(np.mean((qrv-d['labels']['validation'])**2)),'spectral_relative_normal_residual':float(np.linalg.norm((g+best[1]*np.eye(len(g)))@coef-rhs)/np.linalg.norm(rhs)),'QR_relative_normal_residual':float(np.linalg.norm((g+best[1]*np.eye(len(g)))@qr-rhs)/np.linalg.norm(rhs))};(out/(site+'_'+variant.replace('+','_')+'_QR_CHECK.json')).write_text(json.dumps(record_solver,indent=2), encoding='utf8');print(record_solver,flush=True);solver_note='QR_checked_small_alpha_sensitivity'
    del augx,augy,qrp,qrv,qr
   old=(X['test']@c['coef']+c['intercept']-c['target_min'])/c['target_scale'];a=d['archive'];y=a['labels'];mask=a['target_valid'].all(1)[:,None]&a['target_valid']&np.isfinite(a['last_power'])[:,None]&np.isfinite(a['daily'])
   record=dict(site=site,model=variant,selected_alpha=best[1],validation_MSE_kW2=best[0],real_matrix_solver_maxdiff_kW=float(np.max(np.abs(p-pstable))),endpoint=best[1] in [1e-4,1e8],post_hoc_sensitivity=True,solver_note=solver_note)
   for scope,mm in [('full',mask),('power-active',mask&(y>a['daylight_threshold'])),('low-power',mask&(y<=a['daylight_threshold']))]:results.append(dict(**record,scope=scope,points=int(mm.sum()),origins=int(mm.any(1).sum()),old_RMSE=np.sqrt(np.mean((old[mm]-y[mm])**2)),new_RMSE=np.sqrt(np.mean((p[mm]-y[mm])**2))))
   stem=variant.replace('+','_');np.savez_compressed(folder/(stem+'_predictions.npz'),predictions=p,**{k:v for k,v in a.items() if k!='predictions'});np.savez_compressed(folder/(stem+'_coefficients.npz'),**(c|{'coef':coef}));(folder/(stem+'_completed.json')).write_text(json.dumps(record,indent=2), encoding='utf8')
   if site=='YULARA_COMBINED':
    for split in ['validation','test']:
     yy=d['labels'][split];orig=d['origins'][split];pp=(X[split]@c['coef']+c['intercept']-c['target_min'])/c['target_scale'];se=np.where(np.isfinite(yy),(pp-yy)**2,0).sum(1);i=int(se.argmax());weather=d['design'][split].reshape(-1,72,7)[i,:,4]
     for j in range(144):diagnostic_cases.append(dict(site=site,model=variant,split=split,selection='maximum_original_SSE_diagnostic_not_representative',origin=str(orig[i]),lead=j+1,truth=yy[i,j],original_prediction=pp[i,j],expanded_prediction=((X[split][i]@coef+c['intercept']-c['target_min'])/c['target_scale'])[j],history_weather_missing_count=int(weather.sum())))
   print(site,variant,'expanded selection',best[:2],flush=True)
  pd.DataFrame(scores).to_csv(out/'expanded_alpha_validation.csv',index=False);pd.DataFrame(results).to_csv(out/'expanded_ridge_metrics.csv',index=False)
  if patterns:pd.DataFrame(patterns).to_csv(out/'yulara_missing_patterns.csv',index=False)
  if diagnostic_cases:pd.DataFrame(diagnostic_cases).to_csv(out/'yulara_diagnostic_cases.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--destination',required=True);p.add_argument('--sites',nargs='+',default=SITES);p.add_argument('--matrix-cache');a=p.parse_args()
 with threadpool_limits(limits=4):run(a.paths,a.destination,a.sites,a.matrix_cache)

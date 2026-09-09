"""Actual-matrix verification and distribution/contribution diagnostics for existing fits."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from designs import reconstruct,variant_design,load_npz,SITES,HERE
def run(paths,sites):
 out=HERE/'results';out.mkdir(exist_ok=True);distributions=[];concentration=[];features=[];solvers=[];align=[];worst=[];groups=[]
 for site in sites:
  d=reconstruct(paths,site);align+=d['alignment'];print(site,'matrices reconstructed',flush=True)
  for variant in ['Ridge','Ridge+day']:
   raw,X,c,names=variant_design(d,variant);Y=(d['labels']['train']*c['target_scale']+c['target_min'])-c['intercept'];xt=X['train'];alpha=json.loads((d['folder']/(variant.replace('+','_')+'_completed.json')).read_text(encoding='utf8'))['selected_alpha']
   gram=xt.T@xt;rhs=xt.T@Y;system=gram+alpha*np.eye(xt.shape[1]);stable=cho_solve(cho_factor(system),rhs);saved=c['coef'];eig=np.linalg.eigvalsh(gram)
   residual=lambda z:np.linalg.norm(system@z-rhs)/np.linalg.norm(rhs)
   obj=lambda z:float(np.square(xt@z-Y).sum()+alpha*np.square(z).sum())
   prediction=lambda x,z:(x@z+c['intercept']-c['target_min'])/c['target_scale']
   ps={s:prediction(x,saved) for s,x in X.items()};pb={s:prediction(x,stable) for s,x in X.items()}
   old=load_npz(d['folder']/(variant.replace('+','_')+'_predictions.npz'));assert np.array_equal(old['forecast_origin'],d['archive']['forecast_origin']);assert np.array_equal(old['target_valid'],d['archive']['target_valid']);assert np.array_equal(old['target_start'],d['archive']['target_start']);assert np.allclose(old['labels'],d['archive']['labels'],equal_nan=True)
   assert np.allclose(old['predictions'],ps['test'],rtol=1e-9,atol=1e-7);assert np.allclose(old['daily'],d['daily']['test'],equal_nan=True)
   assert np.allclose(xt.mean(0),0,atol=1e-10);assert np.allclose(raw['train'].mean(0),c['feature_mean'],atol=1e-10)
   for s in X:
    solvers.append(dict(site=site,model=variant,split=s,alpha=alpha,rows=len(xt),columns=xt.shape[1],saved_objective=obj(saved) if s=='train' else np.nan,stable_objective=obj(stable) if s=='train' else np.nan,saved_relative_normal_residual=residual(saved),stable_relative_normal_residual=residual(stable),max_prediction_difference_kW=np.max(np.abs(ps[s]-pb[s])),RMSE_prediction_difference_kW=np.sqrt(np.mean((ps[s]-pb[s])**2)),gram_eigen_min=eig[0],gram_eigen_max=eig[-1],regularized_condition=(eig[-1]+alpha)/(max(eig[0],0)+alpha),coefficient_Frobenius=np.linalg.norm(saved),intercept_scaled_min=c['intercept'].min(),intercept_scaled_max=c['intercept'].max()))
   train_min=float(d['labels']['train'].min());train_max=float(d['labels']['train'].max())
   std=raw['train'].std(0);feature_scale=c['feature_scale']
   for s in X:
    y=d['labels'][s];p=ps[s];valid=np.isfinite(y)
    if s=='test':valid &= d['archive']['target_valid']
    for what,arr in [('truth',y[valid]),('prediction',p[valid])]:
     q=np.quantile(arr,[0,.001,.01,.05,.5,.95,.99,.999,1]);distributions.append(dict(site=site,model=variant,split=s,kind=what,n=len(arr),**dict(zip(['min','p001','p01','p05','median','p95','p99','p999','max'],q)),negative_fraction=np.mean(arr<0),outside_train_fraction=np.mean((arr<train_min)|(arr>train_max)),train_min=train_min,train_max=train_max))
    se=np.where(valid,(p-y)**2,0).sum(1);order=np.argsort(-se);total=se.sum()
    concentration.append(dict(site=site,model=variant,split=s,origins=len(se),points=valid.sum(),RMSE=np.sqrt(total/valid.sum()),SSE=total,top1pct_SSE_share=se[order[:max(1,int(np.ceil(.01*len(se))))]].sum()/total,top5pct_SSE_share=se[order[:max(1,int(np.ceil(.05*len(se))))]].sum()/total))
    for rank,i in enumerate(order[:20]):worst.append(dict(site=site,model=variant,split=s,rank=rank+1,origin=str(d['origins'][s][i]),SSE=se[i],SSE_share=se[i]/total,pred_min=p[i].min(),pred_max=p[i].max(),truth_max=np.nanmax(y[i])))
    for j,name in enumerate(names):
     features.append(dict(site=site,model=variant,split=s,column=j,feature=name,train_std=std[j],scale=feature_scale[j],train_constant=std[j]==0,near_constant=(0<std[j]<1e-6),train_nonzero_fraction=np.mean(raw['train'][:,j]!=0),raw_mean=raw[s][:,j].mean(),raw_min=raw[s][:,j].min(),raw_max=raw[s][:,j].max(),standardized_mean=X[s][:,j].mean(),standardized_absmax=np.abs(X[s][:,j]).max(),coef_norm=np.linalg.norm(saved[j])/float(c['target_scale'][0])))
    # Additive prediction components by input channel, computed on all rows.
    channel_groups={}
    for j,name in enumerate(names):channel_groups.setdefault(name.split(':')[-1] if ':' in name else ('day_power' if name.startswith('day_power') else 'day_missing'),[]).append(j)
    sum_components=np.zeros_like(p)
    for group,ids in channel_groups.items():
     component=X[s][:,ids]@saved[ids]/c['target_scale'];sum_components+=component
     groups.append(dict(site=site,model=variant,split=s,group=group,component_RMS_kW=np.sqrt(np.mean(component**2)),component_mean_kW=component.mean(),component_absmax_kW=np.abs(component).max(),worst_origin_component_RMS_kW=np.sqrt(np.mean(component[order[0]]**2))))
    assert np.allclose(sum_components+(c['intercept']-c['target_min'])/c['target_scale'],p,atol=1e-6)
   print(site,variant,'verified alpha',alpha,'max Test diff',solvers[-1]['max_prediction_difference_kW'],flush=True)
  for filename,rows in [('ridge_distributions',distributions),('ridge_concentration',concentration),('ridge_features',features),('ridge_solver_comparison',solvers),('ridge_alignment',align),('ridge_worst_origins',worst),('ridge_feature_contributions',groups)]:pd.DataFrame(rows).to_csv(out/(filename+'.csv'),index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--sites',nargs='+',default=['YULARA_COMBINED']);a=p.parse_args()
 with threadpool_limits(limits=4):run(a.paths,a.sites)

from __future__ import annotations
import csv, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent; EXP=HERE.parent; INFO_PATH=EXP/'asoc_multirate_information_screen'/'run_information_screen.py'; DATA_PATH=EXP/'asoc_multirate_information_screen'/'results'/'prepared_data.npz'; ART=EXP/'asoc_multirate_information_screen'/'results'/'MEAN_ONLY'; CONFIG_PATH=EXP/'asoc_multirate_information_screen'/'config.json'; OUT=HERE/'TRAJECTORY_ERROR_METRICS.csv'

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
INFO=module('opp_info',INFO_PATH)

def metric(y,p):
 e=p-y;mse=float(np.mean(e*e));den=float(np.sum((y-y.mean())**2));return {'rmse_kw':math.sqrt(mse),'mae_kw':float(np.mean(abs(e))),'bias_kw':float(np.mean(e)),'r2':1-float(np.sum(e*e))/den if den else math.nan}

def predictions(split,seed,d,c,device):
 origins=d[f'{split}_origins']; ds=INFO.WindowDataset(d['scaled_features'],d['power'],origins,d['base_cols'],c['lookback'],c['horizon'],float(d['target_center']),float(d['target_scale']));ld=DataLoader(ds,batch_size=512,shuffle=False,num_workers=0);m=INFO.ModernTCN(len(d['base_cols']),c).to(device);ck=torch.load(ART/str(seed)/'best_validation.pt',map_location=device,weights_only=True);m.load_state_dict(ck['state_dict']);m.eval();out=[]
 with torch.no_grad():
  for x,_ in ld:out.append(m(x.to(device)).cpu().numpy())
 p=np.concatenate(out)*float(d['target_scale'])+float(d['target_center']);y=np.stack([d['power'][o+1:o+13] for o in origins]);return origins,y.astype(np.float32),p.astype(np.float32),ck

def trajectory_quantities(y,p,origin_power):
 dy=np.diff(np.column_stack([origin_power,y]),axis=1);dp=np.diff(np.column_stack([origin_power,p]),axis=1);e=p-y
 return {'window_rmse':np.sqrt(np.mean(e*e,1)),'window_mae':np.mean(abs(e),1),'bias':e.mean(1),'peak_error':p.max(1)-y.max(1),'max_abs_error':abs(e).max(1),'energy_sum_error':e.sum(1),'true_tv':abs(dy).sum(1),'pred_tv':abs(dp).sum(1),'tv_ratio':abs(dp).sum(1)/np.maximum(abs(dy).sum(1),1e-8),'true_peak_rate':abs(dy).max(1),'pred_peak_rate':abs(dp).max(1),'dy':dy,'dp':dp}

def lag_diagnostic(y,p,maxlag=3):
 raw=np.sqrt(np.mean((p-y)**2,1));scores=[]
 for lag in range(-maxlag,maxlag+1):
  if lag<0: yy=y[:,:lag];pp=p[:,-lag:]
  elif lag>0: yy=y[:,lag:];pp=p[:,:-lag]
  else: yy=y;pp=p
  scores.append(np.sqrt(np.mean((pp-yy)**2,1)))
 a=np.stack(scores,1);ix=a.argmin(1);return ix-maxlag,a[np.arange(len(y)),ix],raw

def rows_for_split(rows,split,seed,origins,y,p,d,thresholds):
 power=d['power'];times=pd.to_datetime(d['times'][origins]);origin_power=power[origins];q=trajectory_quantities(y,p,origin_power);dy,dp=q['dy'],q['dp'];total_sse=float(np.sum((p-y)**2));
 for h in range(12):
  m=metric(y[:,h],p[:,h]); true_std=float(np.std(y[:,h])); rows.append({'section':'lead_time','split':split,'seed':seed,'scope':'full_timeline','horizon':h+1,'n':len(y),**m,'diff_mae_kw':float(np.mean(abs(dp[:,h]-dy[:,h]))),'prediction_to_truth_std_ratio':float(np.std(p[:,h])/true_std) if true_std else math.nan,'prediction_to_truth_change_amplitude_ratio':float(np.mean(abs(dp[:,h]))/max(np.mean(abs(dy[:,h])),1e-12)),'change_direction_accuracy':float(np.mean(np.sign(dp[:,h])==np.sign(dy[:,h]))),'sse_share':float(np.sum((p[:,h]-y[:,h])**2)/total_sse)})
 lag,aligned,raw=lag_diagnostic(y,p);rows.append({'section':'failure_type','split':split,'seed':seed,'scope':'all_windows','horizon':12,'n':len(y),'mean_window_bias_kw':float(q['bias'].mean()),'mean_peak_error_kw':float(q['peak_error'].mean()),'mean_max_abs_error_kw':float(q['max_abs_error'].mean()),'mean_energy_sum_error_kw_steps':float(q['energy_sum_error'].mean()),'mean_true_tv_kw':float(q['true_tv'].mean()),'mean_pred_tv_kw':float(q['pred_tv'].mean()),'aggregate_tv_ratio':float(q['pred_tv'].sum()/q['true_tv'].sum()),'mean_peak_rate_ratio':float(q['pred_peak_rate'].mean()/q['true_peak_rate'].mean()),'dominant_best_lag_steps':int(pd.Series(lag).mode().iloc[0]),'mean_best_lag_steps':float(lag.mean()),'raw_window_rmse_kw':float(raw.mean()),'lag_aligned_window_rmse_kw':float(aligned.mean()),'lag_alignment_gain_kw':float((raw-aligned).mean())})
 # Scenario masks. Definitions are fixed by Train thresholds passed in.
 hour=times.hour.to_numpy();month=times.month.to_numpy();mean_std=np.nanmean(d['raw_features'][origins][:,[8,16,24]],1);valid=np.nanmin(d['raw_features'][origins][:,[6,14,22]],1);future_change=np.max(abs(dy),1);cur=origin_power
 scenarios={'full_timeline':np.ones(len(y),bool),'daylight':(y>.063).any(1),'sunrise':(hour>=5)&(hour<9),'sunset':(hour>=16)&(hour<20),'stable_low_change':future_change<=thresholds['change_low'],'medium_change':(future_change>thresholds['change_low'])&(future_change<thresholds['change_high']),'high_change':future_change>=thresholds['change_high'],'past_irradiance_volatility_low':mean_std<=thresholds['irr_std_median'],'past_irradiance_volatility_high':mean_std>thresholds['irr_std_median'],'input_hf_partly_missing':valid<.999999,'input_hf_complete':valid>=.999999}
 for mo in sorted(np.unique(month)):scenarios[f'month_{mo:02d}']=month==mo
 for name,mask in [('solar_night',(hour<5)|(hour>=20)),('solar_morning',(hour>=5)&(hour<10)),('solar_midday',(hour>=10)&(hour<15)),('solar_evening',(hour>=15)&(hour<20)),('power_low',cur<=thresholds['power_q33']),('power_mid',(cur>thresholds['power_q33'])&(cur<thresholds['power_q67'])),('power_high',cur>=thresholds['power_q67'])]:scenarios[name]=mask
 for name,mask in scenarios.items():
  if not mask.any():continue
  mm=metric(y[mask],p[mask]);rows.append({'section':'scenario','split':split,'seed':seed,'scope':name,'horizon':12,'n':int(mask.sum()),**mm,'diff_mae_kw':float(np.mean(abs(dp[mask]-dy[mask]))),'tv_ratio':float(q['pred_tv'][mask].sum()/max(q['true_tv'][mask].sum(),1e-12)),'sse_share':float(np.sum((p[mask]-y[mask])**2)/total_sse)})
 # Direction error taxonomy; tolerance is Train-only.
 t=thresholds['direction_tol'];truth_up=dy>t;truth_down=dy<-t;pred_up=dp>t;pred_down=dp<-t;truth_flat=abs(dy)<=t;pred_flat=abs(dp)<=t
 for name,mask in {'rise_predicted_down':truth_up&pred_down,'fall_predicted_up':truth_down&pred_up,'true_change_predicted_flat':(~truth_flat)&pred_flat,'true_flat_predicted_change':truth_flat&(~pred_flat)}.items(): rows.append({'section':'direction_error','split':split,'seed':seed,'scope':name,'horizon':'all12_steps','n':int(mask.size),'event_count':int(mask.sum()),'event_rate':float(mask.mean()),'threshold_kw':t})
 return q

def explanatory_features(d,origins):
 r=d['raw_features'];p=d['power'];names=['current_power','recent_power_slope_6','recent_power_range_6','MB0_mean','MB1_mean','MB2_mean','irradiance_std_mean','tod_sin','tod_cos','doy_sin','doy_cos','minimum_valid_fraction','recent_missing_fraction'];x=[]
 for o in origins:
  hist=p[o-5:o+1];valid=r[o-71:o+1][:,[6,14,22]];x.append([p[o],(hist[-1]-hist[0])/5,float(np.ptp(hist)),r[o,5],r[o,13],r[o,21],np.nanmean(r[o,[8,16,24]]),r[o,1],r[o,2],r[o,3],r[o,4],np.nanmin(r[o,[6,14,22]]),float(np.mean(valid<.999999))])
 return np.asarray(x,np.float32),names

def main():
 HERE.mkdir(parents=True,exist_ok=True);c=json.loads(CONFIG_PATH.read_text(encoding='utf-8'));sources=[DATA_PATH,CONFIG_PATH]+[ART/str(s)/n for s in c['seeds'] for n in ('best_validation.pt','test_predictions.npz')];before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in sources};z=np.load(DATA_PATH,allow_pickle=True);d={k:z[k] for k in z.files};z.close();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[];cache={}
 # Thresholds derive only from Train labels and origin-observable Train features.
 tr=d['train_origins'];tr_y=np.stack([d['power'][o+1:o+13] for o in tr]);tr_prev=np.stack([d['power'][o:o+12] for o in tr]);changes=np.max(abs(tr_y-tr_prev),1);nonzero=abs(tr_y-tr_prev);nonzero=nonzero[nonzero>0];irr=np.nanmean(d['raw_features'][tr][:,[8,16,24]],1);cur=d['power'][tr]
 thresholds={'change_low':float(np.quantile(changes,.33)),'change_high':float(np.quantile(changes,.67)),'irr_std_median':float(np.nanmedian(irr)),'power_q33':float(np.quantile(cur,.33)),'power_q67':float(np.quantile(cur,.67)),'direction_tol':float(np.quantile(nonzero,.25))}
 for k,v in thresholds.items():rows.append({'section':'train_threshold','split':'train','seed':'all','scope':k,'horizon':'NA','value':v,'n':len(tr)})
 for seed in c['seeds']:
  for split in ('train','validation','test'):
   origins,y,p,ck=predictions(split,seed,d,c,device);cache[(seed,split)]=(origins,y,p);q=rows_for_split(rows,split,seed,origins,y,p,d,thresholds);saved_diff=math.nan
   if split=='test': saved=np.load(ART/str(seed)/'test_predictions.npz');saved_diff=float(np.max(abs(saved['predictions']-p)));assert saved_diff<=2e-6 and np.array_equal(saved['labels'],y) and np.array_equal(saved['forecast_origin_timestamp_ns'],d['times'][origins])
   rows.append({'section':'artifact_check','split':split,'seed':seed,'scope':'complete_split','horizon':12,'n':len(y),'prediction_shape':str(tuple(p.shape)),'labels_finite':bool(np.isfinite(y).all()),'predictions_finite':bool(np.isfinite(p).all()),'checkpoint_epoch':ck['epoch'],'checkpoint_validation_rmse_kw':ck['validation_rmse_kw'],'target_center':float(d['target_center']),'target_scale':float(d['target_scale']),'saved_prediction_max_abs_difference_kw':saved_diff,'artifact_source':str(ART/str(seed)/'best_validation.pt')})
 # Error predictability is designed on Train and measured on Validation only; never fitted to Test.
 for seed in c['seeds']:
  otr,ytr,ptr=cache[(seed,'train')];ova,yva,pva=cache[(seed,'validation')];xt,names=explanatory_features(d,otr);xv,_=explanatory_features(d,ova);et=np.sqrt(np.mean((ptr-ytr)**2,1));ev=np.sqrt(np.mean((pva-yva)**2,1));fill=np.nanmedian(xt,0);xt=np.where(np.isfinite(xt),xt,fill);xv=np.where(np.isfinite(xv),xv,fill)
  for j,name in enumerate(names):
   rt=spearmanr(xt[:,j],et).statistic;rv=spearmanr(xv[:,j],ev).statistic;rows.append({'section':'error_predictability','split':'train_validation','seed':seed,'scope':name,'horizon':12,'n':len(xv),'train_spearman':float(rt) if np.isfinite(rt) else math.nan,'validation_spearman':float(rv) if np.isfinite(rv) else math.nan,'uses_only_origin_or_past':True})
  model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.0,random_state=seed).fit(xt,et);pv=model.predict(xv);rows.append({'section':'error_predictability_model','split':'train_to_validation','seed':seed,'scope':'fixed_hist_gradient_boosting_explainer','horizon':12,'n':len(xv),'validation_r2':float(r2_score(ev,pv)),'validation_mae_kw':float(np.mean(abs(pv-ev))),'baseline_mae_kw':float(np.mean(abs(ev-np.median(et)))),'uses_only_origin_or_past':True})
 # Test artifact identity/fairness.
 ref=None
 for seed in c['seeds']:
  a=np.load(ART/str(seed)/'test_predictions.npz');ref=a if ref is None else ref;assert a['predictions'].shape==(len(d['test_origins']),12) and np.isfinite(a['predictions']).all() and np.isfinite(a['labels']).all();assert np.array_equal(a['labels'],ref['labels']) and np.array_equal(a['forecast_origin_timestamp_ns'],ref['forecast_origin_timestamp_ns'])
 fields=[]
 for row in rows:
  for k in row:
   if k not in fields:fields.append(k)
 with OUT.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 after={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in sources};assert before==after and OUT.resolve().parent==HERE.resolve()
 print(json.dumps({'device':str(device),'rows':len(rows),'thresholds':thresholds,'trained_neural_network':False}))
if __name__=='__main__':main()

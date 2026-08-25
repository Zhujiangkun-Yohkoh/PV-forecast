from __future__ import annotations
import copy,csv,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parent;CONFIG=ROOT/'config.json';RESULTS=ROOT/'results';METRICS=ROOT/'metrics_per_seed.csv'
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
INFO=load_module('alicd_info',ROOT.parent/'asoc_multirate_information_screen'/'run_information_screen.py')
def cfg():return json.loads(CONFIG.read_text(encoding='utf-8'))
def resolve(p):return (ROOT/Path(p)).resolve()
def seed_all(seed):np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

class ALICDDataset(Dataset):
 def __init__(self,features,power,origins,cols,c,center,scale):self.features,self.power,self.origins,self.cols,self.c,self.center,self.scale=features,power,origins,cols,c,center,scale
 def __len__(self):return len(self.origins)
 def __getitem__(self,i):
  o=int(self.origins[i]);x=self.features[o-self.c['lookback']+1:o+1,self.cols];y=(self.power[o+1:o+self.c['horizon']+1]-self.center)/self.scale;y0=(self.power[o]-self.center)/self.scale
  return torch.from_numpy(x.copy()),torch.from_numpy(y.astype(np.float32)),torch.tensor(y0,dtype=torch.float32),torch.tensor(o,dtype=torch.long)

class ALICD(nn.Module):
 def __init__(self,input_dim,c):
  super().__init__();base=INFO.ModernTCN(input_dim,c);self.backbone=base.net;d=c['model']['channels']*c['lookback'];h=c['horizon'];self.delta_head=nn.Linear(d,h);self.anchor_head=nn.Linear(d,3);self.anchor_indices=tuple(c['anchor_indices_zero_based'])
  C=torch.tril(torch.ones(h,h,dtype=torch.float64));S=torch.zeros(3,h,dtype=torch.float64)
  for i,j in enumerate(self.anchor_indices):S[i,j]=1
  A=S@C
  if int(torch.linalg.matrix_rank(A))!=3:raise ValueError('A is not full row rank')
  P=A.T@torch.linalg.inv(A@A.T);self.register_buffer('C',C);self.register_buffer('S',S);self.register_buffer('A',A);self.register_buffer('projection',P)
 def forward(self,x,y0_scaled):
  z=self.backbone(x.transpose(1,2)).flatten(1);delta_raw=self.delta_head(z);anchor_raw=self.anchor_head(z);A=self.A.to(delta_raw);P=self.projection.to(delta_raw);C=self.C.to(delta_raw);b=anchor_raw-y0_scaled[:,None];correction=(b-delta_raw@A.T)@P.T;delta_projected=delta_raw+correction;trajectory=y0_scaled[:,None]+delta_projected@C.T;pretrajectory=y0_scaled[:,None]+delta_raw@C.T
  return {'trajectory':trajectory,'delta_raw':delta_raw,'anchor_raw':anchor_raw,'delta_projected':delta_projected,'pretrajectory':pretrajectory,'correction':correction}

def target_delta(y,y0):return torch.cat([(y[:,0]-y0)[:,None],y[:,1:]-y[:,:-1]],1)
def losses(out,y,y0,c):
 lt=F.mse_loss(out['trajectory'],y);la=F.mse_loss(out['anchor_raw'],y[:,c['anchor_indices_zero_based']]);li=F.mse_loss(out['delta_projected'],target_delta(y,y0));w=c['loss_weights'];total=w['trajectory']*lt+w['anchor']*la+w['increment']*li;return lt,la,li,total

def prepare(c):
 z=np.load(resolve(c['prepared_data']),allow_pickle=True);d={k:z[k] for k in z.files};z.close();center=float(d['target_center']);scale=float(d['target_scale']);sets={s:ALICDDataset(d['scaled_features'],d['power'],d[f'{s}_origins'],d['base_cols'],c,center,scale) for s in ('train','validation','test')};lds={s:DataLoader(v,batch_size=c['training']['batch_size'],shuffle=s=='train',num_workers=c['training']['num_workers'],pin_memory=torch.cuda.is_available()) for s,v in sets.items()};return d,sets,lds,center,scale

def train_model(model,train_loader,validation_loader,c,device,run_dir):
 """Validation-only checkpointing; intentionally no Test loader argument."""
 t=c['training'];opt=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']);run_dir.mkdir(parents=True,exist_ok=True);log=run_dir/'epochs.jsonl';log.write_text('',encoding='utf-8');best=math.inf;best_epoch=0;stale=0;finite=True;times=[];started=time.perf_counter()
 for epoch in range(1,t['max_epochs']+1):
  tick=time.perf_counter();model.train();parts=[]
  for x,y,y0,_ in train_loader:
   x,y,y0=x.to(device),y.to(device),y0.to(device);opt.zero_grad(set_to_none=True);out=model(x,y0);ls=losses(out,y,y0,c);loss=ls[-1]
   if not torch.isfinite(loss):finite=False;raise FloatingPointError('non-finite training loss')
   loss.backward()
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):finite=False;raise FloatingPointError('non-finite gradient')
   torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip_norm']);opt.step();parts.append([float(v.detach().cpu()) for v in ls])
  model.eval();vp=[]
  with torch.no_grad():
   for x,y,y0,_ in validation_loader:
    x,y,y0=x.to(device),y.to(device),y0.to(device);vp.append([float(v.cpu()) for v in losses(model(x,y0),y,y0,c)])
  tr=np.mean(parts,0);va=np.mean(vp,0);elapsed=time.perf_counter()-tick;times.append(elapsed);rec={'epoch':epoch,'train_trajectory_loss':tr[0],'train_anchor_loss':tr[1],'train_increment_loss':tr[2],'train_total_loss':tr[3],'validation_trajectory_loss':va[0],'validation_anchor_loss':va[1],'validation_increment_loss':va[2],'validation_total_loss':va[3],'seconds':elapsed}
  with log.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
  torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'validation_total_loss':float(va[3])},run_dir/'last.pt')
  if va[3]<best-t['min_delta']:
   best,best_epoch,stale=float(va[3]),epoch,0;torch.save({'epoch':epoch,'state_dict':copy.deepcopy(model.state_dict()),'validation_total_loss':float(va[3])},run_dir/'best_validation.pt')
  else:
   stale+=1
   if stale>=t['patience']:break
 return {'actual_epochs':epoch,'best_epoch':best_epoch,'best_validation_total_loss':best,'stop_reason':'max_epochs' if epoch==t['max_epochs'] else 'early_stopping','numerically_finite':finite,'training_seconds':time.perf_counter()-started,'mean_epoch_seconds':float(np.mean(times))}

def predict_alicd(model,loader,device,center,scale):
 model.eval();store={k:[] for k in ('trajectory','delta_raw','anchor_raw','delta_projected','pretrajectory','correction')};ys=[];orig=[];start=time.perf_counter()
 with torch.no_grad():
  for x,y,y0,o in loader:
   out=model(x.to(device),y0.to(device));
   for k in store:store[k].append(out[k].cpu().numpy())
   ys.append(y.numpy());orig.append(o.numpy())
 elapsed=time.perf_counter()-start
 for k in store:store[k]=np.concatenate(store[k])
 store['trajectory_kw']=store['trajectory']*scale+center;store['pretrajectory_kw']=store['pretrajectory']*scale+center;store['labels_kw']=np.concatenate(ys)*scale+center;store['origins']=np.concatenate(orig);return store,elapsed

def predict_baseline(seed,split,d,c,device,center,scale):
 ds=INFO.WindowDataset(d['scaled_features'],d['power'],d[f'{split}_origins'],d['base_cols'],c['lookback'],c['horizon'],center,scale);ld=DataLoader(ds,batch_size=512,shuffle=False);m=INFO.ModernTCN(len(d['base_cols']),c).to(device);ck=torch.load(resolve(c['trajectory_results'])/str(seed)/'best_validation.pt',map_location=device,weights_only=True);m.load_state_dict(ck['state_dict']);m.eval();p=[]
 start=time.perf_counter()
 with torch.no_grad():
  for x,_ in ld:p.append(m(x.to(device)).cpu().numpy())
 return np.concatenate(p)*scale+center,time.perf_counter()-start

def basic(y,p):
 e=p-y;den=float(np.sum((y-y.mean())**2));return {'rmse_kw':float(np.sqrt(np.mean(e*e))),'mae_kw':float(np.mean(abs(e))),'bias_kw':float(np.mean(e)),'r2':1-float(np.sum(e*e))/den if den else math.nan}
def dynamics(y,p,y0,tol):
 dy=np.diff(np.column_stack([y0,y]),axis=1);dp=np.diff(np.column_stack([y0,p]),axis=1);tv_y=abs(dy).sum(1);tv_p=abs(dp).sum(1);truth_change=abs(dy)>tol;pred_change=abs(dp)>tol
 return {'diff_mae_kw':float(np.mean(abs(dp-dy))),'change_amplitude_ratio':float(np.mean(abs(dp))/max(np.mean(abs(dy)),1e-12)),'total_variation_ratio':float(tv_p.sum()/max(tv_y.sum(),1e-12)),'peak_rate_ratio':float(abs(dp).max(1).mean()/max(abs(dy).max(1).mean(),1e-12)),'direction_accuracy':float(np.mean(np.sign(dp)==np.sign(dy))),'true_change_predicted_flat_rate':float(np.mean(truth_change&~pred_change)),'true_flat_predicted_change_rate':float(np.mean(~truth_change&pred_change))}

def scenario_masks(d,origins,y,c):
 times=pd.to_datetime(d['times'][origins]);h=times.hour.to_numpy();power=d['power'];y0=power[origins];dy=np.diff(np.column_stack([y0,y]),axis=1);high=np.max(abs(dy),1)>=c['scenario_thresholds_train_only']['high_change_kw'];day=(y>.063).any(1);irr=np.nanmean(d['raw_features'][origins][:,[8,16,24]],1);mid=(h>=10)&(h<15)
 return {'full_timeline':np.ones(len(y),bool),'daylight':day,'high_change_daylight':high&day,'midday':mid,'sunrise':(h>=5)&(h<9),'sunset':(h>=16)&(h<20),'stable_low_change':np.max(abs(dy),1)==0,'h9_h12':np.ones(len(y),bool),'high_hf_irradiance_volatility':irr>c['scenario_thresholds_train_only']['irradiance_std_median'],'low_hf_irradiance_volatility':irr<=c['scenario_thresholds_train_only']['irradiance_std_median']}

def evaluate(rows,model,seed,split,d,c,p,y,origins,info,diag=None):
 y0=d['power'][origins];tol=c['scenario_thresholds_train_only']['direction_tolerance_kw'];total_sse=float(np.sum((p-y)**2))
 for h in range(12):rows.append({'section':'lead_time','model':model,'seed':seed,'split':split,'scope':'full_timeline','horizon':h+1,'n':len(y),**basic(y[:,h],p[:,h]),**dynamics(y[:,:h+1],p[:,:h+1],y0,tol),**info})
 for scope,mask in scenario_masks(d,origins,y,c).items():
  if not mask.any():continue
  yy,pp=y[mask],p[mask];start=8 if scope=='h9_h12' else 0;rows.append({'section':'scenario','model':model,'seed':seed,'split':split,'scope':scope,'horizon':'9-12' if start else 12,'n':int(mask.sum()),**basic(yy[:,start:],pp[:,start:]),**dynamics(yy[:,start:],pp[:,start:],y0[mask] if start==0 else yy[:,start-1],tol),'sse_share':float(np.sum((pp[:,start:]-yy[:,start:])**2)/total_sse),**info})
 if diag is not None:
  idx=c['anchor_indices_zero_based'];scale=float(d['target_scale']);anchor_kw=diag['anchor_raw']*scale+float(d['target_center']);constraint=np.max(abs(diag['trajectory'][:,idx]-diag['anchor_raw']),axis=0);corr=np.linalg.norm(diag['correction'],axis=1);raw=np.linalg.norm(diag['delta_raw'],axis=1)
  for j,h in enumerate((3,6,12)):rows.append({'section':'projection','model':model,'seed':seed,'split':split,'scope':f'anchor_h{h}','horizon':h,'n':len(y),'anchor_rmse_kw':float(np.sqrt(np.mean((anchor_kw[:,j]-y[:,idx[j]])**2))),'anchor_bias_kw':float(np.mean(anchor_kw[:,j]-y[:,idx[j]])),'constraint_max_abs_scaled':float(constraint[j]),**info})
  rows.append({'section':'projection','model':model,'seed':seed,'split':split,'scope':'projection_summary','horizon':12,'n':len(y),'preprojection_rmse_kw':basic(y,diag['pretrajectory_kw'])['rmse_kw'],'postprojection_rmse_kw':basic(y,p)['rmse_kw'],'mean_correction_norm_scaled':float(corr.mean()),'correction_to_raw_increment_norm_ratio':float(np.mean(corr/np.maximum(raw,1e-12))),'projection_tv_ratio':dynamics(y,p,y0,tol)['total_variation_ratio'],'preprojection_tv_ratio':dynamics(y,diag['pretrajectory_kw'],y0,tol)['total_variation_ratio'],**info})

def write(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with METRICS.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run():
 c=cfg();d,sets,lds,center,scale=prepare(c);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[]
 for seed in c['seeds']:
  seed_all(seed);model=ALICD(len(d['base_cols']),c).to(device);run_dir=RESULTS/'ALICD'/str(seed);training=train_model(model,lds['train'],lds['validation'],c,device,run_dir);best=torch.load(run_dir/'best_validation.pt',map_location=device,weights_only=True);model.load_state_dict(best['state_dict']);base_params=sum(p.numel() for p in INFO.ModernTCN(len(d['base_cols']),c).parameters());backbone_params=sum(p.numel() for p in model.backbone.parameters());delta_params=sum(p.numel() for p in model.delta_head.parameters());anchor_params=sum(p.numel() for p in model.anchor_head.parameters());total=sum(p.numel() for p in model.parameters())
  for split in ('validation','test'):
   diag,infer=predict_alicd(model,lds[split],device,center,scale);origins=d[f'{split}_origins'];y=np.stack([d['power'][o+1:o+13] for o in origins]);assert np.array_equal(diag['origins'],origins) and np.allclose(diag['labels_kw'],y,atol=2e-6);baseline,base_infer=predict_baseline(seed,split,d,c,device,center,scale)
   if split=='test':
    with np.load(resolve(c['trajectory_results'])/str(seed)/'test_predictions.npz') as old:
     assert np.array_equal(old['forecast_origin_timestamp_ns'],d['times'][origins]) and np.allclose(old['labels'],y,atol=2e-6) and np.allclose(old['predictions'],baseline,atol=2e-6)
   info={'actual_epochs':training['actual_epochs'],'best_epoch':training['best_epoch'],'best_validation_total_loss':training['best_validation_total_loss'],'stop_reason':training['stop_reason'],'numerically_finite':training['numerically_finite'],'training_seconds':training['training_seconds'],'mean_epoch_seconds':training['mean_epoch_seconds'],'backbone_parameters':backbone_params,'delta_head_parameters':delta_params,'anchor_head_parameters':anchor_params,'total_parameters':total,'baseline_parameters':base_params,'parameter_increase_pct':(total/base_params-1)*100,'inference_ms_per_sample':infer/len(y)*1000,'artifact_source':str(run_dir/'test_predictions.npz' if split=='test' else run_dir/'best_validation.pt')}
   evaluate(rows,'TRAJECTORY_ONLY',seed,split,d,c,baseline,y,origins,{'baseline_parameters':base_params,'inference_ms_per_sample':base_infer/len(y)*1000,'artifact_source':str(resolve(c['trajectory_results'])/str(seed)/'best_validation.pt')});evaluate(rows,'ALICD',seed,split,d,c,diag['trajectory_kw'],y,origins,info,diag)
   if split=='test':np.savez_compressed(run_dir/'test_predictions.npz',predictions=diag['trajectory_kw'].astype(np.float32),labels=y.astype(np.float32),forecast_origin_timestamp_ns=d['times'][origins],origins=origins,delta_raw=diag['delta_raw'].astype(np.float32),anchor_raw=diag['anchor_raw'].astype(np.float32),delta_projected=diag['delta_projected'].astype(np.float32),pretrajectory_kw=diag['pretrajectory_kw'].astype(np.float32))
  write(rows)
 print(json.dumps({'device':str(device),'completed_runs':3,'metrics_rows':len(rows),'neural_training':True}))
if __name__=='__main__':run()

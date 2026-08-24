from __future__ import annotations
import argparse, copy, csv, importlib.util, inspect, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent; CONFIG=ROOT/'config.json'; RESULTS=ROOT/'results'; METRICS=ROOT/'metrics_per_seed.csv'

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

INFO=load_module('frozen_info',ROOT.parent/'asoc_multirate_information_screen'/'run_information_screen.py')
JOINT=load_module('frozen_joint',ROOT.parent/'asoc_joint_trajectory_onset_screen'/'run_joint_onset_screen.py')

def cfg(): return json.loads(CONFIG.read_text(encoding='utf-8'))
def resolve(p): return (ROOT/Path(p)).resolve()
def seed_all(seed): np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class FrozenTrajectoryHazard(nn.Module):
 def __init__(self,input_dim,c,trajectory_state):
  super().__init__(); base=INFO.ModernTCN(input_dim,c); base.load_state_dict(trajectory_state)
  self.backbone=base.net; self.power_head=base.out; self.h=c['horizon']; self.hazard_head=nn.Linear(c['model']['channels']*c['lookback'],self.h*3)
  for p in list(self.backbone.parameters())+list(self.power_head.parameters()): p.requires_grad_(False)
  self.backbone.eval(); self.power_head.eval()
 def train(self,mode=True):
  super().train(mode); self.backbone.eval(); self.power_head.eval(); self.hazard_head.train(mode); return self
 def forward(self,x):
  with torch.no_grad(): z=self.backbone(x.transpose(1,2)).flatten(1); power=self.power_head(z)
  return power,self.hazard_head(z.detach()).reshape(-1,self.h,3)

def snapshot_frozen(model):
 return {f'p:{prefix}.{n}':v.detach().cpu().clone() for prefix,module in [('backbone',model.backbone),('power_head',model.power_head)] for n,v in module.named_parameters()} | {f'b:{prefix}.{n}':v.detach().cpu().clone() for prefix,module in [('backbone',model.backbone),('power_head',model.power_head)] for n,v in module.named_buffers()}

def frozen_unchanged(model,snap):
 now=snapshot_frozen(model); return set(now)==set(snap) and all(torch.equal(now[k],v) for k,v in snap.items())

def make_data(c):
 d=np.load(resolve(c['prepared_data']),allow_pickle=True); columns=d['base_cols']; center=float(d['target_center']); scale=float(d['target_scale']); threshold=float(d['ramp_threshold'])
 sets={s:JOINT.JointDataset(d['scaled_features'],d['power'],d[f'{s}_origins'],columns,c,center,scale,threshold) for s in ('train','validation','test')}
 loaders={s:DataLoader(ds,batch_size=c['training']['batch_size'],shuffle=s=='train',num_workers=c['training']['num_workers'],pin_memory=torch.cuda.is_available()) for s,ds in sets.items()}
 return d,sets,loaders,center,scale,threshold

def train_hazard(model,train_loader,validation_loader,c,device,run_dir):
 """Train only the hazard head; deliberately accepts no Test loader."""
 t=c['training']; params=list(model.hazard_head.parameters()); opt=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay'])
 assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in params}
 snap=snapshot_frozen(model); best=math.inf; best_epoch=0; stale=0; times=[]; start=time.perf_counter(); finite=True; run_dir.mkdir(parents=True,exist_ok=True); log=run_dir/'epochs.jsonl'; log.write_text('',encoding='utf-8')
 for epoch in range(1,t['max_epochs']+1):
  tick=time.perf_counter(); model.train(); losses=[]
  for x,_,_,first,cause in train_loader:
   x,first,cause=x.to(device),first.to(device),cause.to(device); opt.zero_grad(set_to_none=True); _,hazard=model(x); loss=JOINT.hazard_nll(hazard,first,cause)
   if not torch.isfinite(loss): finite=False; raise FloatingPointError('non-finite training loss')
   loss.backward()
   if any(p.grad is not None for p in list(model.backbone.parameters())+list(model.power_head.parameters())): raise AssertionError('frozen gradient detected')
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params): finite=False; raise FloatingPointError('missing/non-finite hazard gradient')
   torch.nn.utils.clip_grad_norm_(params,t['gradient_clip_norm']); opt.step(); losses.append(float(loss.detach().cpu()))
  model.eval(); vals=[]
  with torch.no_grad():
   for x,_,_,first,cause in validation_loader:
    _,hazard=model(x.to(device)); vals.append(float(JOINT.hazard_nll(hazard,first.to(device),cause.to(device)).cpu()))
  val=float(np.mean(vals)); elapsed=time.perf_counter()-tick; times.append(elapsed); rec={'epoch':epoch,'train_first_event_nll':float(np.mean(losses)),'validation_first_event_nll':val,'seconds':elapsed}
  with log.open('a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
  torch.save({'epoch':epoch,'hazard_head_state':model.hazard_head.state_dict(),'optimizer':opt.state_dict(),'validation_first_event_nll':val},run_dir/'last.pt')
  if val<best-t['min_delta']:
   best,best_epoch,stale=val,epoch,0; torch.save({'epoch':epoch,'hazard_head_state':copy.deepcopy(model.hazard_head.state_dict()),'validation_first_event_nll':val},run_dir/'best_validation.pt')
  else:
   stale+=1
   if stale>=t['patience']: break
 assert frozen_unchanged(model,snap)
 return {'actual_epochs':epoch,'best_epoch':best_epoch,'best_validation_first_event_nll':best,'stop_reason':'max_epochs' if epoch==t['max_epochs'] else 'early_stopping','numerically_finite':finite,'training_seconds':time.perf_counter()-start,'mean_epoch_seconds':float(np.mean(times))}

def predict(model,loader,device,center,scale):
 model.eval(); pp=[];ee=[]; start=time.perf_counter()
 with torch.no_grad():
  for x,*_ in loader:
   p,e=model(x.to(device)); pp.append((p.cpu().numpy()*scale+center).astype(np.float32)); ee.append(e.cpu().numpy().astype(np.float32))
 return np.concatenate(pp),np.concatenate(ee),time.perf_counter()-start

def model_counts(model):
 bp=sum(p.numel() for p in model.backbone.parameters()); pp=sum(p.numel() for p in model.power_head.parameters()); hp=sum(p.numel() for p in model.hazard_head.parameters()); return bp,pp,hp,bp+pp+hp,hp

def artifact(kind,seed,c):
 if kind=='TRAJECTORY_ONLY': return resolve(c['trajectory_results'])/str(seed)/'test_predictions.npz'
 if kind in ('STEP_MULTITASK','STANDARD_ONSET_HAZARD'):
  folder='STEP_MULTITASK' if kind=='STEP_MULTITASK' else 'ONSET_HAZARD'; return resolve(c['joint_results'])/folder/str(seed)/'test_predictions.npz'
 return resolve(c['power_anchored_results'])/str(seed)/'test_predictions.npz'

def add_metrics(rows,model_name,kind,seed,a,d,c,threshold,info):
 origins=d['test_origins']; power=d['power']; y,delta,step,onset,prior,first,cause=JOINT.event_labels(power,origins,threshold,c['horizon']); times=d['times']; wm,day,transition,_=JOINT.scopes(times,origins,y,onset); p=a['predictions'] if 'predictions' in a.files else a['power_predictions']; event=a['event_output'] if 'event_output' in a.files else None
 if model_name in ('TRAJECTORY_ONLY','FROZEN_TRAJECTORY_HAZARD'): JOINT.append_power(rows,model_name,seed,y,p,step,onset,day,transition,info,str(info['artifact_source']))
 if event is not None:
  n=len(rows); JOINT.append_events(rows,'STEP_MULTITASK' if kind=='STEP_MULTITASK' else 'ONSET_HAZARD',seed,y,delta,step,onset,first,cause,prior,power[origins],p,event,wm,day,transition,info)
  for r in rows[n:]: r['model']=model_name
 return y,p,origins

def write_csv(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields: fields.append(k)
 with METRICS.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def run():
 c=cfg(); seed_all(c['seeds'][0]); d,sets,lds,center,scale,threshold=make_data(c); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
 for seed in c['seeds']:
  seed_all(seed); ck=torch.load(resolve(c['trajectory_results'])/str(seed)/'best_validation.pt',map_location='cpu',weights_only=True); model=FrozenTrajectoryHazard(len(d['base_cols']),c,ck['state_dict']).to(device); before=snapshot_frozen(model); run_dir=RESULTS/'FROZEN_TRAJECTORY_HAZARD'/str(seed)
  training=train_hazard(model,lds['train'],lds['validation'],c,device,run_dir); best=torch.load(run_dir/'best_validation.pt',map_location=device,weights_only=True); model.hazard_head.load_state_dict(best['hazard_head_state']); assert frozen_unchanged(model,before)
  pred,event,infer=predict(model,lds['test'],device,center,scale); old=np.load(artifact('TRAJECTORY_ONLY',seed,c)); labels=old['labels']; timestamps=old['forecast_origin_timestamp_ns']; maxdiff=float(np.max(np.abs(pred-old['predictions']))); meandiff=float(np.mean(np.abs(pred-old['predictions']))); assert maxdiff<=2e-6
  y,delta,step,onset,prior,first,cause=JOINT.event_labels(d['power'],d['test_origins'],threshold,c['horizon']); np.savez_compressed(run_dir/'test_predictions.npz',power_predictions=pred,event_output=event,labels=labels,forecast_origin_timestamp_ns=timestamps,step_mask=step,onset_mask=onset,first_onset=first,cause=cause)
  bp,pp,hp,total,trainable=model_counts(model); info={**training,'backbone_parameters':bp,'power_head_parameters':pp,'event_head_parameters':hp,'total_inference_parameters':total,'trainable_parameters':trainable,'trainable_fraction':trainable/total,'inference_ms_per_sample':infer/len(pred)*1000,'threshold_kw':threshold,'power_max_abs_difference_kw':maxdiff,'power_mean_abs_difference_kw':meandiff,'artifact_source':str(run_dir/'test_predictions.npz')}
  for name,kind in [('TRAJECTORY_ONLY','TRAJECTORY_ONLY'),('STEP_MULTITASK','STEP_MULTITASK'),('STANDARD_ONSET_HAZARD','ONSET_HAZARD'),('POWER_ANCHORED_HAZARD','ONSET_HAZARD'),('FROZEN_TRAJECTORY_HAZARD','ONSET_HAZARD')]:
   ar=np.load(run_dir/'test_predictions.npz') if name=='FROZEN_TRAJECTORY_HAZARD' else np.load(artifact(name,seed,c)); yy,ppp,oo=add_metrics(rows,name,kind,seed,ar,d,c,threshold,info if name=='FROZEN_TRAJECTORY_HAZARD' else {'artifact_source':str(artifact(name,seed,c))})
   assert np.array_equal(yy,labels) and np.array_equal(d['times'][oo],timestamps)
  rows.append({'section':'power_identity','model':'FROZEN_TRAJECTORY_HAZARD','seed':seed,'scope':'complete_test','horizon':12,'max_abs_difference_kw':maxdiff,'mean_abs_difference_kw':meandiff,'rmse_difference_kw':float(np.sqrt(np.mean((pred-labels)**2))-np.sqrt(np.mean((old['predictions']-labels)**2))),'mae_difference_kw':float(np.mean(np.abs(pred-labels))-np.mean(np.abs(old['predictions']-labels))),**info})
  write_csv(rows)
 print(json.dumps({'device':str(device),'runs':len(c['seeds']),'threshold_kw':threshold,'metrics_rows':len(rows)}))

if __name__=='__main__': run()

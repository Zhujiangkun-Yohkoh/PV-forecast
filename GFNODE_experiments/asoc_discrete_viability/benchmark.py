"""Recoverable, leakage-free viability benchmark for the discrete candidate."""
from __future__ import annotations
import argparse, copy, csv, json, random, time, sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parent.parent.parent))

from GFNODE_experiments.asoc_clean_decision.asoc_clean_decision import (
    CleanDataProtocol, DiscreteTrajectoryDecoder, evaluate_prefixes, parameter_count,
)

ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT.parent.parent)); ART=ROOT/'artifacts'; HORIZONS=(12,48,96,144)
def cfg(): return json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
def seed(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def device(c): return torch.device('cuda' if c['device']=='cuda_if_available' and torch.cuda.is_available() else 'cpu')

class ITransformer(nn.Module):
 def __init__(self,n,c):
  super().__init__(); b=c['baselines']['iTransformer']; d=b['d_model']; self.proj=nn.Linear(72,d); l=nn.TransformerEncoderLayer(d,b['heads'],d*2,batch_first=True,activation='gelu'); self.enc=nn.TransformerEncoder(l,b['layers']); self.out=nn.Sequential(nn.Flatten(),nn.Linear(n*d,144))
 def forward(self,x): return self.out(self.enc(self.proj(x.transpose(1,2))))
class PatchTST(nn.Module):
 def __init__(self,n,c):
  super().__init__(); b=c['baselines']['PatchTST']; self.p=b['patch_length']; self.s=b['stride']; d=b['d_model']; self.np=(72-self.p)//self.s+1; self.proj=nn.Linear(self.p*n,d); l=nn.TransformerEncoderLayer(d,b['heads'],d*2,batch_first=True,activation='gelu'); self.enc=nn.TransformerEncoder(l,b['layers']); self.out=nn.Sequential(nn.Flatten(),nn.Linear(self.np*d,144))
 def forward(self,x):
  p=x.unfold(1,self.p,self.s).permute(0,1,3,2).reshape(x.size(0),self.np,-1); return self.out(self.enc(self.proj(p)))
class ModernTCN(nn.Module):
 def __init__(self,n,c):
  super().__init__(); b=c['baselines']['ModernTCN']; ch=b['channels']; layers=[nn.Conv1d(n,ch,1),nn.GELU()]
  for i in range(b['layers']): layers += [nn.Conv1d(ch,ch,b['kernel_size'],padding=b['kernel_size']//2,groups=ch),nn.Conv1d(ch,ch,1),nn.GELU()]
  self.net=nn.Sequential(*layers); self.out=nn.Linear(ch*72,144)
 def forward(self,x): return self.out(self.net(x.transpose(1,2)).flatten(1))
def model(name,n,c):
 if name=='Discrete Candidate': return DiscreteTrajectoryDecoder(n,c['model'])
 if name=='iTransformer': return ITransformer(n,c)
 if name=='PatchTST': return PatchTST(n,c)
 return ModernTCN(n,c)
def loaders(w,b):
 def f(k,shuffle): return DataLoader(TensorDataset(torch.from_numpy(w[k].x),torch.from_numpy(w[k].y_scaled)),batch_size=b,shuffle=shuffle,pin_memory=torch.cuda.is_available())
 return f('train',True),f('validation',False),f('test',False)
def status(rows):
 with (ROOT/'run_status.csv').open('w',newline='',encoding='utf-8') as h:
  wr=csv.DictWriter(h,fieldnames=['run_id','model','dataset','seed','status']);wr.writeheader();wr.writerows(rows)
def train(m,tr,va,c,dev,run):
 opt=torch.optim.AdamW(m.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']); loss=nn.MSELoss(); best=1e99; stale=0; beststate=None; started=time.perf_counter(); log=run/'epochs.jsonl';
 for e in range(1,c['training']['max_epochs']+1):
  m.train()
  for x,y in tr:
   opt.zero_grad(set_to_none=True); z=loss(m(x.to(dev)),y.to(dev));
   if not torch.isfinite(z): raise FloatingPointError('non-finite loss')
   z.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  m.eval(); vs=[]
  with torch.no_grad():
   for x,y in va: vs.append(float(loss(m(x.to(dev)),y.to(dev)).cpu()))
  v=float(np.mean(vs)); (run/'last.pt').write_bytes(b'') if False else torch.save({'epoch':e,'state_dict':m.state_dict(),'optimizer':opt.state_dict()},run/'last.pt')
  with log.open('a',encoding='utf-8') as h: h.write(json.dumps({'epoch':e,'validation_mse':v})+'\n')
  if v<best-1e-8: best=v;be=e;stale=0;beststate=copy.deepcopy(m.state_dict());torch.save({'epoch':e,'state_dict':beststate,'validation_mse':v},run/'best_validation.pt')
  else:
   stale+=1
   if stale>=c['training']['patience']: break
 return beststate,{'actual_epochs':e,'best_epoch':be,'best_validation_mse':best,'training_seconds':time.perf_counter()-started,'numerically_finite':True}
def predict(m,te,dev):
 m.eval();a=[]
 with torch.no_grad():
  for x,_ in te:a.append(m(x.to(dev)).cpu().numpy())
 return np.concatenate(a)
def rows_for(dataset,name,s,w,proto,info,run):
 _,_,te=loaders(w,cfg()['training']['batch_size']); raw=proto.target_scaler.inverse_transform(predict_current.reshape(-1,1)).reshape(predict_current.shape).astype('float32'); t=w['test']; np.savez_compressed(run/'test_H144.npz',predictions=raw,labels=t.y_raw,daylight_mask=t.day_mask,target_start=t.target_start)
 out=evaluate_prefixes(t.y_raw,raw,t.day_mask,proto.target_range)
 for r in out:r.update({'run_id':f'{name}_{dataset}_{s}','dataset':dataset,'model':name,'seed':s,'parameter_count':parameter_count(current_model),**info,'prediction_file':str((run/'test_H144.npz').relative_to(ROOT))})
 return out
def write(path,rows):
 if rows:
  with path.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 global current_model,predict_current
 c=cfg(); ART.mkdir(exist_ok=True); dev=device(c); allrows=[]; states=[]
 for ds in c['datasets']:
  p=CleanDataProtocol(c,ds);p.load_regularized_raw();w=p.fit_transform()
  for name in ('Discrete Candidate','iTransformer','PatchTST','ModernTCN'):
   for s in c['seeds']:
    rid=f'{name}_{ds}_{s}'; run=ART/rid; run.mkdir(parents=True,exist_ok=True); states.append({'run_id':rid,'model':name,'dataset':ds,'seed':s,'status':'running'});status(states)
    if name=='Discrete Candidate' and ds in ('Sanyo','Qcells'):
     old=ROOT.parent/'asoc_clean_decision'/'results'/ds/'Discrete'/f'seed_{s}'/'test_H144_predictions_and_labels.npz'
     if old.exists():
      src=np.load(old); raw=src['predictions']; t=w['test']
      # Ordinary equality check: same target and timestamp are required for reuse.
      if np.array_equal(src['labels'],t.y_raw) and np.array_equal(src['target_start'],t.target_start):
       shutil.copy2(old,run/'test_H144.npz'); out=evaluate_prefixes(t.y_raw,raw,t.day_mask,p.target_range)
       for r in out:r.update({'run_id':rid,'dataset':ds,'model':name,'seed':s,'parameter_count':98738,'actual_epochs':'reused','best_epoch':'reused','best_validation_mse':'reused','training_seconds':0.0,'numerically_finite':True,'prediction_file':str((run/'test_H144.npz').relative_to(ROOT))})
       allrows.extend(out);states[-1]['status']='completed';status(states);write(ROOT/'benchmark_per_seed.csv',allrows);continue
    seed(s);current_model=model(name,w['train'].x.shape[-1],c).to(dev); tr,va,_=loaders(w,c['training']['batch_size']); best,info=train(current_model,tr,va,c,dev,run);current_model.load_state_dict(best);predict_current=predict(current_model,loaders(w,c['training']['batch_size'])[2],dev);allrows.extend(rows_for(ds,name,s,w,p,info,run));states[-1]['status']='completed';status(states);write(ROOT/'benchmark_per_seed.csv',allrows)
 write(ROOT/'benchmark_per_seed.csv',allrows)
if __name__=='__main__': main()

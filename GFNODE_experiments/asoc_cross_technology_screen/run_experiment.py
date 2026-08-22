import argparse,csv,json,random,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
R=Path(__file__).resolve().parent;REPO=R.parent.parent;sys.path.insert(0,str(REPO))
from GFNODE_experiments.asoc_clean_decision.asoc_clean_decision import CleanDataProtocol,evaluate_prefixes,parameter_count
N=('Sanyo','Hanwha','Qcells'); H=(12,48,96,144)
def c():return json.loads((R/'config.json').read_text())
def sd(x):random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x)
class Block(nn.Module):
 def __init__(self,i,ch):super().__init__();self.n=nn.Sequential(nn.Conv1d(i,ch,1),nn.GELU(),nn.Conv1d(ch,ch,5,padding=2,groups=ch),nn.Conv1d(ch,ch,1),nn.GELU())
 def forward(self,x):return self.n(x.transpose(1,2)).flatten(1)
class JointEarlyFusionModernTCN(nn.Module):
 def __init__(self,cc=48):super().__init__();self.backbone=Block(15,cc);self.heads=nn.ModuleList([nn.Linear(cc*72,144) for _ in N])
 def forward(self,weather,private):z=self.backbone(torch.cat([weather,private.permute(0,2,1,3).reshape(private.size(0),private.size(2),-1)],-1));return torch.stack([h(z) for h in self.heads],1)
class SharedPrivateModernTCN(nn.Module):
 def __init__(self,cc=48):super().__init__();self.shared=Block(6,cc);self.private=nn.ModuleList([Block(3,cc//2) for _ in N]);self.fuse=nn.ModuleList([nn.Linear((cc+cc//2)*72,cc*72) for _ in N]);self.heads=nn.ModuleList([nn.Linear(cc*72,144) for _ in N])
 def forward(self,weather,private):
  s=self.shared(weather);return torch.stack([self.heads[i](self.fuse[i](torch.cat([s,self.private[i](private[:,i])],-1))) for i in range(3)],1)
def masked_loss(p,y,m):
 vals=[]
 for i in range(3):
  q=m[:,i].bool();vals.append(((p[:,i][q]-y[:,i][q])**2).mean() if q.any() else p[:,i].sum()*0)
 return torch.stack(vals).mean()
def evaluate_masked(y,p,valid,day,denominator):
 rows=[]
 for h in H:
  for scope,mask in [('regular_full_timeline',valid[:,:h]),('predefined_daylight',valid[:,:h]&day[:,:h])]:
   a=y[:,:h][mask];b=p[:,:h][mask];err=b-a;rmse=float(np.sqrt(np.mean(err**2)));rows.append({'evaluated_targets':int(mask.sum()),'rmse':rmse,'mae':float(np.mean(np.abs(err))),'r2':float(1-np.sum(err**2)/np.sum((a-a.mean())**2)) if len(a)>1 and np.sum((a-a.mean())**2)>0 else float('nan'),'nrmse':rmse/float(denominator),'horizon':h,'scope':scope,'prefix_source':'same_H144_prediction'})
 return rows
def build():
 base=json.loads((REPO/'GFNODE_experiments/asoc_discrete_viability/config.json').read_text());ps={}
 for n in N:p=CleanDataProtocol(base,n);p.load_regularized_raw();p.fit_transform();ps[n]=p
 out={}
 for split in ('train','validation','test'):
  # Preserve every structural 5-minute start.  Technology target validity stays in its own mask.
  frames={n:ps[n].transformed[split] for n in N};length=min(len(f) for f in frames.values());limit=length-base['lookback']-base['horizon']+1
  weather=[];priv=[];ys=[];valid=[];raw=[];day=[];ts=[];te=[]
  for i in range(limit):
   weather.append(frames['Sanyo'].iloc[i:i+72][[f'x_{j}' for j in range(1,7)]].to_numpy(np.float32))
   priv.append(np.stack([frames[n].iloc[i:i+72][['x_0','x_7','x_14']].to_numpy(np.float32) for n in N]))
   rr=[];ss=[];vv=[];dd=[]
   for n in N:
    f=frames[n].iloc[i+72:i+216];r=f['_target_raw'].to_numpy(np.float32);v=f['_target_valid'].to_numpy(bool);rr.append(r);vv.append(v);dd.append(f['_day_valid'].to_numpy(bool));z=np.zeros(144,np.float32);z[v]=ps[n].target_scaler.transform(r[v].reshape(-1,1)).reshape(-1) if v.any() else z[v];ss.append(z)
   weather[-1]=weather[-1];ys.append(np.stack(ss));valid.append(np.stack(vv));raw.append(np.stack(rr));day.append(np.stack(dd));t=np.array([frames[n].index[i+72].to_datetime64() for n in N]);ts.append(t);te.append(t+np.timedelta64(715,'m'))
  out[split]=tuple(map(np.stack,(weather,priv,ys,valid,raw,day,ts,te)))
 return base,ps,out
def loader(d,b,sh):return DataLoader(TensorDataset(*[torch.from_numpy(x) for x in d[:4]]),batch_size=b,shuffle=sh)
def train(model,tr,va,cf,dev,run,resume=False):
 opt=torch.optim.AdamW(model.parameters(),lr=cf['training']['learning_rate'],weight_decay=cf['training']['weight_decay']);best=1e99;be=0;stale=0;first=1;start=time.perf_counter()
 if resume:
  if (run/'best_validation.pt').exists():
   bk=torch.load(run/'best_validation.pt',map_location=dev,weights_only=False);best=float(bk['validation_loss']);be=int(bk['epoch'])
  if (run/'last.pt').exists():
   try:
    ck=torch.load(run/'last.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['state']);opt.load_state_dict(ck['opt']);first=int(ck['epoch'])+1
   except RuntimeError:
    # An interrupted checkpoint is not trusted; resume from the durable validation checkpoint.
    if be: model.load_state_dict(bk['state']);first=be+1
 for e in range(first,cf['training']['max_epochs']+1):
  model.train()
  for w,p,y,m in tr:opt.zero_grad();z=masked_loss(model(w.to(dev),p.to(dev)),y.to(dev),m.to(dev));z.backward();opt.step()
  model.eval();v=[]
  with torch.no_grad():
   for w,p,y,m in va:v.append(float(masked_loss(model(w.to(dev),p.to(dev)),y.to(dev),m.to(dev)).cpu()))
  x=float(np.mean(v));(run/'epochs.jsonl').open('a').write(json.dumps({'epoch':e,'validation_loss':x})+'\n');torch.save({'epoch':e,'state':model.state_dict(),'opt':opt.state_dict()},run/'last.pt')
  if x<best:best=x;be=e;stale=0;torch.save({'epoch':e,'state':model.state_dict(),'validation_loss':x},run/'best_validation.pt')
  else:stale+=1
  if stale>=cf['training']['patience']:break
 return e,be,best,time.perf_counter()-start
def pred(m,d,b,dev):
 a=[];m.eval()
 with torch.no_grad():
  for w,p,_,_ in loader(d,b,False):a.append(m(w.to(dev),p.to(dev)).cpu().numpy())
 return np.concatenate(a)
def write(path,rows):
 with (R/path).open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 cf=c();base,ps,data=build();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=pd.read_csv(R/'metrics.csv').to_dict('records') if (R/'metrics.csv').exists() else [];status=pd.read_csv(R/'run_status.csv').to_dict('records') if (R/'run_status.csv').exists() else [];art=R/'artifacts';art.mkdir(exist_ok=True)
 for name,Cls in [('JointEarlyFusionModernTCN',JointEarlyFusionModernTCN),('SharedPrivateModernTCN',SharedPrivateModernTCN)]:
  for s in cf['seeds']:
   sd(s);run=art/name/str(s);run.mkdir(parents=True,exist_ok=True);rid=f'{name}_{s}'
   if (run/'test_H144.npz').exists():continue
   status=[x for x in status if x['run_id']!=rid];status.append({'run_id':rid,'status':'running'});write('run_status.csv',status);m=Cls(cf['channels']).to(dev);e,be,vl,sec=train(m,loader(data['train'],cf['training']['batch_size'],True),loader(data['validation'],cf['training']['batch_size'],False),cf,dev,run,args.resume);ck=torch.load(run/'best_validation.pt',weights_only=False);m.load_state_dict(ck['state']);sp=pred(m,data['test'],cf['training']['batch_size'],dev);raw=np.empty_like(sp)
   for i,n in enumerate(N):raw[:,i]=ps[n].target_scaler.inverse_transform(sp[:,i].reshape(-1,1)).reshape(sp[:,i].shape)
   np.savez_compressed(run/'test_H144.npz',predictions=raw,labels=data['test'][4],target_valid_mask=data['test'][3],daylight_mask=data['test'][5],target_start=data['test'][6],target_end=data['test'][7])
   for i,n in enumerate(N):
    # Evaluate exactly the independent model's complete-H144 starts for a fair label-identical comparison.
    keep=np.isin(data['test'][6][:,i],ps[n].windows['test'].target_start)
    for x in evaluate_masked(data['test'][4][keep,i],raw[keep,i],data['test'][3][keep,i],data['test'][5][keep,i],ps[n].target_range):x.update({'model':name,'seed':s,'dataset':n,'parameters':parameter_count(m),'actual_epochs':e,'best_epoch':be,'best_validation_loss':vl,'training_seconds':sec});rows.append(x)
   status[-1]['status']='completed';write('run_status.csv',status);write('metrics.csv',rows)
 write('metrics.csv',rows)
if __name__=='__main__':main()

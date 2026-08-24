"""Leakage-free joint H12 power trajectory and first-ramp-onset screen."""
from __future__ import annotations
import copy,csv,json,math,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/"results"; METRICS=ROOT/"metrics_per_seed.csv"; PREFIXES=(3,6,12)
def cfg(): return json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def event_labels(power,origins,threshold,horizon=12):
 y=np.stack([power[o+1:o+horizon+1] for o in origins]);prev=np.stack([power[o:o+horizon] for o in origins]);delta=y-prev;step=np.abs(delta)>=threshold
 prior=np.asarray([abs(power[o]-power[o-1])>=threshold for o in origins]);onset=step&~np.column_stack([prior,step[:,:-1]])
 first=np.full(len(origins),-1,np.int64);cause=np.zeros(len(origins),np.int64)
 for i in range(len(origins)):
  where=np.flatnonzero(onset[i])
  if len(where): first[i]=where[0];cause[i]=1 if delta[i,where[0]]>0 else 2
 return y,delta,step,onset,prior,first,cause

class JointDataset(Dataset):
 def __init__(self,features,power,origins,columns,c,center,scale,threshold):
  self.features,self.origins,self.columns=features,origins,columns;self.c=c;y,_,step,_,_,first,cause=event_labels(power,origins,threshold,c["horizon"]);self.y=((y-center)/scale).astype(np.float32);self.step=step.astype(np.float32);self.first=first;self.cause=cause
 def __len__(self):return len(self.origins)
 def __getitem__(self,i):
  o=int(self.origins[i]);x=self.features[o-self.c["lookback"]+1:o+1,self.columns]
  return torch.from_numpy(x.copy()),torch.from_numpy(self.y[i]),torch.from_numpy(self.step[i]),torch.tensor(self.first[i]),torch.tensor(self.cause[i])

class ModernTCNBackbone(nn.Module):
 def __init__(self,n,c):
  super().__init__();m=c["model"];ch=m["channels"];layers=[nn.Conv1d(n,ch,1),nn.GELU()]
  for _ in range(m["layers"]):layers += [nn.Conv1d(ch,ch,m["kernel_size"],padding=m["kernel_size"]//2,groups=ch),nn.Conv1d(ch,ch,1),nn.GELU()]
  self.net=nn.Sequential(*layers);self.output_dim=ch*c["lookback"]
 def forward(self,x):return self.net(x.transpose(1,2)).flatten(1)

class JointModel(nn.Module):
 def __init__(self,n,c,kind):
  super().__init__();self.kind=kind;self.backbone=ModernTCNBackbone(n,c);d=self.backbone.output_dim;h=c["horizon"];self.power_head=nn.Linear(d,h);self.event_head=nn.Linear(d,h if kind=="STEP_MULTITASK" else h*3);self.h=h
 def forward(self,x):
  z=self.backbone(x);power=self.power_head(z);event=self.event_head(z);return power,event if self.kind=="STEP_MULTITASK" else event.reshape(-1,self.h,3)

def hazard_nll(logits,first,cause):
 logp=F.log_softmax(logits,dim=-1);none_prefix=logp[:,:,0].cumsum(dim=1);rows=torch.arange(len(first),device=logits.device);event=first>=0;k=first.clamp_min(0);before=torch.where(k>0,none_prefix[rows,(k-1).clamp_min(0)],torch.zeros_like(k,dtype=logp.dtype));event_log=logp[rows,k,cause.clamp_min(0)];event_loss=-(before+event_log);no_event_loss=-none_prefix[:,-1]
 return torch.where(event,event_loss,no_event_loss).mean()

def loaders(d,c):
 return {s:DataLoader(d[s],batch_size=c["training"]["batch_size"],shuffle=s=="train",num_workers=c["training"]["num_workers"],pin_memory=torch.cuda.is_available()) for s in d}

def train_model(model,train_loader,validation_loader,c,device,run_dir,target_scale,pos_weight):
 """No Test loader; checkpointing uses the candidate's Validation composite objective only."""
 t=c["training"];opt=torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"]);pw=torch.tensor(pos_weight,device=device);best=math.inf;stale=0;best_epoch=0;started=time.perf_counter();times=[];log=run_dir/"epochs.jsonl";log.write_text("",encoding="utf-8");finite=True
 for epoch in range(1,t["max_epochs"]+1):
  tick=time.perf_counter();model.train();parts=[]
  for x,y,step,first,cause in train_loader:
   x,y,step,first,cause=x.to(device),y.to(device),step.to(device),first.to(device),cause.to(device);opt.zero_grad(set_to_none=True);power,event=model(x);lp=F.mse_loss(power,y);le=F.binary_cross_entropy_with_logits(event,step,pos_weight=pw) if model.kind=="STEP_MULTITASK" else hazard_nll(event,first,cause);lam=t["lambda_step"] if model.kind=="STEP_MULTITASK" else t["lambda_hazard"];loss=lp+lam*le
   if not torch.isfinite(loss):finite=False;raise FloatingPointError("non-finite loss")
   loss.backward()
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):finite=False;raise FloatingPointError("non-finite gradient")
   torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip_norm"]);opt.step();parts.append((float(lp.detach().cpu()),float(le.detach().cpu()),float(loss.detach().cpu())))
  model.eval();vp=[];ve=[]
  with torch.no_grad():
   for x,y,step,first,cause in validation_loader:
    x,y,step,first,cause=x.to(device),y.to(device),step.to(device),first.to(device),cause.to(device);power,event=model(x);vp.append(float(F.mse_loss(power,y).cpu()));ve.append(float((F.binary_cross_entropy_with_logits(event,step,pos_weight=pw) if model.kind=="STEP_MULTITASK" else hazard_nll(event,first,cause)).cpu()))
  vpower,vevent=float(np.mean(vp)),float(np.mean(ve));lam=t["lambda_step"] if model.kind=="STEP_MULTITASK" else t["lambda_hazard"];objective=vpower+lam*vevent;elapsed=time.perf_counter()-tick;times.append(elapsed);record={"epoch":epoch,"train_power_loss":float(np.mean(parts,axis=0)[0]),"train_event_loss":float(np.mean(parts,axis=0)[1]),"train_objective":float(np.mean(parts,axis=0)[2]),"validation_power_loss":vpower,"validation_event_loss":vevent,"validation_objective":objective,"seconds":elapsed};
  with log.open("a",encoding="utf-8") as f:f.write(json.dumps(record)+"\n")
  torch.save({"epoch":epoch,"state_dict":model.state_dict(),"optimizer":opt.state_dict(),"validation_objective":objective},run_dir/"last.pt")
  if objective<best-t["min_delta"]:best,best_epoch,stale=objective,epoch,0;torch.save({"epoch":epoch,"state_dict":copy.deepcopy(model.state_dict()),"validation_objective":objective},run_dir/"best_validation.pt")
  else:
   stale+=1
   if stale>=t["patience"]:break
 reason="max_epochs" if epoch==t["max_epochs"] else "early_stopping"
 return {"actual_epochs":epoch,"best_epoch":best_epoch,"best_validation_objective":best,"stop_reason":reason,"numerically_finite":finite,"training_seconds":time.perf_counter()-started,"mean_epoch_seconds":float(np.mean(times))}

def predict(model,loader,device,center,scale):
 model.eval();powers=[];events=[];start=time.perf_counter()
 with torch.no_grad():
  for x,*_ in loader:
   p,e=model(x.to(device));powers.append((p.cpu().numpy()*scale+center).astype(np.float32));events.append(e.cpu().numpy())
 elapsed=time.perf_counter()-start;return np.concatenate(powers),np.concatenate(events),elapsed

def binary(y,score,threshold=.5):
 y=y.astype(int);pred=score>=threshold;pr,re,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0);unique=np.unique(y)
 return {"auroc":roc_auc_score(y,score) if len(unique)==2 else math.nan,"auprc":average_precision_score(y,score) if y.any() else math.nan,"brier":float(np.mean((score-y)**2)),"f1":f1,"precision":pr,"recall":re}

def power_metrics(y,p,mask):
 yy,pp=y[mask],p[mask];return {"rmse_kw":float(np.sqrt(np.mean((pp-yy)**2))),"mae_kw":float(np.mean(np.abs(pp-yy)))}

def event_predictions(kind,event,power_pred,prior_state,origin_power):
 if kind=="STEP_MULTITASK":
  prob=1/(1+np.exp(-event));window=1-np.prod(1-prob,axis=1);binary_step=prob>=.5;pred_first=np.full(len(prob),-1);pred_cause=np.zeros(len(prob),int)
  for i in range(len(prob)):
   previous=prior_state[i]
   for k in range(12):
    if binary_step[i,k] and not previous:pred_first[i]=k;break
    previous=binary_step[i,k]
   if pred_first[i]>=0:
    k=int(pred_first[i]);before=power_pred[i,k-1] if k else origin_power[i];delta=power_pred[i,k]-before;pred_cause[i]=1 if delta>0 else 2
  return prob,window,pred_first,pred_cause,None
 logp=event-event.max(axis=-1,keepdims=True);conditional=np.exp(logp);conditional/=conditional.sum(axis=-1,keepdims=True);survival=np.ones((len(event),13));event_prob=np.zeros((len(event),12,2))
 for k in range(12):event_prob[:,k,:]=survival[:,k,None]*conditional[:,k,1:];survival[:,k+1]=survival[:,k]*conditional[:,k,0]
 window=1-survival[:,-1];pred_event=window>=.5;pred_first=np.where(pred_event,event_prob.sum(axis=2).argmax(axis=1),-1);pred_cause=np.zeros(len(event),int)
 for i,k in enumerate(pred_first):
  if k>=0:pred_cause[i]=1+int(event_prob[i,k,1]>event_prob[i,k,0])
 return None,window,pred_first,pred_cause,(conditional,event_prob,survival)

def scopes(times_ns,origins,y,onset):
 ot=pd.to_datetime(times_ns[origins]);hours=np.column_stack([(ot+pd.Timedelta(minutes=5*(k+1))).hour for k in range(12)]);transition=((hours>=5)&(hours<9))|((hours>=16)&(hours<20));day=y>.063
 onset_day=np.asarray([bool(day[i,k]) if k>=0 else bool(day[i].any()) for i,k in enumerate(np.where(onset.any(1),onset.argmax(1),-1))]);return {"full_timeline":np.ones(len(y),bool),"daylight":day.any(1),"sunrise_sunset_transition":transition.any(1)},day,transition,onset_day

def append_power(rows,model,seed,y,p,step,onset,day,transition,info,source_path):
 for h in PREFIXES:
  mask=np.ones(y[:,:h].shape,bool);m=power_metrics(y[:,:h],p[:,:h],mask);rows.append({"section":"power","model":model,"seed":seed,"scope":"full_timeline","horizon":h,**m,"diff_mae_kw":float(np.mean(np.abs(np.diff(p[:,:h],axis=1)-np.diff(y[:,:h],axis=1)))) if h>1 else math.nan,**info,"artifact_source":source_path})
 for scope,mask in (("daylight",day),("ramp_step",step),("onset_near",np.broadcast_to(onset.any(1)[:,None],y.shape))):rows.append({"section":"power","model":model,"seed":seed,"scope":scope,"horizon":12,**power_metrics(y,p,mask),"diff_mae_kw":float(np.mean(np.abs(np.diff(p,axis=1)-np.diff(y,axis=1)))),**info,"artifact_source":source_path})

def append_events(rows,kind,seed,y,delta,step,onset,first,cause,prior,origin_power,power_pred,event,window_masks,day,transition,info):
 step_prob,window,pred_first,pred_cause,hazard=event_predictions(kind,event,power_pred,prior,origin_power)
 if step_prob is not None:
  for scope,mask in (("full_timeline",np.ones(step.shape,bool)),("daylight",day),("sunrise_sunset_transition",transition)):
   m=binary(step[mask],step_prob[mask]);rows.append({"section":"ramp_step","model":kind,"seed":seed,"scope":scope,"horizon":"all12","n":int(mask.sum()),"positive_count":int(step[mask].sum()),"prevalence":float(step[mask].mean()),**m,**info})
  for h in range(12):m=binary(step[:,h],step_prob[:,h]);rows.append({"section":"ramp_step","model":kind,"seed":seed,"scope":"full_timeline","horizon":h+1,"n":len(step),"positive_count":int(step[:,h].sum()),"prevalence":float(step[:,h].mean()),**m,**info})
 true_event=first>=0
 for scope,mask in window_masks.items():
  b=binary(true_event[mask],window[mask]);predicted_event=pred_first>=0
  if kind=="STEP_MULTITASK":
   pr,re,f1,_=precision_recall_fscore_support(true_event[mask],predicted_event[mask],average="binary",zero_division=0);b.update({"precision":pr,"recall":re,"f1":f1})
  both=mask&true_event&predicted_event;miss=mask&true_event&~predicted_event;time_err=np.abs(pred_first[both]-first[both]);up=mask&(cause==1);down=mask&(cause==2);correct_dir=predicted_event&(pred_cause==cause)
  rows.append({"section":"first_onset","model":kind,"seed":seed,"scope":scope,"horizon":12,"n":int(mask.sum()),"positive_count":int((mask&true_event).sum()),"prevalence":float(true_event[mask].mean()),**b,"miss_rate":float(miss.sum()/max(1,(mask&true_event).sum())),"onset_time_mae_steps":float(time_err.mean()) if len(time_err) else math.nan,"onset_time_mae_minutes":float(time_err.mean()*5) if len(time_err) else math.nan,"onset_time_n":int(len(time_err)),"up_count":int(up.sum()),"down_count":int(down.sum()),"up_recall":float((correct_dir&up).sum()/max(1,up.sum())),"down_recall":float((correct_dir&down).sum()/max(1,down.sum())),"direction_accuracy":float(correct_dir[both].mean()) if both.any() else math.nan,**info})
 for h in range(12):
  truth=first==h;identified=(pred_first>=0)&true_event;rows.append({"section":"onset_lead","model":kind,"seed":seed,"scope":"full_timeline","horizon":h+1,"n":int(truth.sum()),"positive_count":int(truth.sum()),"prevalence":float(truth.mean()),"identification_rate":float((identified&truth).sum()/max(1,truth.sum())),"up_count":int((truth&(cause==1)).sum()),"down_count":int((truth&(cause==2)).sum()),**info})
 for b in range(10):
  lo,hi=b/10,(b+1)/10;mask=(window>=lo)&(window<(hi if b<9 else hi+1e-12));rows.append({"section":"reliability","model":kind,"seed":seed,"scope":"full_timeline","horizon":12,"probability_bin":f"[{lo:.1f},{hi:.1f}{')' if b<9 else ']'}","n":int(mask.sum()),"mean_predicted_probability":float(window[mask].mean()) if mask.any() else math.nan,"observed_event_rate":float(true_event[mask].mean()) if mask.any() else math.nan,**info})
 return step_prob,window,pred_first,pred_cause,hazard

def write(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with METRICS.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def recovered_info(existing,kind,seed,run):
 if len(existing):
  match=existing[(existing.model==kind)&(existing.seed==seed)]
  if len(match):return {k:match.iloc[0][k] for k in ("actual_epochs","best_epoch","best_validation_objective","stop_reason","numerically_finite","training_seconds","mean_epoch_seconds","backbone_parameters","power_head_parameters","event_head_parameters","total_parameters","inference_ms_per_sample","threshold_kw","step_pos_weight")}
 records=[json.loads(x) for x in (run/"epochs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()];best=torch.load(run/"best_validation.pt",map_location="cpu",weights_only=True);return {"actual_epochs":len(records),"best_epoch":best["epoch"],"best_validation_objective":best["validation_objective"],"stop_reason":"max_epochs" if len(records)==cfg()["training"]["max_epochs"] else "early_stopping","numerically_finite":True,"training_seconds":sum(x["seconds"] for x in records),"mean_epoch_seconds":float(np.mean([x["seconds"] for x in records])),"inference_ms_per_sample":math.nan}

def main():
 c=cfg();d=np.load((ROOT/c["prepared_data"]).resolve(),allow_pickle=False);features,power,columns=d["scaled_features"],d["power"],d["base_cols"];center,scale,threshold=float(d["target_center"]),float(d["target_scale"]),float(d["ramp_threshold"]);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");existing=pd.read_csv(METRICS) if METRICS.exists() else pd.DataFrame()
 all_labels={s:event_labels(power,d[f"{s}_origins"],threshold) for s in ("train","validation","test")};train_step=all_labels["train"][2];pos_weight=float((train_step.size-train_step.sum())/train_step.sum());rows=[];RESULTS.mkdir(parents=True,exist_ok=True)
 for split,(sy,sd,ss,so,sp,sf,sc) in all_labels.items():
  n=len(sy);event=sf>=0
  rows += [{"section":"prevalence","model":"LABELS","seed":"","scope":split,"horizon":12,"n":ss.size,"positive_count":int(ss.sum()),"prevalence":float(ss.mean()),"event_type":"ramp_step"},{"section":"prevalence","model":"LABELS","seed":"","scope":split,"horizon":12,"n":n,"positive_count":int(event.sum()),"prevalence":float(event.mean()),"event_type":"first_onset"},{"section":"prevalence","model":"LABELS","seed":"","scope":split,"horizon":12,"n":n,"positive_count":int((sc==1).sum()),"prevalence":float((sc==1).mean()),"event_type":"upward_onset"},{"section":"prevalence","model":"LABELS","seed":"","scope":split,"horizon":12,"n":n,"positive_count":int((sc==2).sum()),"prevalence":float((sc==2).mean()),"event_type":"downward_onset"}]
 test_o=d["test_origins"];y,delta,step,onset,prior,first,cause=all_labels["test"];window_masks,day,transition,_=scopes(d["times"],test_o,y,onset)
 base_info={"backbone_parameters":19136,"power_head_parameters":55308,"event_head_parameters":0,"total_parameters":74444,"actual_epochs":"reused","best_epoch":"reused","stop_reason":"reused","numerically_finite":True,"training_seconds":0,"mean_epoch_seconds":0,"inference_ms_per_sample":math.nan,"threshold_kw":threshold,"step_pos_weight":pos_weight}
 for s in c["seeds"]:
  path=(ROOT/c["trajectory_only_results"]).resolve()/str(s)/"test_predictions.npz";a=np.load(path);assert np.array_equal(a["labels"],y) and np.array_equal(a["forecast_origin_timestamp_ns"],d["times"][test_o]);append_power(rows,"TRAJECTORY_ONLY",s,y,a["predictions"],step,onset,day,transition,base_info,str(path))
 for kind in c["models"]:
  for s in c["seeds"]:
   seed_all(s);run=RESULTS/kind/str(s);run.mkdir(parents=True,exist_ok=True);artifact=run/"test_predictions.npz";model=JointModel(len(columns),c,kind).to(device)
   if artifact.exists():
    a=np.load(artifact);pred,event=a["power_predictions"],a["event_output"];run_info=recovered_info(existing,kind,s,run)
   else:
    sets={sp:JointDataset(features,power,d[f"{sp}_origins"],columns,c,center,scale,threshold) for sp in ("train","validation","test")};load=loaders(sets,c);info=train_model(model,load["train"],load["validation"],c,device,run,scale,pos_weight);ck=torch.load(run/"best_validation.pt",map_location=device,weights_only=True);model.load_state_dict(ck["state_dict"]);pred,event,infer=predict(model,load["test"],device,center,scale);run_info={**info,"inference_ms_per_sample":infer/len(y)*1000}
   backbone=sum(p.numel() for p in model.backbone.parameters());power_params=sum(p.numel() for p in model.power_head.parameters());event_params=sum(p.numel() for p in model.event_head.parameters());run_info.update({"backbone_parameters":backbone,"power_head_parameters":power_params,"event_head_parameters":event_params,"total_parameters":sum(p.numel() for p in model.parameters()),"threshold_kw":threshold,"step_pos_weight":pos_weight});append_power(rows,kind,s,y,pred,step,onset,day,transition,run_info,str(artifact));sp,win,pf,pc,haz=append_events(rows,kind,s,y,delta,step,onset,first,cause,prior,power[test_o],pred,event,window_masks,day,transition,run_info)
   if not artifact.exists():np.savez_compressed(artifact,power_predictions=pred,event_output=event,step_probability=sp if sp is not None else np.array([]),window_onset_probability=win,predicted_first=pf,predicted_cause=pc,labels=y,forecast_origin_timestamp_ns=d["times"][test_o],step_mask=step,onset_mask=onset,first_onset=first,cause=cause)
 write(rows)
if __name__=="__main__":main()

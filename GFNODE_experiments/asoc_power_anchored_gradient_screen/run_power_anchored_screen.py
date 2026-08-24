"""Power-anchored asymmetric gradient projection for the existing onset-hazard model."""
from __future__ import annotations
import copy,csv,json,math,random,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;RESULTS=ROOT/"results";METRICS=ROOT/"metrics_per_seed.csv";JOINT_DIR=ROOT.parent/"asoc_joint_trajectory_onset_screen";sys.path.insert(0,str(JOINT_DIR))
import run_joint_onset_screen as joint

def cfg():return json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def project_event_gradient(g_power,g_event,epsilon=1e-12):
 dot=sum((a*b).sum() for a,b in zip(g_power,g_event));power_sq=sum((a*a).sum() for a in g_power);event_sq=sum((b*b).sum() for b in g_event);conflict=dot<0;coefficient=torch.where(conflict,dot/(power_sq+epsilon),torch.zeros_like(dot));projected=tuple(b-coefficient*a for a,b in zip(g_power,g_event));cosine=dot/(torch.sqrt(power_sq*event_sq)+epsilon);correction=torch.sqrt(sum(((p-b)**2).sum() for p,b in zip(projected,g_event)));ratio=correction/(torch.sqrt(event_sq)+epsilon)
 return projected,{"dot":dot,"cosine":cosine,"power_norm":torch.sqrt(power_sq),"event_norm":torch.sqrt(event_sq),"correction_ratio":ratio,"conflict":conflict}

def anchored_gradients(model,power_loss,event_loss,lambda_hazard,epsilon):
 shared=list(model.backbone.parameters());power_head=list(model.power_head.parameters());hazard_head=list(model.event_head.parameters());gp=torch.autograd.grad(power_loss,shared+power_head,retain_graph=True);ge=torch.autograd.grad(lambda_hazard*event_loss,shared+hazard_head);gp_shared,gp_head=gp[:len(shared)],gp[len(shared):];ge_shared,ge_head=ge[:len(shared)],ge[len(shared):];projected,stats=project_event_gradient(gp_shared,ge_shared,epsilon)
 for p,gp_value,ge_value in zip(shared,gp_shared,projected):p.grad=gp_value+ge_value
 for p,g in zip(power_head,gp_head):p.grad=g
 for p,g in zip(hazard_head,ge_head):p.grad=g
 return stats,gp_shared,ge_shared,projected

def summarize_gradient(batch_stats):
 cosine=torch.stack([x["cosine"] for x in batch_stats]).cpu().numpy();negative=cosine[cosine<0];power_norm=torch.stack([x["power_norm"] for x in batch_stats]).mean().cpu();event_norm=torch.stack([x["event_norm"] for x in batch_stats]).mean().cpu();correction=torch.stack([x["correction_ratio"] for x in batch_stats]).mean().cpu()
 return {"effective_batches":len(batch_stats),"conflict_batches":int((cosine<0).sum()),"conflict_rate":float((cosine<0).mean()),"raw_cosine_mean":float(cosine.mean()),"raw_cosine_median":float(np.median(cosine)),"raw_cosine_sd":float(cosine.std(ddof=1)) if len(cosine)>1 else 0.0,"negative_cosine_q25":float(np.quantile(negative,.25)) if len(negative) else math.nan,"negative_cosine_median":float(np.median(negative)) if len(negative) else math.nan,"power_grad_norm_mean":float(power_norm),"event_grad_norm_mean":float(event_norm),"projection_correction_ratio_mean":float(correction)}

def train_power_anchored(model,train_loader,validation_loader,c,device,run_dir):
 """No Test loader. Best checkpoint uses the unchanged Validation composite objective."""
 t=c["training"];opt=torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"]);best=math.inf;best_epoch=0;stale=0;epoch_times=[];started=time.perf_counter();log=run_dir/"epochs.jsonl";log.write_text("",encoding="utf-8");finite=True
 for epoch in range(1,t["max_epochs"]+1):
  tick=time.perf_counter();model.train();losses=[];gradient_stats=[]
  for x,y,step,first,cause in train_loader:
   x,y,first,cause=x.to(device),y.to(device),first.to(device),cause.to(device);opt.zero_grad(set_to_none=True);power,hazard=model(x);lp=F.mse_loss(power,y);le=joint.hazard_nll(hazard,first,cause);total=lp+t["lambda_hazard"]*le
   if not torch.isfinite(total):finite=False;raise FloatingPointError("non-finite loss")
   stats,*_=anchored_gradients(model,lp,le,t["lambda_hazard"],t["projection_epsilon"])
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):finite=False;raise FloatingPointError("non-finite gradient")
   torch.nn.utils.clip_grad_norm_(model.parameters(),t["gradient_clip_norm"]);opt.step();losses.append((float(lp.detach().cpu()),float(le.detach().cpu()),float(total.detach().cpu())));gradient_stats.append({k:v.detach() for k,v in stats.items()})
  model.eval();vp=[];ve=[]
  with torch.no_grad():
   for x,y,step,first,cause in validation_loader:
    x,y,first,cause=x.to(device),y.to(device),first.to(device),cause.to(device);power,hazard=model(x);vp.append(float(F.mse_loss(power,y).cpu()));ve.append(float(joint.hazard_nll(hazard,first,cause).cpu()))
  vpower,vevent=float(np.mean(vp)),float(np.mean(ve));objective=vpower+t["lambda_hazard"]*vevent;elapsed=time.perf_counter()-tick;epoch_times.append(elapsed);means=np.mean(losses,axis=0);record={"epoch":epoch,"train_power_loss":float(means[0]),"train_first_event_loss":float(means[1]),"train_composite_loss":float(means[2]),"validation_power_loss":vpower,"validation_first_event_loss":vevent,"validation_objective":objective,"seconds":elapsed,**summarize_gradient(gradient_stats)}
  with log.open("a",encoding="utf-8") as f:f.write(json.dumps(record)+"\n")
  torch.save({"epoch":epoch,"state_dict":model.state_dict(),"optimizer":opt.state_dict(),"validation_objective":objective},run_dir/"last.pt")
  if objective<best-t["min_delta"]:best,best_epoch,stale=objective,epoch,0;torch.save({"epoch":epoch,"state_dict":copy.deepcopy(model.state_dict()),"validation_objective":objective},run_dir/"best_validation.pt")
  else:
   stale+=1
   if stale>=t["patience"]:break
 return {"actual_epochs":epoch,"best_epoch":best_epoch,"best_validation_objective":best,"stop_reason":"max_epochs" if epoch==t["max_epochs"] else "early_stopping","numerically_finite":finite,"training_seconds":time.perf_counter()-started,"mean_epoch_seconds":float(np.mean(epoch_times))}

def load_prior_info(prior_metrics,model,seed):
 x=prior_metrics[(prior_metrics.model==model)&(prior_metrics.seed==seed)&(prior_metrics.section=="power")].iloc[0];keys=("actual_epochs","best_epoch","best_validation_objective","stop_reason","numerically_finite","training_seconds","mean_epoch_seconds","backbone_parameters","power_head_parameters","event_head_parameters","total_parameters","inference_ms_per_sample","threshold_kw","step_pos_weight");return {k:x[k] for k in keys}

def write(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with METRICS.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def evaluate_all(c,d,anchored_info):
 rows=[];power=d["power"];test_o=d["test_origins"];threshold=float(d["ramp_threshold"]);y,delta,step,onset,prior,first,cause=joint.event_labels(power,test_o,threshold);window_masks,day,transition,_=joint.scopes(d["times"],test_o,y,onset);prior_metrics=pd.read_csv((ROOT/c["joint_metrics"]).resolve());base={"actual_epochs":"reused","best_epoch":"reused","best_validation_objective":"reused","stop_reason":"reused","numerically_finite":True,"training_seconds":0,"mean_epoch_seconds":0,"backbone_parameters":19136,"power_head_parameters":55308,"event_head_parameters":0,"total_parameters":74444,"inference_ms_per_sample":math.nan,"threshold_kw":threshold,"step_pos_weight":float((joint.event_labels(power,d["train_origins"],threshold)[2].size-joint.event_labels(power,d["train_origins"],threshold)[2].sum())/joint.event_labels(power,d["train_origins"],threshold)[2].sum())}
 for seed in c["seeds"]:
  path=(ROOT/c["trajectory_only_results"]).resolve()/str(seed)/"test_predictions.npz";a=np.load(path);assert np.array_equal(a["labels"],y) and np.array_equal(a["forecast_origin_timestamp_ns"],d["times"][test_o]);joint.append_power(rows,"TRAJECTORY_ONLY",seed,y,a["predictions"],step,onset,day,transition,base,str(path))
 for model,folder,kind in (("STEP_MULTITASK","STEP_MULTITASK","STEP_MULTITASK"),("STANDARD_ONSET_HAZARD","ONSET_HAZARD","ONSET_HAZARD")):
  for seed in c["seeds"]:
   path=(ROOT/c["joint_results"]).resolve()/folder/str(seed)/"test_predictions.npz";a=np.load(path);assert np.array_equal(a["labels"],y) and np.array_equal(a["forecast_origin_timestamp_ns"],d["times"][test_o]);info=load_prior_info(prior_metrics,"ONSET_HAZARD" if model.startswith("STANDARD") else model,seed);joint.append_power(rows,model,seed,y,a["power_predictions"],step,onset,day,transition,info,str(path));before=len(rows);joint.append_events(rows,kind,seed,y,delta,step,onset,first,cause,prior,power[test_o],a["power_predictions"],a["event_output"],window_masks,day,transition,info)
   for row in rows[before:]:row["model"]=model
 for seed in c["seeds"]:
  path=RESULTS/"POWER_ANCHORED_HAZARD"/str(seed)/"test_predictions.npz";a=np.load(path);assert np.array_equal(a["labels"],y) and np.array_equal(a["forecast_origin_timestamp_ns"],d["times"][test_o]);info=anchored_info[seed];joint.append_power(rows,"POWER_ANCHORED_HAZARD",seed,y,a["power_predictions"],step,onset,day,transition,info,str(path));joint.append_events(rows,"POWER_ANCHORED_HAZARD",seed,y,delta,step,onset,first,cause,prior,power[test_o],a["power_predictions"],a["event_output"],window_masks,day,transition,info)
 # Explicit scope audit rows.
 true_event=first>=0
 for name,mask in window_masks.items():rows.append({"section":"scope_audit","model":"LABELS","seed":"","scope":name,"horizon":12,"n":int(mask.sum()),"positive_count":int((mask&true_event).sum()),"prevalence":float(true_event[mask].mean()),"mask_equal_to_full":bool(np.array_equal(mask,window_masks["full_timeline"])),"daylight_definition":"any future true target power > 0.063 kW" if name=="daylight" else "fixed ACST clock or full"})
 write(rows)

def main():
 c=cfg();d=np.load((ROOT/c["prepared_data"]).resolve(),allow_pickle=False);features,power,columns=d["scaled_features"],d["power"],d["base_cols"];center,scale,threshold=float(d["target_center"]),float(d["target_scale"]),float(d["ramp_threshold"]);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");RESULTS.mkdir(parents=True,exist_ok=True);anchored_info={}
 for seed in c["seeds"]:
  seed_all(seed);run=RESULTS/"POWER_ANCHORED_HAZARD"/str(seed);run.mkdir(parents=True,exist_ok=True);sets={sp:joint.JointDataset(features,power,d[f"{sp}_origins"],columns,c,center,scale,threshold) for sp in ("train","validation","test")};load=joint.loaders(sets,c);model=joint.JointModel(len(columns),c,"ONSET_HAZARD").to(device);info=train_power_anchored(model,load["train"],load["validation"],c,device,run);ck=torch.load(run/"best_validation.pt",map_location=device,weights_only=True);model.load_state_dict(ck["state_dict"]);pred,event,infer=joint.predict(model,load["test"],device,center,scale);y,delta,step,onset,prior,first,cause=joint.event_labels(power,d["test_origins"],threshold);info.update({"backbone_parameters":sum(p.numel() for p in model.backbone.parameters()),"power_head_parameters":sum(p.numel() for p in model.power_head.parameters()),"event_head_parameters":sum(p.numel() for p in model.event_head.parameters()),"total_parameters":sum(p.numel() for p in model.parameters()),"inference_ms_per_sample":infer/len(y)*1000,"threshold_kw":threshold,"step_pos_weight":math.nan});anchored_info[seed]=info;np.savez_compressed(run/"test_predictions.npz",power_predictions=pred,event_output=event,labels=y,forecast_origin_timestamp_ns=d["times"][d["test_origins"]],step_mask=step,onset_mask=onset,first_onset=first,cause=cause)
 evaluate_all(c,d,anchored_info)
if __name__=="__main__":main()

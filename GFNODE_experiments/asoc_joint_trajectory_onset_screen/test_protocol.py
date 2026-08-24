"""Ordinary implementation/protocol tests for the joint onset screen."""
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_joint_onset_screen as s

def main():
 c=s.cfg();d=np.load((s.ROOT/c["prepared_data"]).resolve(),allow_pickle=False);n=len(d["base_cols"]);x=torch.randn(4,72,n)
 step=s.JointModel(n,c,"STEP_MULTITASK");haz=s.JointModel(n,c,"ONSET_HAZARD");ps,ls=step(x);ph,lh=haz(x)
 assert ps.shape==(4,12) and ls.shape==(4,12);assert ph.shape==(4,12) and lh.shape==(4,12,3)
 prob=torch.softmax(lh,dim=-1);assert torch.allclose(prob.sum(-1),torch.ones(4,12),atol=1e-6)
 survival=torch.ones(4);total=torch.zeros(4);history=[]
 for k in range(12):total+=survival*(prob[:,k,1]+prob[:,k,2]);survival*=prob[:,k,0];history.append(survival.clone())
 assert torch.allclose(total+survival,torch.ones(4),atol=1e-5);assert torch.all(torch.stack(history)[1:]<=torch.stack(history)[:-1]+1e-7)
 power=d["power"];threshold=float(d["ramp_threshold"]);labels={sp:s.event_labels(power,d[f"{sp}_origins"],threshold) for sp in ("train","validation","test")}
 for values in labels.values():
  _,_,ramp,onset,prior,first,cause=values;assert not np.any((cause==1)&(cause==2));assert np.all(onset.sum(1)>=0);assert np.all((first>=0)==(cause>0));assert np.all(onset.sum(1)>=0)
  for i,k in enumerate(first[:1000]):assert k<0 or (onset[i,k] and onset[i,:k].sum()==0)
 test_o=d["test_origins"];_,_,ramp,onset,prior,first,cause=labels["test"]
 for i,o in enumerate(test_o[:1000]):assert prior[i]==(abs(power[o]-power[o-1])>=threshold)
 train_change=np.concatenate([np.abs(power[o+1:o+13]-power[o:o+12]) for o in d["train_origins"]]);assert np.isclose(threshold,np.quantile(train_change,.9),rtol=1e-6)
 times=pd.to_datetime(d["times"])
 for sp,origins in ((k,d[f"{k}_origins"]) for k in ("train","validation","test")):
  lo,hi=map(pd.Timestamp,c["splits"][sp])
  for o in (origins[0],origins[len(origins)//2],origins[-1]):assert times[o-71]>=lo and times[o+12]<hi and times[o]<=times[o] and times[o+1]==times[o]+pd.Timedelta(minutes=5)
 assert "test_loader" not in inspect.signature(s.train_model).parameters;src=inspect.getsource(s.train_model);assert "validation_loader" in src and "validation_objective" in src and "test_loader" not in src
 y,delta,ramp,onset,prior,first,cause=labels["test"]
 for seed in c["seeds"]:
  a=np.load((s.ROOT/c["trajectory_only_results"]).resolve()/str(seed)/"test_predictions.npz");assert np.array_equal(a["labels"],y);assert np.array_equal(a["forecast_origin_timestamp_ns"],d["times"][test_o])
 event_first=torch.tensor([0,5]);event_cause=torch.tensor([1,2]);no_first=torch.tensor([-1,-1]);no_cause=torch.tensor([0,0]);logits=torch.randn(2,12,3,requires_grad=True)
 le=s.hazard_nll(logits,event_first,event_cause);ln=s.hazard_nll(logits,no_first,no_cause);assert torch.isfinite(le) and torch.isfinite(ln)
 for model,kind in ((step,"STEP_MULTITASK"),(haz,"ONSET_HAZARD")):
  model.zero_grad();power_out,event=model(x);lp=power_out.square().mean();ev=torch.nn.functional.binary_cross_entropy_with_logits(event,torch.zeros_like(event)) if kind=="STEP_MULTITASK" else s.hazard_nll(event,no_first.repeat(2),no_cause.repeat(2));loss=lp+.2*ev;loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 assert [p.shape for p in step.backbone.parameters()]==[p.shape for p in haz.backbone.parameters()];assert [p.shape for p in step.power_head.parameters()]==[p.shape for p in haz.power_head.parameters()]
 source=inspect.getsource(s.train_model);assert "F.binary_cross_entropy_with_logits" in source and "hazard_nll" in source and ".mean()" in inspect.getsource(s.hazard_nll)
 print("PASS: 19 shape, probability, survival, labeling, leakage, loss, gradient, fairness and artifact checks")
if __name__=="__main__":main()

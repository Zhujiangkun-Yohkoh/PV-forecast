"""Ordinary tests for asymmetric power-anchored gradient projection."""
import inspect
import numpy as np
import pandas as pd
import torch
import run_power_anchored_screen as s

def main():
 c=s.cfg();d=np.load((s.ROOT/c["prepared_data"]).resolve(),allow_pickle=False);n=len(d["base_cols"]);s.seed_all(42);standard=s.joint.JointModel(n,c,"ONSET_HAZARD");s.seed_all(42);anchored=s.joint.JointModel(n,c,"ONSET_HAZARD")
 assert sum(p.numel() for p in standard.parameters())==sum(p.numel() for p in anchored.parameters())==240368
 assert all(torch.equal(a,b) for a,b in zip(standard.state_dict().values(),anchored.state_dict().values()))
 x=torch.randn(4,72,n);power,hazard=anchored(x);assert power.shape==(4,12) and hazard.shape==(4,12,3)
 gp=(torch.tensor([1.,0.]),);ge=(torch.tensor([-1.,1.]),);proj,stats=s.project_event_gradient(gp,ge);assert (proj[0]*gp[0]).sum()>=-1e-7 and torch.equal(gp[0],torch.tensor([1.,0.]))
 ge2=(torch.tensor([1.,1.]),);proj2,stats2=s.project_event_gradient(gp,ge2);assert torch.equal(proj2[0],ge2[0])
 first=torch.tensor([0,5,-1,-1]);cause=torch.tensor([1,2,0,0]);lp=power.square().mean();le=s.joint.hazard_nll(hazard,first,cause);expected_power=torch.autograd.grad(lp,list(anchored.power_head.parameters()),retain_graph=True);expected_event=torch.autograd.grad(.2*le,list(anchored.event_head.parameters()),retain_graph=True);anchored.zero_grad();stats,gps,ges,projected=s.anchored_gradients(anchored,lp,le,.2,1e-12)
 assert all(torch.equal(a,b) for a,b in zip(gps,[g.clone() for g in gps]));assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in anchored.parameters())
 assert all(torch.allclose(p.grad,g) for p,g in zip(anchored.power_head.parameters(),expected_power))
 assert all(torch.allclose(p.grad,g) for p,g in zip(anchored.event_head.parameters(),expected_event))
 for p,gp0,ge0 in zip(anchored.backbone.parameters(),gps,projected):assert torch.allclose(p.grad,gp0+ge0)
 before=[p.grad.clone() for p in anchored.parameters()];anchored.zero_grad(set_to_none=True);assert all(p.grad is None for p in anchored.parameters());assert all(torch.isfinite(g).all() for g in before)
 logits=torch.randn(4,12,3,requires_grad=True);assert torch.isfinite(s.joint.hazard_nll(logits,torch.tensor([0,5,8,2]),torch.tensor([1,2,1,2])));assert torch.isfinite(s.joint.hazard_nll(logits,torch.full((4,),-1),torch.zeros(4,dtype=torch.long)))
 src=inspect.getsource(s.train_power_anchored);assert "test_loader" not in inspect.signature(s.train_power_anchored).parameters and "validation_loader" in src and "validation_objective" in src
 power_raw=d["power"];threshold=float(d["ramp_threshold"]);train_change=np.concatenate([np.abs(power_raw[o+1:o+13]-power_raw[o:o+12]) for o in d["train_origins"]]);assert np.isclose(threshold,np.quantile(train_change,.9),rtol=1e-6)
 times=pd.to_datetime(d["times"])
 for sp in ("train","validation","test"):
  origins=d[f"{sp}_origins"];lo,hi=map(pd.Timestamp,c["splits"][sp]);assert times[origins[0]-71]>=lo and times[origins[-1]+12]<hi and times[origins[0]]<times[origins[0]+1]
 test_o=d["test_origins"];y,*_=s.joint.event_labels(power_raw,test_o,threshold);reference_times=d["times"][test_o]
 files=[]
 for seed in c["seeds"]:
  files += [(s.ROOT/c["trajectory_only_results"]).resolve()/str(seed)/"test_predictions.npz",(s.ROOT/c["joint_results"]).resolve()/"STEP_MULTITASK"/str(seed)/"test_predictions.npz",(s.ROOT/c["joint_results"]).resolve()/"ONSET_HAZARD"/str(seed)/"test_predictions.npz"]
 for path in files:
  a=np.load(path);assert np.array_equal(a["labels"],y) and np.array_equal(a["forecast_origin_timestamp_ns"],reference_times)
 window_masks,day,transition,_=s.joint.scopes(d["times"],test_o,y,s.joint.event_labels(power_raw,test_o,threshold)[3]);assert not np.array_equal(window_masks["full_timeline"],window_masks["daylight"]);assert window_masks["full_timeline"].sum()==17401 and window_masks["daylight"].sum()==9655
 power2,hazard2=anchored(x);lp2=power2.square().mean();le2=s.joint.hazard_nll(hazard2,first,cause);total=lp2+.2*le2;anchored.zero_grad();stats,*_=s.anchored_gradients(anchored,lp2,le2,.2,1e-12);assert torch.isfinite(total) and all(p.grad is None or torch.isfinite(p.grad).all() for p in anchored.parameters())
 print("PASS: 19 initialization, projection, gradient-isolation, loss, leakage, artifact and scope checks")
if __name__=="__main__":main()

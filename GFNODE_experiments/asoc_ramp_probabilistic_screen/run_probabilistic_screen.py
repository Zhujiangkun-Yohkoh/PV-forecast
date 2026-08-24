"""Site 17 probabilistic trajectory screen with optional predicted-ramp width modulation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METRICS = ROOT / "metrics_per_seed.csv"
SOURCE_DIR = ROOT.parent / "asoc_multirate_information_screen"
sys.path.insert(0, str(SOURCE_DIR))
import run_information_screen as source  # noqa: E402

PREFIXES = (3, 6, 12)
SCOPES = ("full_test", "daylight", "ramp", "non_ramp")


def config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class ProbabilityDataset(Dataset):
    def __init__(self, features: np.ndarray, power: np.ndarray, origins: np.ndarray, columns: np.ndarray, cfg: dict, center: float, scale: float, threshold: float):
        self.features, self.power, self.origins, self.columns = features, power, origins, columns
        self.lookback, self.horizon, self.center, self.scale, self.threshold = cfg["lookback"], cfg["horizon"], center, scale, threshold
    def __len__(self): return len(self.origins)
    def __getitem__(self, i):
        o = int(self.origins[i]); x = self.features[o-self.lookback+1:o+1, self.columns]
        y_raw = self.power[o+1:o+self.horizon+1]; previous = self.power[o:o+self.horizon]
        ramp = (np.abs(y_raw-previous) >= self.threshold).astype(np.float32)
        y = ((y_raw-self.center)/self.scale).astype(np.float32)
        return torch.from_numpy(x.copy()), torch.from_numpy(y), torch.from_numpy(ramp)


class ModernTCNBackbone(nn.Module):
    """Same 64-channel/four-block encoder as commit 265cd618."""
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__(); m=cfg["model"]; ch=m["channels"]
        layers: list[nn.Module] = [nn.Conv1d(input_dim,ch,1),nn.GELU()]
        for _ in range(m["layers"]):
            layers += [nn.Conv1d(ch,ch,m["kernel_size"],padding=m["kernel_size"]//2,groups=ch),nn.Conv1d(ch,ch,1),nn.GELU()]
        self.net=nn.Sequential(*layers); self.output_dim=ch*cfg["lookback"]
    def forward(self,x): return self.net(x.transpose(1,2)).flatten(1)


class NCQModel(nn.Module):
    def __init__(self,input_dim:int,cfg:dict,ramp_aware:bool):
        super().__init__(); self.ramp_aware=ramp_aware; h=cfg["horizon"]; self.backbone=ModernTCNBackbone(input_dim,cfg); d=self.backbone.output_dim
        self.median_head=nn.Linear(d,h); self.lower_head=nn.Linear(d,h); self.upper_head=nn.Linear(d,h)
        if ramp_aware:
            self.ramp_head=nn.Linear(d,h); self.alpha_lower=nn.Parameter(torch.zeros(h)); self.alpha_upper=nn.Parameter(torch.zeros(h))
    def forward(self,x):
        """No label argument: interval modulation can only use predicted ramp probability."""
        z=self.backbone(x); q50=self.median_head(z); lower=F.softplus(self.lower_head(z)); upper=F.softplus(self.upper_head(z)); logits=None
        if self.ramp_aware:
            logits=self.ramp_head(z); probability=torch.sigmoid(logits)
            lower=lower*(1+F.softplus(self.alpha_lower)*probability); upper=upper*(1+F.softplus(self.alpha_upper)*probability)
        quantiles=torch.stack((q50-lower,q50,q50+upper),dim=-1)
        return quantiles,logits


def pinball_loss(prediction:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    q=torch.tensor((0.1,0.5,0.9),device=prediction.device,dtype=prediction.dtype); error=target.unsqueeze(-1)-prediction
    return torch.maximum(q*error,(q-1)*error).mean()


def train_model(model:nn.Module,train_loader:DataLoader,validation_loader:DataLoader,cfg:dict,device:torch.device,run_dir:Path,pos_weight:float)->dict:
    """Validation-pinball checkpointing only; deliberately no Test-loader parameter."""
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    best=math.inf; stale=0; best_epoch=0; epoch_times=[]; started=time.perf_counter(); log=run_dir/"epochs.jsonl"; log.write_text("",encoding="utf-8")
    positive_weight=torch.tensor(pos_weight,device=device)
    for epoch in range(1,cfg["training"]["max_epochs"]+1):
        tick=time.perf_counter(); model.train(); train_losses=[]
        for x,y,ramp in train_loader:
            x,y,ramp=x.to(device),y.to(device),ramp.to(device); opt.zero_grad(set_to_none=True); quantiles,logits=model(x); loss=pinball_loss(quantiles,y)
            if logits is not None: loss=loss+cfg["training"]["lambda_ramp"]*F.binary_cross_entropy_with_logits(logits,ramp,pos_weight=positive_weight)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); validation=[]
        with torch.no_grad():
            for x,y,_ in validation_loader: validation.append(float(pinball_loss(model(x.to(device))[0],y.to(device)).cpu()))
        val=float(np.mean(validation)); elapsed=time.perf_counter()-tick; epoch_times.append(elapsed)
        state={"epoch":epoch,"state_dict":model.state_dict(),"optimizer":opt.state_dict(),"validation_mean_pinball":val}; torch.save(state,run_dir/"last.pt")
        with log.open("a",encoding="utf-8") as f: f.write(json.dumps({"epoch":epoch,"training_loss":float(np.mean(train_losses)),"validation_mean_pinball":val,"seconds":elapsed})+"\n")
        if val<best-1e-8:
            best,best_epoch,stale=val,epoch,0; torch.save({"epoch":epoch,"state_dict":copy.deepcopy(model.state_dict()),"validation_mean_pinball":val},run_dir/"best_validation.pt")
        else:
            stale+=1
            if stale>=cfg["training"]["patience"]: break
    return {"actual_epochs":epoch,"best_epoch":best_epoch,"best_validation_mean_pinball":best,"training_seconds":time.perf_counter()-started,"mean_epoch_seconds":float(np.mean(epoch_times))}


def predict(model:nn.Module,loader:DataLoader,device:torch.device,center:float,scale:float):
    model.eval(); qs=[]; ps=[]; started=time.perf_counter()
    with torch.no_grad():
        for x,_,_ in loader:
            q,logits=model(x.to(device)); qs.append((q.cpu().numpy()*scale+center).astype(np.float32)); ps.append(np.full(q.shape[:2],np.nan,np.float32) if logits is None else torch.sigmoid(logits).cpu().numpy())
    elapsed=time.perf_counter()-started; return np.concatenate(qs),np.concatenate(ps),elapsed


def probability_metrics(y,q,mask)->dict:
    yy=y[mask]; qq=q[mask]; lower,median,upper=qq[:,0],qq[:,1],qq[:,2]
    errors=yy[:,None]-qq; levels=np.array([.1,.5,.9]); pin=float(np.maximum(levels*errors,(levels-1)*errors).mean())
    covered=(yy>=lower)&(yy<=upper); width=upper-lower; winkler=width+10*np.maximum(lower-yy,0)+10*np.maximum(yy-upper,0)
    mse=float(np.mean((median-yy)**2)); denom=float(np.sum((yy-yy.mean())**2))
    return {"mean_pinball":pin,"coverage_80":float(covered.mean()),"mean_width_kw":float(width.mean()),"winkler_score":float(winkler.mean()),"calibration_error":abs(float(covered.mean())-.8),
            "crossing_rate":float(np.mean((qq[:,0]>qq[:,1])|(qq[:,1]>qq[:,2]))),"q50_rmse_kw":math.sqrt(mse),"q50_nrmse":math.sqrt(mse)/6.3,"q50_mae_kw":float(np.mean(np.abs(median-yy))),"q50_r2":1-float(np.sum((median-yy)**2))/denom if denom else math.nan}


def ramp_metrics(labels:np.ndarray,probability:np.ndarray,mask:np.ndarray)->dict:
    y=labels[mask].astype(int); p=probability[mask]; pred=p>=.5
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    return {"ramp_auroc":roc_auc_score(y,p) if len(np.unique(y))>1 else math.nan,"ramp_auprc":average_precision_score(y,p) if y.any() else math.nan,"ramp_brier":float(np.mean((p-y)**2)),"ramp_precision":precision,"ramp_recall":recall,"ramp_f1":f1}


def evaluate(condition,seed,y,q,probability,ramp,daylight,cfg,info,params,infer_seconds)->list[dict]:
    rows=[]
    for horizon in PREFIXES:
        yh,qh,rh,dh=y[:,:horizon],q[:,:horizon],ramp[:,:horizon],daylight[:,:horizon]
        masks={"full_test":np.ones(yh.shape,bool),"daylight":dh,"ramp":rh,"non_ramp":~rh}
        for scope,mask in masks.items():
            row={"condition":condition,"seed":seed,"horizon":horizon,"scope":scope,**probability_metrics(yh,qh,mask)}
            if condition=="RAMP_AWARE_NCQ": row.update(ramp_metrics(rh,probability[:,:horizon],mask))
            else: row.update({k:math.nan for k in ("ramp_auroc","ramp_auprc","ramp_brier","ramp_precision","ramp_recall","ramp_f1")})
            abs_change=np.abs(yh-info["prior"][:,:horizon]); valid=np.isfinite(probability[:,:horizon])&mask
            row["ramp_probability_abs_change_spearman"]=float(spearmanr(probability[:,:horizon][valid],abs_change[valid]).statistic) if valid.sum()>2 else math.nan
            row.update({"parameter_count":params,**info["training"],"inference_seconds":infer_seconds,"inference_ms_per_sample":infer_seconds/len(y)*1000,"test_samples":len(y),"ramp_threshold_kw":info["threshold"],"ramp_pos_weight":info["pos_weight"]})
            rows.append(row)
    return rows


def write_metrics(rows):
    with METRICS.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main()->None:
    cfg=config(); prepared=(ROOT/ cfg["prepared_data"]).resolve()
    if not prepared.exists(): raise FileNotFoundError(f"Required commit-265cd618 prepared data missing: {prepared}")
    d=np.load(prepared,allow_pickle=False); features,power,columns=d["scaled_features"],d["power"],d["base_cols"]
    center,scale,threshold=float(d["target_center"]),float(d["target_scale"]),float(d["ramp_threshold"])
    train_origins=d["train_origins"]; train_ramps=np.concatenate([(np.abs(power[o+1:o+cfg["horizon"]+1]-power[o:o+cfg["horizon"]])>=threshold) for o in train_origins]); positives=int(train_ramps.sum()); pos_weight=float((len(train_ramps)-positives)/positives)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); RESULTS.mkdir(parents=True,exist_ok=True); all_rows=[]
    for condition in cfg["conditions"]:
        for seed in cfg["seeds"]:
            set_seed(seed); run_dir=RESULTS/condition/str(seed); run_dir.mkdir(parents=True,exist_ok=True)
            datasets={s:ProbabilityDataset(features,power,d[f"{s}_origins"],columns,cfg,center,scale,threshold) for s in ("train","validation","test")}
            loaders={s:DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=s=="train",num_workers=0,pin_memory=torch.cuda.is_available()) for s,ds in datasets.items()}
            model=NCQModel(len(columns),cfg,condition=="RAMP_AWARE_NCQ").to(device); training=train_model(model,loaders["train"],loaders["validation"],cfg,device,run_dir,pos_weight)
            checkpoint=torch.load(run_dir/"best_validation.pt",map_location=device,weights_only=True); model.load_state_dict(checkpoint["state_dict"]); q,probability,infer_seconds=predict(model,loaders["test"],device,center,scale)
            origins=d["test_origins"]; y=np.stack([power[o+1:o+cfg["horizon"]+1] for o in origins]); prior=np.stack([power[o:o+cfg["horizon"]] for o in origins]); ramp=np.abs(y-prior)>=threshold; daylight=y>.01*cfg["capacity_kw"]
            timestamps=d["times"][origins]; np.savez_compressed(run_dir/"test_probabilistic.npz",quantiles=q,ramp_probability=probability,labels=y,forecast_origin_timestamp_ns=timestamps,ramp_mask=ramp,daylight_mask=daylight)
            params=sum(p.numel() for p in model.parameters()); info={"training":training,"prior":prior,"threshold":threshold,"pos_weight":pos_weight}
            all_rows.extend(evaluate(condition,seed,y,q,probability,ramp,daylight,cfg,info,params,infer_seconds)); write_metrics(all_rows)
    write_metrics(all_rows)


if __name__=="__main__": main()

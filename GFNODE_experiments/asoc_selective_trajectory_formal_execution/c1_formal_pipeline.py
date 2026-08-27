"""Frozen C1 formal pipeline primitives.

Production code is deliberately separated from authorization: importing and testing this
module never starts training.  Real execution requires ``authorize_real_execution=True``.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PRIMARY_POPULATION = "THREE_ARRAY_COMMON_MASK_AVAILABLE_ORIGIN_DAYLIGHT"
RISK_METHODS = ("FULL_RISK_MODEL", "RECENT_VARIATION", "MODEL_PERSISTENCE_DISAGREEMENT")


def utc_to_acst(values: np.ndarray) -> np.ndarray:
    return values.astype("datetime64[s]") + np.timedelta64(570, "m")


def right_closed_bin(timestamp: np.datetime64) -> np.datetime64:
    """Return the right edge t for the interval (t-5 min, t]."""
    sec = timestamp.astype("datetime64[s]").astype(np.int64)
    return np.asarray(((sec + 299) // 300) * 300, dtype="datetime64[s]")[()]


def combine_source_seconds(chunks: list[np.ndarray], year: int) -> dict[str, Any]:
    """Combine explicitly supplied chunks by their actual UTC records, never filenames."""
    start = np.datetime64(f"{year}-01-01T00:00:00", "s").astype(np.int64)
    end = np.datetime64(f"{year + 1}-01-01T00:00:00", "s").astype(np.int64)
    ordered = sorted((np.asarray(x, dtype="datetime64[s]") for x in chunks), key=lambda x: x[0] if len(x) else np.datetime64("NaT"))
    seen = np.zeros(end-start, dtype=bool)
    duplicate = inverse = out_of_year = 0
    previous: int | None = None
    block_rows = []
    for block in ordered:
        raw = block.astype(np.int64)
        if len(raw):
            block_rows.append((str(block[0]), str(block[-1]), len(block)))
        inside=(raw>=start)&(raw<end); out_of_year += int((~inside).sum())
        vals=raw[inside]-start
        inverse += int(np.sum(np.diff(raw)<0))
        if previous is not None and len(raw) and raw[0] < previous: inverse += 1
        if len(vals):
            duplicate += int(seen[vals].sum()) + int(len(vals)-len(np.unique(vals)))
            seen[vals]=True
        if len(raw): previous=int(raw[-1])
    expected = end - start
    locations=np.flatnonzero(seen)
    return {"unique_seconds": int(seen.sum()), "missing_seconds": int(expected - seen.sum()),
            "duplicate_seconds": duplicate, "inverse_records": inverse,
            "out_of_year_records": out_of_year, "block_order": block_rows,
            "first_utc": str(np.asarray(start+int(locations[0]), dtype="datetime64[s]")[()]) if len(locations) else "UNKNOWN",
            "last_utc": str(np.asarray(start+int(locations[-1]), dtype="datetime64[s]")[()]) if len(locations) else "UNKNOWN"}


def window_origins(stage_index: np.ndarray, valid: np.ndarray, lookback: int = 72, horizon: int = 12) -> np.ndarray:
    """Origins whose complete history/target remain in one stage and valid segment."""
    stage_index = np.asarray(stage_index)
    valid = np.asarray(valid, bool)
    out = []
    for i in range(lookback - 1, len(valid) - horizon):
        lo, hi = i - lookback + 1, i + horizon
        if valid[lo:hi + 1].all() and np.all(stage_index[lo:hi + 1] == stage_index[i]):
            out.append(i)
    return np.asarray(out, dtype=np.int64)


def common_origins(per_array: dict[str, np.ndarray]) -> np.ndarray:
    values = [np.asarray(v, np.int64) for v in per_array.values()]
    return np.asarray(sorted(set(values[0]).intersection(*(set(v) for v in values[1:]))), dtype=np.int64)


def primary_daylight_common(origins: np.ndarray, powers: dict[str, np.ndarray], thresholds: dict[str, float]) -> np.ndarray:
    mask = np.ones(len(origins), dtype=bool)
    for name, values in powers.items():
        mask &= np.isfinite(values[origins]) & (values[origins] > thresholds[name])
    return mask


def build_14_features(power: np.ndarray, mb_mean: np.ndarray, mb_fraction: np.ndarray,
                      mb_mask: np.ndarray, times_acst: np.ndarray) -> np.ndarray:
    """Construct the frozen causal 14-channel grid."""
    dt = times_acst.astype("datetime64[s]")
    day = dt.astype("datetime64[D]")
    second = (dt - day).astype("timedelta64[s]").astype(float)
    year = dt.astype("datetime64[Y]")
    doy = (day - year.astype("datetime64[D]")).astype("timedelta64[D]").astype(float)
    cols = [power, np.sin(2*np.pi*second/86400), np.cos(2*np.pi*second/86400),
            np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)]
    for c in range(3):
        cols += [mb_mean[:, c], mb_fraction[:, c], mb_mask[:, c]]
    return np.column_stack(cols).astype(np.float32)


def causal_windows(features: np.ndarray, targets: np.ndarray, origins: np.ndarray,
                   lookback: int = 72, horizon: int = 12) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([features[i-lookback+1:i+1] for i in origins])
    y = np.stack([targets[i+1:i+horizon+1] for i in origins])
    return x, y


@dataclass
class TrainOnlyPreprocessor:
    fill: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    target_mean: float | None = None
    target_scale: float | None = None
    target_range: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, stage: str) -> "TrainOnlyPreprocessor":
        if stage != "BASE_TRAIN":
            raise ValueError("preprocessing may only fit on BASE_TRAIN")
        flat = x.reshape(-1, x.shape[-1]).astype(float)
        self.fill = np.nanmedian(flat, axis=0)
        filled = np.where(np.isfinite(flat), flat, self.fill)
        self.mean = filled.mean(axis=0); self.scale = filled.std(axis=0)
        for idx in (6,7,9,10,12,13):
            self.mean[idx] = 0.0; self.scale[idx] = 1.0
        self.scale[self.scale <= 1e-8] = 1.0
        yf = y[np.isfinite(y)].astype(float)
        self.target_mean = float(yf.mean()); self.target_scale = float(yf.std())
        if self.target_scale <= 1e-8: self.target_scale = 1.0
        self.target_range = float(yf.max() - yf.min())
        if self.target_range <= 0: raise ValueError("BASE_TRAIN target range must be positive")
        return self

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.fill is None: raise RuntimeError("preprocessor not fitted")
        return ((np.where(np.isfinite(x), x, self.fill) - self.mean) / self.scale).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.target_mean) / self.target_scale).astype(np.float32)

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_scale + self.target_mean


def make_model(config: dict[str, Any]):
    import torch.nn as nn
    base = config["base_forecaster"]
    layers: list[nn.Module] = [nn.Conv1d(14, 64, 1), nn.GELU()]
    for dilation in base["dilation_schedule"]:
        layers += [nn.Conv1d(64, 64, 5, padding=2*dilation, dilation=dilation, groups=64),
                   nn.Conv1d(64, 64, 1), nn.GELU()]
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(64 * config["lookback"], config["horizon"]))


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def train_base_model(model, train_loader, validation_loader, preprocessor: TrainOnlyPreprocessor,
                     config: dict[str, Any], checkpoint_path: Path, seed: int) -> dict[str, Any]:
    """Formal training API: intentionally has no Final-Test loader argument."""
    import torch
    set_seed(seed); tcfg = config["training"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    best = math.inf; bad = 0; best_epoch = 0
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        for batch in train_loader:
            xb,yb=(batch["x"],batch["y"]) if isinstance(batch,dict) else batch[:2]; xb=xb.to(device); yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True); pred = model(xb)
            loss = torch.mean((pred-yb)**2)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["gradient_clipping"]); optimizer.step()
        model.eval(); se = n = 0
        with torch.inference_mode():
            for batch in validation_loader:
                xb,yb=(batch["x"],batch["y"]) if isinstance(batch,dict) else batch[:2]; xb=xb.to(device); yb=yb.to(device)
                pred = preprocessor.inverse_y(model(xb).cpu().numpy()); truth = preprocessor.inverse_y(yb.cpu().numpy())
                se += float(np.sum((pred-truth)**2)); n += truth.size
        rmse = math.sqrt(se/n)
        if rmse < best - tcfg["min_delta"]:
            best, best_epoch, bad = rmse, epoch, 0
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "validation_rmse_kw": rmse,
                        "input_dim":14, "lookback":72, "horizon":12}, checkpoint_path)
        else:
            bad += 1
            if bad >= tcfg["patience"]: break
    return {"best_epoch": best_epoch, "validation_rmse_kw": best, "stop_epoch": epoch}


def predict_stage(model, loader, preprocessor: TrainOnlyPreprocessor) -> dict[str,np.ndarray]:
    """No-gradient stage prediction retaining the causal foundations needed by risk features."""
    import torch
    device=next(model.parameters()).device; buckets: dict[str,list[np.ndarray]]={k:[] for k in ("prediction","label","origin","history_power","history_mb","history_valid","daylight")}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if not isinstance(batch,dict): raise TypeError("formal loaders must return named batch dictionaries")
            pred=model(batch["x"].to(device)).cpu().numpy()
            buckets["prediction"].append(preprocessor.inverse_y(pred)); buckets["label"].append(preprocessor.inverse_y(batch["y"].cpu().numpy()))
            for key in ("origin","history_power","history_mb","history_valid","daylight"): buckets[key].append(np.asarray(batch[key]))
    result={k:np.concatenate(v) for k,v in buckets.items()}; result["origin"]=result["origin"].astype("datetime64[ns]"); return result


def last_value_persistence(last_power: np.ndarray, horizon: int = 12) -> np.ndarray:
    return np.repeat(np.asarray(last_power)[:, None], horizon, axis=1)


def _stats(v: np.ndarray) -> list[float]:
    d = np.diff(v); x = np.arange(len(v), dtype=float)
    slope = float(np.polyfit(x, v, 1)[0]) if len(v) > 1 else 0.0
    return [float(v[-1]), float(np.mean(v)), float(np.std(v)), float(np.ptp(v)),
            float(np.max(np.abs(d))) if len(d) else 0., float(np.mean(np.abs(d))) if len(d) else 0., slope,
            float(v[-1]-np.mean(v))]


def build_risk_features(history_power: np.ndarray, history_mb: np.ndarray, history_valid: np.ndarray,
                        predictions: np.ndarray, origins_acst: np.ndarray, daylight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows=[]; recent=[]; disagreement=[]
    for i in range(len(predictions)):
        dt=origins_acst[i]; day=dt.astype("datetime64[D]"); sec=float((dt-day)/np.timedelta64(1,"s")); doy=float((day-dt.astype("datetime64[Y]").astype("datetime64[D]"))/np.timedelta64(1,"D"))
        row=[math.sin(2*math.pi*sec/86400),math.cos(2*math.pi*sec/86400),math.sin(2*math.pi*doy/365.25),math.cos(2*math.pi*doy/365.25),sec/3600,float(daylight[i])]
        for length in (12,36,72): row += _stats(history_power[i,-length:])
        for c in range(3):
            v=history_mb[i,-12:,c]; d=np.diff(v)
            row += [float(np.nanmean(v)),float(np.nanstd(v)),float(np.nanmax(v)-np.nanmin(v)),float(np.nanmax(np.abs(d))),float(np.mean(history_valid[i,-12:,c]))]
        persist=np.full(12,history_power[i,-1]); diff=np.abs(predictions[i]-persist)
        pd=[float(diff.mean()),float(diff.max()),float(diff[-1]),float(np.abs(np.diff(predictions[i])).sum()),float(np.ptp(predictions[i])),float(np.max(np.abs(np.diff(predictions[i]))))]
        row += pd; rows.append(row); recent.append(_stats(history_power[i,-12:])[4]); disagreement.append(pd[0])
    return np.asarray(rows,float),np.asarray(recent,float),np.asarray(disagreement,float)


def trajectory_loss_kw(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(pred)-np.asarray(label))**2,axis=1))


def fit_risk_estimator(x: np.ndarray, loss_kw: np.ndarray, target_range: float, stage: str,
                       config: dict[str, Any], seed: int):
    if stage != "RISK_FIT": raise ValueError("risk estimator may only fit on RISK_FIT")
    from sklearn.ensemble import HistGradientBoostingRegressor
    params={k:v for k,v in config["risk_estimator"].items() if k not in ("class","random_state")}
    model=HistGradientBoostingRegressor(**params,random_state=seed)
    model.fit(x,np.log1p(loss_kw/target_range)); return model


def calibrate_threshold(scores: np.ndarray, stage: str, nominal: float=.8) -> dict[str, Any]:
    if stage != "RISK_CALIBRATION": raise ValueError("threshold may only use RISK_CALIBRATION")
    scores=np.asarray(scores,float); index=int(math.ceil(nominal*(len(scores)-1)))
    threshold=float(np.quantile(scores,nominal,method="higher")); accepted=scores<=threshold
    return {"n":len(scores),"zero_based_index":index,"threshold":threshold,"accepted_count":int(accepted.sum()),"realized_coverage":float(accepted.mean())}


def aurc(scores: np.ndarray, losses: np.ndarray, origins: np.ndarray) -> float:
    order=np.lexsort((np.asarray(origins).astype("datetime64[ns]").astype(np.int64),np.asarray(scores)))
    losses=np.asarray(losses,float)[order]; grid=np.arange(.05,1.001,.01)
    risk=np.asarray([losses[:max(1,int(math.ceil(c*len(losses))))].mean() for c in grid])
    denom=.95*losses.mean()
    return float(np.trapezoid(risk,grid)/denom)


def bootstrap_metrics(labels: np.ndarray, pred: np.ndarray, persistence: np.ndarray, accepted: np.ndarray,
                      origins: np.ndarray, replicates: int, seed: int, block_days: int = 7) -> dict[str, float]:
    """Fixed-mask continuous moving-block bootstrap."""
    days=np.asarray(origins).astype("datetime64[D]"); unique=np.unique(days); rng=np.random.default_rng(seed); vals=[]
    starts=np.arange(max(1,len(unique)-block_days+1))
    for _ in range(replicates):
        chosen=[]
        while len(chosen)<len(unique):
            s=int(rng.choice(starts)); chosen.extend(unique[s:s+block_days].tolist())
        idx=np.concatenate([np.flatnonzero(days==d) for d in chosen[:len(unique)]])
        ai=idx[accepted[idx]]
        if not len(ai): continue
        full=float(np.sqrt(np.mean((pred[idx]-labels[idx])**2))); acc=float(np.sqrt(np.mean((pred[ai]-labels[ai])**2))); per=float(np.sqrt(np.mean((persistence[ai]-labels[ai])**2)))
        vals.append((1-acc/full,1-acc/per,float(len(ai)/len(idx))))
    a=np.asarray(vals)
    return {"valid_replicates":len(a),"skipped_replicates":replicates-len(a),
            **{f"{name}_{q}":float(np.quantile(a[:,j],p)) for j,name in enumerate(("rmse_reduction","persistence_skill","coverage")) for q,p in (("mean",.5),("lo",.025),("hi",.975))}}


def evaluate_success(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the seven preregistered primary-population conditions."""
    full=[r for r in rows if r["risk_method"]=="FULL_RISK_MODEL"]
    by_array={a:[r for r in full if r["array"]==a] for a in sorted({r["array"] for r in full})}
    macro_cov=float(np.mean([np.mean([r["coverage"] for r in v]) for v in by_array.values()]))
    array_cov={a:float(np.mean([r["coverage"] for r in v])) for a,v in by_array.items()}
    macro_aurc=float(np.mean([r["best_simple_aurc_improvement"] for r in full]))
    array_dir=sum(np.mean([r["best_simple_aurc_improvement"] for r in v])>0 for v in by_array.values())
    array_persist=sum(np.mean([r["accepted_rmse"]-r["persistence_rmse"] for r in v])<=0 for v in by_array.values())
    seeds=sorted({r["seed"] for r in full}); seed_red=[np.mean([r["rmse_reduction"] for r in full if r["seed"]==s]) for s in seeds]
    seed_aurc=sum(np.mean([r["best_simple_aurc_improvement"] for r in full if r["seed"]==s])>0 for s in seeds)
    c=config["success_conditions"]
    checks={"macro_coverage":c["macro_coverage"][0]<=macro_cov<=c["macro_coverage"][1],"minimum_array_coverage":min(array_cov.values())>=c["minimum_array_coverage"],"macro_aurc_improvement":macro_aurc>=c["macro_aurc_improvement"],"arrays_aurc_direction":array_dir>=c["arrays_aurc_direction"],"arrays_beating_persistence":array_persist>=c["arrays_beating_persistence"],"mean_seed_macro_rmse_reduction":float(np.mean(seed_red))>=c["mean_seed_macro_rmse_reduction"],"seeds_with_aurc_direction":seed_aurc>=c["seeds_with_aurc_direction"]}
    return {"passed":all(checks.values()),"checks":checks,"decision":"C1_FORMAL_METHOD_PASS" if all(checks.values()) else "C1_FORMAL_METHOD_FAIL"}


def evaluate_locked_test(test: dict[str,np.ndarray], scores: dict[str,np.ndarray], thresholds: dict[str,dict[str,Any]]) -> list[dict[str,Any]]:
    """One-shot locked evaluation; all methods share origins, labels and primary scope."""
    origins=test["origin"].astype("datetime64[ns]"); labels=test["label"]; pred=test["prediction"]
    persistence=last_value_persistence(test["history_power"][:,-1]); losses=trajectory_loss_kw(pred,labels)
    if any(len(v)!=len(origins) for v in scores.values()): raise ValueError("risk methods do not share origins")
    rows=[]
    for method,score in scores.items():
        accepted=np.asarray(score)<=thresholds[method]["threshold"]
        if not accepted.any(): raise ValueError("locked threshold accepts no Final-Test origins")
        ar=float(np.sqrt(np.mean((pred[accepted]-labels[accepted])**2))); ur=float(np.sqrt(np.mean((pred-labels)**2))); pr=float(np.sqrt(np.mean((persistence[accepted]-labels[accepted])**2)))
        rows.append({"risk_method":method,"coverage":float(accepted.mean()),"accepted_count":int(accepted.sum()),"accepted_rmse":ar,"unselected_rmse":ur,"persistence_rmse":pr,"rmse_reduction":1-ar/ur,"aurc":aurc(score,losses,origins)})
    simple=min(r["aurc"] for r in rows if r["risk_method"]!="FULL_RISK_MODEL")
    for row in rows: row["best_simple_aurc_improvement"]=(simple-row["aurc"])/simple
    return rows


def execute_formal(config: dict[str, Any], prepared: dict[str, Any] | None, results_dir: Path,
                   data_ready: bool, authorize_real_execution: bool = False) -> dict[str, Any]:
    """Complete guarded entry; never mistakes lack of authorization for unimplemented code."""
    if not data_ready:
        return {"status":"DATA_FAIL","training_started":False,"completed_runs":0}
    if not authorize_real_execution:
        return {"status":"READY_AWAITING_GPU_AUTHORIZATION","training_started":False,"completed_runs":0}
    if prepared is None: raise ValueError("prepared five-stage arrays required")
    results_dir.mkdir(parents=True,exist_ok=True)
    frozen=[]
    # The orchestration deliberately delegates each frozen phase to the tested production
    # primitives above.  Final-Test access is delayed until every base/risk/calibration artifact exists.
    for array in config["arrays"]:
        for seed in config["seeds"]:
            item=prepared[array]; model=make_model(config); set_seed(seed)
            train_info=train_base_model(model,item["train_loader"],item["validation_loader"],item["preprocessor"],config,results_dir/f"{array}_{seed}_best.pt",seed)
            import torch
            checkpoint=torch.load(results_dir/f"{array}_{seed}_best.pt",map_location=next(model.parameters()).device,weights_only=True); model.load_state_dict(checkpoint["model"])
            rf=predict_stage(model,item["risk_fit_loader"],item["preprocessor"]); rx,recent,disagree=build_risk_features(rf["history_power"],rf["history_mb"],rf["history_valid"],rf["prediction"],rf["origin"],rf["daylight"])
            risk_loss=trajectory_loss_kw(rf["prediction"],rf["label"]); risk=fit_risk_estimator(rx,risk_loss,item["preprocessor"].target_range,"RISK_FIT",config,seed)
            cal=predict_stage(model,item["calibration_loader"],item["preprocessor"]); cx,crecent,cdisagree=build_risk_features(cal["history_power"],cal["history_mb"],cal["history_valid"],cal["prediction"],cal["origin"],cal["daylight"])
            calibration_scores={"FULL_RISK_MODEL":np.expm1(risk.predict(cx)),"RECENT_VARIATION":crecent,"MODEL_PERSISTENCE_DISAGREEMENT":cdisagree}
            thresholds={m:calibrate_threshold(calibration_scores[m],"RISK_CALIBRATION",config["calibration"]["nominal_coverage"]) for m in RISK_METHODS}
            frozen.append({"array":array,"seed":seed,"model":model,"risk":risk,"thresholds":thresholds,"train":train_info,"item":item})
    if len(frozen)!=9: raise RuntimeError("fixed 9-run matrix incomplete")
    # Final-Test is first touched only after all nine base/risk/calibration bundles are frozen.
    final_rows=[]
    for run in frozen:
        test=predict_stage(run["model"],run["item"]["final_test_loader_factory"](),run["item"]["preprocessor"])
        tx,recent,disagree=build_risk_features(test["history_power"],test["history_mb"],test["history_valid"],test["prediction"],test["origin"],test["daylight"])
        scores={"FULL_RISK_MODEL":np.expm1(run["risk"].predict(tx)),"RECENT_VARIATION":recent,"MODEL_PERSISTENCE_DISAGREEMENT":disagree}
        for row in evaluate_locked_test(test,scores,run["thresholds"]): row.update(array=run["array"],seed=run["seed"]); final_rows.append(row)
    success=evaluate_success(final_rows,config)
    return {"status":success["decision"],"training_started":True,"completed_runs":9,"metrics":final_rows,"success":success}


def prepare_from_audit_state(config: dict[str,Any], state_path: Path) -> dict[str,Any]:
    """Build the frozen five-stage loaders directly from the compact read-only audit state."""
    import torch
    from torch.utils.data import DataLoader,Dataset
    state=np.load(state_path,allow_pickle=False); grid=state["grid_ns"].astype("datetime64[ns]"); mb=state["hf_channel_mean"].T
    count=state["hf_channel_count"].T.astype(float); fraction=np.clip(count/300.,0,1); valid=(count>0).astype(float)
    class NamedDataset(Dataset):
        def __init__(self,payload): self.payload=payload
        def __len__(self): return len(self.payload["y"])
        def __getitem__(self,i): return {k:(torch.as_tensor(v[i]) if k not in ("origin",) else v[i]) for k,v in self.payload.items()}
    def loader(payload,shuffle=False): return DataLoader(NamedDataset(payload),batch_size=config["training"]["batch_size"],shuffle=shuffle,num_workers=config["training"]["num_workers"])
    out={}; arrays=config["arrays"]
    for ai,array in enumerate(arrays):
        power=state[f"pv_power_{ai}"]; features=build_14_features(power,mb,fraction,valid,grid)
        stage_payload={}
        for stage in config["stages"]:
            origins=state[f"origins__{stage}__COMMON"].astype(np.int64)
            if stage in ("RISK_FIT","RISK_CALIBRATION","FINAL_TEST"):
                origins=state[f"primary_daylight__{stage}"].astype(np.int64)
            x,y=causal_windows(features,power,origins,config["lookback"],config["horizon"])
            stage_payload[stage]={"x":x,"y":y,"origin":grid[origins].astype("datetime64[ns]").astype(np.int64),
                "history_power":np.stack([power[i-71:i+1] for i in origins]).astype(np.float32),
                "history_mb":np.stack([mb[i-71:i+1] for i in origins]).astype(np.float32),
                "history_valid":np.stack([valid[i-71:i+1] for i in origins]).astype(np.float32),
                "daylight":np.ones(len(origins),dtype=np.float32)}
        pre=TrainOnlyPreprocessor().fit(stage_payload["BASE_TRAIN"]["x"],stage_payload["BASE_TRAIN"]["y"],"BASE_TRAIN")
        for payload in stage_payload.values(): payload["x"]=pre.transform_x(payload["x"]); payload["y"]=pre.transform_y(payload["y"])
        out[array]={"preprocessor":pre,"train_loader":loader(stage_payload["BASE_TRAIN"],True),"validation_loader":loader(stage_payload["BASE_MODEL_VALIDATION"]),"risk_fit_loader":loader(stage_payload["RISK_FIT"]),"calibration_loader":loader(stage_payload["RISK_CALIBRATION"]),"final_test_loader_factory":(lambda payload=stage_payload["FINAL_TEST"]:loader(payload))}
    return out

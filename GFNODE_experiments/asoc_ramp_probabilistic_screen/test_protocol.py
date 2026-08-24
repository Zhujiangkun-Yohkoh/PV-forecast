"""Ordinary leakage and model-structure tests for the probabilistic screen."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_probabilistic_screen as screen


def main()->None:
    c=screen.config(); d=np.load((screen.ROOT/c["prepared_data"]).resolve(),allow_pickle=False); times=pd.to_datetime(d["times"]); power=d["power"]
    origins={s:d[f"{s}_origins"] for s in ("train","validation","test")}; test_origins=origins["test"]
    labels=np.stack([power[o+1:o+c["horizon"]+1] for o in test_origins]); timestamps=d["times"][test_origins]

    # Deterministic reference, both probability conditions, labels and timestamps are common.
    deterministic=screen.SOURCE_DIR/"results"/"MEAN_ONLY"
    for seed in c["seeds"]:
        artifact=deterministic/str(seed)/"test_predictions.npz"; assert artifact.exists()
        old=np.load(artifact); assert np.array_equal(old["labels"],labels); assert np.array_equal(old["forecast_origin_timestamp_ns"],timestamps)

    # Disjoint, split-contained windows with causal inputs and exact H12 timing.
    assert not set(origins["train"])&set(origins["validation"]); assert not set(origins["train"])&set(origins["test"]); assert not set(origins["validation"])&set(origins["test"])
    for split,oo in origins.items():
        lo,hi=map(pd.Timestamp,c["splits"][split])
        for o in (oo[0],oo[len(oo)//2],oo[-1]):
            history=times[o-c["lookback"]+1:o+1]; target=times[o+1:o+c["horizon"]+1]
            assert history[-1]<=times[o] and history[0]>=lo and target[-1]<hi
            assert target[0]==times[o]+pd.Timedelta(minutes=5) and target[-1]==times[o]+pd.Timedelta(minutes=60)

    # Structured parameterization guarantees shape and ordering without post-hoc sorting.
    for aware in (False,True):
        model=screen.NCQModel(len(d["base_cols"]),c,aware); q,r=model(torch.randn(4,c["lookback"],len(d["base_cols"])))
        assert q.shape==(4,12,3); assert torch.all(q[:,:,0]<=q[:,:,1]) and torch.all(q[:,:,1]<=q[:,:,2])
        assert (r is None) if not aware else r.shape==(4,12)
    source=inspect.getsource(screen.NCQModel.forward); assert "sort" not in source and "ramp" not in inspect.signature(screen.NCQModel.forward).parameters

    # Ramp threshold and first-step label use Train only and the last historical PV value.
    train_changes=np.concatenate([np.abs(power[o+1:o+c["horizon"]+1]-power[o:o+c["horizon"]]) for o in origins["train"]])
    assert np.isclose(float(d["ramp_threshold"]),np.quantile(train_changes,.9),rtol=1e-6)
    o=int(origins["test"][0]); expected=abs(power[o+1]-power[o])>=float(d["ramp_threshold"])
    ds=screen.ProbabilityDataset(d["scaled_features"],power,origins["test"][:1],d["base_cols"],c,float(d["target_center"]),float(d["target_scale"]),float(d["ramp_threshold"])); assert bool(ds[0][2][0])==bool(expected)

    # Train-only scaling provenance and MEAN_ONLY feature set (no failed dynamics fields).
    names=list(d["feature_names"]); selected=[names[int(i)] for i in d["base_cols"]]
    assert not any(any(token in name for token in ("_std","_range","_slope","max_abs_diff","first_last_change")) for name in selected)
    train_rows=(times>=c["splits"]["train"][0])&(times<c["splits"]["train"][1]); protected={names.index(f"{ch}_valid_mask") for ch in screen.source.CHANNELS}|{names.index(f"{ch}_valid_fraction") for ch in screen.source.CHANNELS}
    raw=d["raw_features"]
    for j in d["base_cols"]:
        vals=raw[train_rows,j]; vals=vals[np.isfinite(vals)]
        if int(j) not in protected: assert np.isclose(d["center"][j],np.mean(vals),rtol=1e-5,atol=1e-5)

    # Test loader cannot enter training/checkpoint selection; only Validation pinball is used.
    signature=inspect.signature(screen.train_model); assert "test_loader" not in signature.parameters
    training_source=inspect.getsource(screen.train_model); assert "validation_loader" in training_source and "validation_mean_pinball" in training_source and "test_loader" not in training_source
    print("PASS: 14 common-index, causal, non-crossing, ramp-label, Train-fit and Validation-checkpoint checks")


if __name__=="__main__": main()

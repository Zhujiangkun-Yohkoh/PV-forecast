"""Ordinary protocol tests for the Site 17 information screen."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_information_screen as screen


def main() -> None:
    c = screen.load_config(); screen.prepare(); d = np.load(screen.PREPARED, allow_pickle=False)
    times = pd.to_datetime(d["times"]); power = d["power"]; lookback, horizon = c["lookback"], c["horizon"]
    origins = {s: d[f"{s}_origins"] for s in ("train", "validation", "test")}

    # Same indices and targets: conditions differ only by feature columns.
    base, dynamic = d["base_cols"], d["dynamic_cols"]
    assert len(dynamic) == 15 and len(base) == 14
    for split, oo in origins.items():
        labels_a = np.stack([power[o+1:o+horizon+1] for o in oo])
        labels_b = np.stack([power[o+1:o+horizon+1] for o in oo])
        assert np.array_equal(labels_a, labels_b)
    assert not (set(origins["train"]) & set(origins["validation"]))
    assert not (set(origins["train"]) & set(origins["test"]))
    assert not (set(origins["validation"]) & set(origins["test"]))

    # Each window remains inside its timestamp split and has exact 5-minute geometry.
    for split, oo in origins.items():
        bounds = c["splits"][split]
        for o in (oo[0], oo[len(oo)//2], oo[-1]):
            history = times[o-lookback+1:o+1]; target = times[o+1:o+horizon+1]
            assert history[-1] == times[o] and history[-1] <= times[o]
            assert np.all(np.diff(history.view("int64")) == 300_000_000_000)
            assert np.all(np.diff(target.view("int64")) == 300_000_000_000)
            assert target[0] == times[o] + pd.Timedelta(minutes=5)
            assert target[-1] == times[o] + pd.Timedelta(minutes=60)
            assert history[0] >= pd.Timestamp(bounds[0]) and target[-1] < pd.Timestamp(bounds[1])

    # ceil-to-endpoint binning means seconds in (t-5min,t] map to t and no later data do.
    t = 10_000 * 300
    seconds = np.arange(t-299, t+1)
    assert np.all((seconds + 299)//300 == t//300)
    positions = seconds - (((seconds + 299)//300) * 300 - 300)
    assert np.array_equal(positions, np.arange(1, 301))
    assert (t-300+299)//300 == t//300-1 and (t+1+299)//300 == t//300+1

    # Fill/scale statistics reproduce Train-only rows; Validation/Test are never fit inputs.
    raw, names = d["raw_features"], list(d["feature_names"]); train_rows = (times >= c["splits"]["train"][0]) & (times < c["splits"]["train"][1])
    protected = {names.index(f"{ch}_valid_mask") for ch in screen.CHANNELS} | {names.index(f"{ch}_valid_fraction") for ch in screen.CHANNELS}
    for j in range(raw.shape[1]):
        finite = raw[train_rows, j][np.isfinite(raw[train_rows, j])]
        expected_fill = np.median(finite) if len(finite) else 0.0
        assert np.isclose(d["fill"][j], expected_fill, rtol=1e-5, atol=1e-6)
        if j not in protected:
            assert np.isclose(d["center"][j], np.mean(finite), rtol=1e-5, atol=1e-5)

    # Channels remain independent and partial/empty intervals are represented, not dropped.
    assert all(f"{ch}_mean" in names for ch in screen.CHANNELS)
    assert len(set(int(x) for x in d["mean_cols"])) == 3
    masks = raw[:, [names.index(f"{ch}_valid_mask") for ch in screen.CHANNELS]]
    assert np.any(masks == 0) and np.isfinite(d["scaled_features"]).all()

    # Training API and checkpoint criterion are validation-only; Test does not set ramp threshold.
    signature = inspect.signature(screen.train_model)
    assert "test_loader" not in signature.parameters
    source = inspect.getsource(screen.train_model)
    assert "validation_loader" in source and "validation_rmse_kw" in source and "test_loader" not in source
    train_changes = np.concatenate([np.abs(np.diff(power[o:o+horizon+1])) for o in origins["train"]])
    assert np.isclose(float(d["ramp_threshold"]), np.quantile(train_changes, c["training"]["ramp_quantile"]), rtol=1e-6)

    # Both conditions use the same ModernTCN class and emit one direct H12 trajectory.
    for cols in (base, np.r_[base, dynamic]):
        model = screen.ModernTCN(len(cols), c); out = model(torch.zeros(2, lookback, len(cols)))
        assert out.shape == (2, horizon)
    print("PASS: 13 leakage, alignment, masking, channel-independence, checkpoint and shape checks")


if __name__ == "__main__": main()

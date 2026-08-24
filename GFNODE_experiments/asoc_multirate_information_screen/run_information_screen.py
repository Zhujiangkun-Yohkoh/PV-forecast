"""Leakage-free Site 17 multirate irradiance information screen."""
from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PREPARED = RESULTS / "prepared_data.npz"
METRICS = ROOT / "metrics_per_seed.csv"
CHANNELS = ("MB0", "MB1", "MB2")
PREFIXES = (3, 6, 12)


def load_config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def parse_pv_timestamp(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.strip().strip('"'))
    except (ValueError, TypeError):
        return None


def read_pv(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, dict]:
    start, end = datetime(2022, 1, 1), datetime(2023, 1, 1)
    grid = pd.date_range(start, end, freq="5min", inclusive="left")
    power = np.full(len(grid), np.nan, dtype=np.float32)
    present = np.zeros(len(grid), dtype=bool)
    malformed = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader([handle.readline().rstrip("\r\n")]))
        width, pidx = len(header), header.index("Active_Power")
        for raw in handle:
            parts = raw.rstrip("\r\n").split(",")
            if len(parts) != width:
                malformed += 1; continue
            ts = parse_pv_timestamp(parts[0])
            if ts is None or not start <= ts < end:
                continue
            seconds = (ts - start).total_seconds()
            if seconds % 300:
                continue
            idx = int(seconds // 300)
            present[idx] = True
            try: power[idx] = float(parts[pidx])
            except ValueError: pass
    return grid, power, {"pv_rows_present": int(present.sum()), "pv_values_valid": int(np.isfinite(power).sum()), "pv_malformed_file_total": malformed}


def aggregate_high_frequency(path: Path, offset_minutes: int, grid: pd.DatetimeIndex) -> tuple[np.ndarray, list[str], dict]:
    """Aggregate each channel independently over (t-5min,t], using UTC only."""
    n = len(grid); epoch = datetime(1970, 1, 1); start_key = int((grid[0].to_pydatetime() - epoch).total_seconds() // 300)
    count = np.zeros((3, n), np.int32); sums = np.zeros((3, n)); sumsq = np.zeros((3, n))
    sumt = np.zeros((3, n)); sumt2 = np.zeros((3, n)); sumtx = np.zeros((3, n))
    mins = np.full((3, n), np.inf); maxs = np.full((3, n), -np.inf)
    first = np.full((3, n), np.nan); last = np.full((3, n), np.nan); maxdiff = np.zeros((3, n))
    use = ["Timestamp_UNIX [s]"] + [f"Irradiance_{c} [W/m-2]" for c in CHANNELS]
    previous_key = np.full(3, -1, dtype=np.int64); previous_value = np.full(3, np.nan)
    rows = 0
    for chunk in pd.read_csv(path, usecols=use, chunksize=750_000, low_memory=False):
        unix = pd.to_numeric(chunk[use[0]], errors="coerce").to_numpy(np.float64)
        valid_ts = np.isfinite(unix)
        local_seconds = unix + offset_minutes * 60
        safe_seconds = np.where(valid_ts, local_seconds, 0).astype(np.int64)
        absolute_key = np.floor_divide(safe_seconds + 299, 300)
        idx = absolute_key - start_key
        # Position within the right-closed interval (t-5min, t]: 1..300 seconds.
        sec_in = local_seconds - (absolute_key * 300 - 300)
        rows += len(chunk)
        for ci, col in enumerate(use[1:]):
            values = pd.to_numeric(chunk[col], errors="coerce").to_numpy(np.float64)
            ok = valid_ts & np.isfinite(values) & (idx >= 0) & (idx < n)
            ii, vv, tt = idx[ok].astype(np.int64), values[ok], sec_in[ok]
            if not len(ii): continue
            np.add.at(count[ci], ii, 1); np.add.at(sums[ci], ii, vv); np.add.at(sumsq[ci], ii, vv * vv)
            np.add.at(sumt[ci], ii, tt); np.add.at(sumt2[ci], ii, tt * tt); np.add.at(sumtx[ci], ii, tt * vv)
            np.minimum.at(mins[ci], ii, vv); np.maximum.at(maxs[ci], ii, vv)
            starts = np.r_[True, ii[1:] != ii[:-1]]; ends = np.r_[ii[1:] != ii[:-1], True]
            first_idx = ii[starts]; first_val = vv[starts]; missing_first = ~np.isfinite(first[ci, first_idx]); first[ci, first_idx[missing_first]] = first_val[missing_first]
            last[ci, ii[ends]] = vv[ends]
            pair = ii[1:] == ii[:-1]
            if pair.any(): np.maximum.at(maxdiff[ci], ii[1:][pair], np.abs(vv[1:][pair] - vv[:-1][pair]))
            if previous_key[ci] == ii[0] and np.isfinite(previous_value[ci]): maxdiff[ci, ii[0]] = max(maxdiff[ci, ii[0]], abs(vv[0] - previous_value[ci]))
            previous_key[ci], previous_value[ci] = ii[-1], vv[-1]
    features, names = [], []
    for ci, channel in enumerate(CHANNELS):
        c = count[ci].astype(np.float64); has = c > 0
        mean = np.divide(sums[ci], c, out=np.full(n, np.nan), where=has)
        var = np.divide(sumsq[ci], c, out=np.zeros(n), where=has) - np.nan_to_num(mean) ** 2
        std = np.where(has, np.sqrt(np.maximum(var, 0)), np.nan)
        rng = np.where(has, maxs[ci] - mins[ci], np.nan); change = last[ci] - first[ci]
        den = c * sumt2[ci] - sumt[ci] ** 2
        slope = np.divide(c * sumtx[ci] - sumt[ci] * sums[ci], den, out=np.full(n, np.nan), where=den > 0)
        fraction = np.minimum(c / 300.0, 1.0); mask = has.astype(np.float64)
        for name, arr in (("mean", mean), ("valid_fraction", fraction), ("valid_mask", mask), ("std", std), ("range", rng), ("first_last_change", change), ("max_abs_diff", np.where(has, maxdiff[ci], np.nan)), ("slope", slope)):
            names.append(f"{channel}_{name}"); features.append(arr.astype(np.float32))
    return np.stack(features, axis=1), names, {"hf_source_rows": rows, "hf_bins_any_valid": int(np.any(count > 0, axis=0).sum())}


def split_mask(times: pd.DatetimeIndex, bounds: list[str]) -> np.ndarray:
    return (times >= pd.Timestamp(bounds[0])) & (times < pd.Timestamp(bounds[1]))


def origins_for_split(times: pd.DatetimeIndex, power: np.ndarray, split: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    valid = np.isfinite(power) & split
    candidates = np.flatnonzero(split)
    out = []
    for origin in candidates:
        lo, hi = origin - lookback + 1, origin + horizon
        if lo < 0 or hi >= len(power): continue
        if split[lo] and split[hi] and valid[lo:hi + 1].all(): out.append(origin)
    return np.asarray(out, dtype=np.int64)


def prepare(force: bool = False) -> None:
    if PREPARED.exists() and not force: return
    c = load_config(); RESULTS.mkdir(parents=True, exist_ok=True)
    grid, power, pv_info = read_pv(Path(c["pv_file"]))
    hf, hf_names, hf_info = aggregate_high_frequency(Path(c["high_frequency_file"]), c["utc_to_pv_clock_minutes"], grid)
    hour = (grid.hour.to_numpy() * 60 + grid.minute.to_numpy()) / 1440.0; doy = (grid.dayofyear.to_numpy() - 1) / 365.0
    time_features = np.stack([np.sin(2*np.pi*hour), np.cos(2*np.pi*hour), np.sin(2*np.pi*doy), np.cos(2*np.pi*doy)], axis=1).astype(np.float32)
    raw = np.column_stack([power, time_features, hf]).astype(np.float32)
    names = ["Active_Power", "tod_sin", "tod_cos", "doy_sin", "doy_cos"] + hf_names
    mean_cols = [names.index(f"{ch}_mean") for ch in CHANNELS]
    common_hf_cols = sum(([names.index(f"{ch}_mean"), names.index(f"{ch}_valid_fraction"), names.index(f"{ch}_valid_mask")] for ch in CHANNELS), [])
    dynamic_cols = sum(([names.index(f"{ch}_{x}") for x in ("std", "range", "first_last_change", "max_abs_diff", "slope")] for ch in CHANNELS), [])
    base_cols = [0, 1, 2, 3, 4] + common_hf_cols
    splits = {k: split_mask(grid, v) for k, v in c["splits"].items()}
    origins = {k: origins_for_split(grid, power, v, c["lookback"], c["horizon"]) for k, v in splits.items()}
    train_rows = splits["train"]
    fill = np.zeros(raw.shape[1], np.float32); center = np.zeros(raw.shape[1], np.float32); scale = np.ones(raw.shape[1], np.float32)
    mask_columns = [names.index(f"{ch}_valid_mask") for ch in CHANNELS]
    fraction_columns = [names.index(f"{ch}_valid_fraction") for ch in CHANNELS]
    for j in range(raw.shape[1]):
        vals = raw[train_rows, j]; finite = vals[np.isfinite(vals)]
        fill[j] = float(np.median(finite)) if len(finite) else 0.0
        if j not in mask_columns + fraction_columns:
            center[j] = float(np.mean(finite)) if len(finite) else 0.0
            scale[j] = float(np.std(finite)) if len(finite) and np.std(finite) > 1e-8 else 1.0
    filled = np.where(np.isfinite(raw), raw, fill); scaled = (filled - center) / scale
    y_train = np.concatenate([power[o+1:o+c["horizon"]+1] for o in origins["train"]])
    target_center, target_scale = float(np.mean(y_train)), float(np.std(y_train))
    train_changes = np.concatenate([np.abs(np.diff(power[o:o+c["horizon"]+1])) for o in origins["train"]])
    ramp_threshold = float(np.quantile(train_changes, c["training"]["ramp_quantile"]))
    timestamp_ns = grid.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    np.savez_compressed(PREPARED, times=timestamp_ns, raw_features=raw, scaled_features=scaled.astype(np.float32), power=power,
                        feature_names=np.asarray(names), base_cols=np.asarray(base_cols), dynamic_cols=np.asarray(dynamic_cols), mean_cols=np.asarray(mean_cols),
                        train_origins=origins["train"], validation_origins=origins["validation"], test_origins=origins["test"],
                        fill=fill, center=center, scale=scale, target_center=target_center, target_scale=target_scale, ramp_threshold=ramp_threshold,
                        metadata=json.dumps({**pv_info, **hf_info}))


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, power: np.ndarray, origins: np.ndarray, columns: np.ndarray, lookback: int, horizon: int, target_center: float, target_scale: float):
        self.features, self.power, self.origins, self.columns = features, power, origins, columns
        self.lookback, self.horizon, self.target_center, self.target_scale = lookback, horizon, target_center, target_scale
    def __len__(self): return len(self.origins)
    def __getitem__(self, i):
        o = int(self.origins[i]); x = self.features[o-self.lookback+1:o+1, self.columns]
        y = (self.power[o+1:o+self.horizon+1] - self.target_center) / self.target_scale
        return torch.from_numpy(x.copy()), torch.from_numpy(y.astype(np.float32))


class ModernTCN(nn.Module):
    """Exact architecture used by asoc_discrete_viability/benchmark.py."""
    def __init__(self, input_dim: int, config: dict):
        super().__init__(); m = config["model"]; ch = m["channels"]
        layers: list[nn.Module] = [nn.Conv1d(input_dim, ch, 1), nn.GELU()]
        for _ in range(m["layers"]):
            layers += [nn.Conv1d(ch, ch, m["kernel_size"], padding=m["kernel_size"]//2, groups=ch), nn.Conv1d(ch, ch, 1), nn.GELU()]
        self.net = nn.Sequential(*layers); self.out = nn.Linear(ch * config["lookback"], config["horizon"])
    def forward(self, x): return self.out(self.net(x.transpose(1, 2)).flatten(1))


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, config: dict, device: torch.device, run_dir: Path, target_scale: float) -> dict:
    """Train with validation-only checkpoint selection; intentionally has no Test loader."""
    opt = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    loss_fn = nn.MSELoss(); best = math.inf; stale = 0; best_epoch = 0; started = time.perf_counter(); epoch_times = []
    log = run_dir / "epochs.jsonl"; log.write_text("", encoding="utf-8")
    for epoch in range(1, config["training"]["max_epochs"] + 1):
        tick = time.perf_counter(); model.train(); losses = []
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_sq = []
        with torch.no_grad():
            for x, y in validation_loader:
                error = (model(x.to(device)) - y.to(device)) * target_scale; val_sq.append(error.square().cpu().numpy())
        val_rmse = float(np.sqrt(np.concatenate(val_sq).mean())); elapsed = time.perf_counter() - tick; epoch_times.append(elapsed)
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": opt.state_dict(), "validation_rmse_kw": val_rmse}, run_dir / "last.pt")
        with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"epoch": epoch, "train_mse_scaled": float(np.mean(losses)), "validation_rmse_kw": val_rmse, "seconds": elapsed}) + "\n")
        if val_rmse < best - 1e-8:
            best, best_epoch, stale = val_rmse, epoch, 0; torch.save({"epoch": epoch, "state_dict": copy.deepcopy(model.state_dict()), "validation_rmse_kw": val_rmse}, run_dir / "best_validation.pt")
        else:
            stale += 1
            if stale >= config["training"]["patience"]: break
    return {"actual_epochs": epoch, "best_epoch": best_epoch, "best_validation_rmse_kw": best, "training_seconds": time.perf_counter()-started, "mean_epoch_seconds": float(np.mean(epoch_times))}


def predict(model: nn.Module, loader: DataLoader, device: torch.device, center: float, scale: float) -> tuple[np.ndarray, float]:
    model.eval(); out = []; started = time.perf_counter()
    with torch.no_grad():
        for x, _ in loader: out.append(model(x.to(device)).cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(out) * scale + center, elapsed


def metric_values(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray, capacity: float) -> dict:
    y, p = labels[mask], predictions[mask]
    if not len(y): return {k: math.nan for k in ("rmse_kw", "nrmse", "mae_kw", "r2")}
    mse = float(np.mean((p-y)**2)); denom = float(np.sum((y-y.mean())**2))
    return {"rmse_kw": math.sqrt(mse), "nrmse": math.sqrt(mse)/capacity, "mae_kw": float(np.mean(np.abs(p-y))), "r2": 1-float(np.sum((p-y)**2))/denom if denom else math.nan}


def evaluate(condition: str, seed: int, labels: np.ndarray, predictions: np.ndarray, origins: np.ndarray, power: np.ndarray, config: dict, info: dict, params: int, infer_seconds: float) -> list[dict]:
    rows = []; threshold = info["ramp_threshold"]
    prior = np.stack([power[o:o+config["horizon"]] for o in origins]); ramp = np.abs(labels-prior) >= threshold
    for horizon in PREFIXES:
        y, p = labels[:, :horizon], predictions[:, :horizon]
        scopes = {"regular_full_timeline": np.ones(y.shape, bool), "daylight": y > 0.01*config["capacity_kw"], "ramp": ramp[:, :horizon]}
        for scope, mask in scopes.items():
            row = {"condition": condition, "seed": seed, "horizon": horizon, "scope": scope, **metric_values(y, p, mask, config["capacity_kw"])}
            row.update({"trajectory_diff_mae_kw": float(np.mean(np.abs(np.diff(p,axis=1)-np.diff(y,axis=1)))) if horizon>1 else math.nan,
                        "parameter_count": params, "parameter_difference_vs_mean_only_pct": info["parameter_difference_pct"], **info["training"],
                        "inference_seconds": infer_seconds, "inference_ms_per_sample": infer_seconds/len(labels)*1000, "test_samples": len(labels), "ramp_threshold_kw": threshold})
            rows.append(row)
    return rows


def run_all() -> None:
    prepare(); c = load_config(); data = np.load(PREPARED, allow_pickle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); RESULTS.mkdir(exist_ok=True)
    features, power = data["scaled_features"], data["power"]; center, scale = float(data["target_center"]), float(data["target_scale"])
    columns = {"MEAN_ONLY": data["base_cols"], "HF_DYNAMICS": np.r_[data["base_cols"], data["dynamic_cols"]]}
    params = {k: sum(p.numel() for p in ModernTCN(len(v), c).parameters()) for k,v in columns.items()}
    difference = (params["HF_DYNAMICS"]-params["MEAN_ONLY"])/params["MEAN_ONLY"]*100
    all_rows = []
    for condition in c["conditions"]:
        for seed in c["seeds"]:
            set_seed(seed); run_dir = RESULTS / condition / str(seed); run_dir.mkdir(parents=True, exist_ok=True)
            datasets = {s: WindowDataset(features, power, data[f"{s}_origins"], columns[condition], c["lookback"], c["horizon"], center, scale) for s in ("train","validation","test")}
            loaders = {s: DataLoader(ds, batch_size=c["training"]["batch_size"], shuffle=s=="train", num_workers=0, pin_memory=torch.cuda.is_available()) for s,ds in datasets.items()}
            model = ModernTCN(len(columns[condition]), c).to(device); training = train_model(model, loaders["train"], loaders["validation"], c, device, run_dir, scale)
            checkpoint = torch.load(run_dir/"best_validation.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["state_dict"])
            predictions, infer_seconds = predict(model, loaders["test"], device, center, scale)
            origins = data["test_origins"]; labels = np.stack([power[o+1:o+c["horizon"]+1] for o in origins]).astype(np.float32)
            timestamps = data["times"][origins]
            prior = np.stack([power[o:o+c["horizon"]] for o in origins]); ramp_mask = np.abs(labels-prior) >= float(data["ramp_threshold"])
            np.savez_compressed(run_dir/"test_predictions.npz", predictions=predictions.astype(np.float32), labels=labels, forecast_origin_timestamp_ns=timestamps,
                                daylight_mask=labels>0.01*c["capacity_kw"], ramp_mask=ramp_mask)
            info = {"ramp_threshold": float(data["ramp_threshold"]), "parameter_difference_pct": difference, "training": training}
            all_rows.extend(evaluate(condition, seed, labels, predictions, origins, power, c, info, params[condition], infer_seconds))
            write_metrics(all_rows)
    write_metrics(all_rows)


def write_metrics(rows: list[dict]) -> None:
    if not rows: return
    with METRICS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args(); prepare(force=args.force_prepare)
    if not args.prepare_only: run_all()


if __name__ == "__main__": main()

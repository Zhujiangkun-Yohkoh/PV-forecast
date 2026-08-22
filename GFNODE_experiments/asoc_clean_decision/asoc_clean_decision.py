"""Leakage-free, fixed-protocol decision experiment for time-conditioned GFNODE.

This module is deliberately independent of the legacy experiment scripts.  It keeps
the full five-minute clock, fits preprocessing objects on Train only, and creates
windows independently in each date split.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HORIZONS = (12, 48, 96, 144)


def load_config() -> dict:
    with (ROOT / "config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SplitWindows:
    x: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    day_mask: np.ndarray
    input_start: np.ndarray
    target_start: np.ndarray
    target_end: np.ndarray


class CleanDataProtocol:
    """Fixed date protocol with explicit Train-only fitted preprocessing."""

    def __init__(self, config: dict, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.fit_log: List[Dict[str, str]] = []
        self.feature_columns: List[str] = []
        self.knn: KNNImputer | None = None
        self.isolation_forest: IsolationForest | None = None
        self.feature_scaler: MinMaxScaler | None = None
        self.target_scaler: MinMaxScaler | None = None
        self.target_range: float = float("nan")
        self.raw: pd.DataFrame | None = None
        self.transformed: Dict[str, pd.DataFrame] = {}
        self.windows: Dict[str, SplitWindows] = {}

    def load_regularized_raw(self) -> pd.DataFrame:
        data_path = (ROOT / self.config["datasets"][self.dataset_name]).resolve()
        frame = pd.read_csv(data_path)
        timestamp = self.config["timestamp_column"]
        frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
        frame = frame.dropna(subset=[timestamp]).sort_values(timestamp)
        # Duplicate source stamps are not permitted to alter chronology: keep last
        # source observation and report the count in the split summary.
        duplicate_count = int(frame.duplicated(timestamp).sum())
        frame = frame.drop_duplicates(timestamp, keep="last").set_index(timestamp)
        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="5min")
        regular = frame.reindex(full_index)
        regular.index.name = timestamp
        regular["_source_timestamp_present"] = regular.index.isin(frame.index)
        regular.attrs["duplicate_count"] = duplicate_count
        regular.attrs["raw_min"] = str(frame.index.min())
        regular.attrs["raw_max"] = str(frame.index.max())
        self.raw = regular
        return regular

    def _date_slice(self, split: str) -> pd.DataFrame:
        if self.raw is None:
            raise RuntimeError("load_regularized_raw must run first")
        start, end = self.config["splits"][split]
        return self.raw.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()

    def fit_transform(self) -> Dict[str, SplitWindows]:
        if self.raw is None:
            self.load_regularized_raw()
        target = self.config["target_column"]
        timestamp = self.config["timestamp_column"]
        all_numeric = self.raw.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [c for c in all_numeric if c != target]
        train_raw = self._date_slice("train")
        # A feature absent for all of Train cannot be estimated without leakage.
        self.feature_columns = [c for c in self.feature_columns if not train_raw[c].isna().all()]
        train_features = train_raw[self.feature_columns].astype(float)
        self.knn = KNNImputer(n_neighbors=self.config["knn_neighbors"])
        self.knn.fit(train_features)
        self.fit_log.append({"preprocessor": "KNNImputer", "operation": "fit", "split": "train"})
        train_imputed = self.knn.transform(train_features)

        if self.config["isolation_forest"]["enabled"]:
            if_cfg = self.config["isolation_forest"]
            self.isolation_forest = IsolationForest(
                contamination=if_cfg["contamination"], random_state=if_cfg["random_state"], n_estimators=100
            )
            self.isolation_forest.fit(train_imputed)
            self.fit_log.append({"preprocessor": "IsolationForest", "operation": "fit", "split": "train"})
        train_augmented = self._augment_features(train_raw, train_imputed)
        self.feature_scaler = MinMaxScaler()
        self.feature_scaler.fit(train_augmented)
        self.fit_log.append({"preprocessor": "feature_MinMaxScaler", "operation": "fit", "split": "train"})

        valid_train_target = train_raw[target].notna() & np.isfinite(train_raw[target]) & (train_raw[target] >= 0)
        self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(train_raw.loc[valid_train_target, [target]].to_numpy(dtype=float))
        self.fit_log.append({"preprocessor": "target_MinMaxScaler", "operation": "fit", "split": "train"})
        self.target_range = float(self.target_scaler.data_max_[0] - self.target_scaler.data_min_[0])

        for split in ("train", "validation", "test"):
            piece = self._date_slice(split)
            imputed = self.knn.transform(piece[self.feature_columns].astype(float))
            augmented = self._augment_features(piece, imputed)
            scaled = self.feature_scaler.transform(augmented).astype(np.float32)
            transformed = pd.DataFrame(scaled, index=piece.index, columns=[f"x_{i}" for i in range(scaled.shape[1])])
            transformed["_target_raw"] = piece[target].astype(float)
            transformed["_target_valid"] = (
                piece[target].notna() & np.isfinite(piece[target]) & (piece[target] >= 0)
            ).to_numpy()
            ghi = self.config["irradiance_column"]
            transformed["_day_valid"] = (piece[ghi].astype(float) >= self.config["daylight_ghi_threshold"]).fillna(False).to_numpy()
            transformed["_source_timestamp_present"] = piece["_source_timestamp_present"].to_numpy()
            self.transformed[split] = transformed
            self.windows[split] = self._build_windows(transformed)
        self._validate_protocol(timestamp)
        return self.windows

    def _augment_features(self, piece: pd.DataFrame, imputed: np.ndarray) -> np.ndarray:
        missing_masks = piece[self.feature_columns].isna().astype(np.float32).to_numpy()
        if self.isolation_forest is None:
            anomaly = np.zeros((len(piece), 1), dtype=np.float32)
        else:
            # This is a marker only; the timestamp is never removed or compressed.
            anomaly = (self.isolation_forest.predict(imputed) == -1).astype(np.float32).reshape(-1, 1)
        return np.concatenate([imputed.astype(np.float32), missing_masks, anomaly], axis=1)

    def _build_windows(self, frame: pd.DataFrame) -> SplitWindows:
        lookback, horizon = self.config["lookback"], self.config["horizon"]
        x_columns = [c for c in frame.columns if c.startswith("x_")]
        xs: List[np.ndarray] = []
        ys_scaled: List[np.ndarray] = []
        ys_raw: List[np.ndarray] = []
        days: List[np.ndarray] = []
        input_start: List[np.datetime64] = []
        target_start: List[np.datetime64] = []
        target_end: List[np.datetime64] = []
        for i in range(0, len(frame) - lookback - horizon + 1):
            target_slice = frame.iloc[i + lookback:i + lookback + horizon]
            if not bool(target_slice["_target_valid"].all()):
                continue
            raw_target = target_slice["_target_raw"].to_numpy(dtype=np.float32)
            scaled_target = self.target_scaler.transform(raw_target.reshape(-1, 1)).reshape(-1).astype(np.float32)
            xs.append(frame.iloc[i:i + lookback][x_columns].to_numpy(dtype=np.float32))
            ys_scaled.append(scaled_target)
            ys_raw.append(raw_target)
            days.append(target_slice["_day_valid"].to_numpy(dtype=bool))
            input_start.append(frame.index[i].to_datetime64())
            target_start.append(frame.index[i + lookback].to_datetime64())
            target_end.append(frame.index[i + lookback + horizon - 1].to_datetime64())
        if not xs:
            raise RuntimeError("No valid windows were available after applying the fixed protocol")
        return SplitWindows(
            x=np.stack(xs), y_scaled=np.stack(ys_scaled), y_raw=np.stack(ys_raw), day_mask=np.stack(days),
            input_start=np.array(input_start), target_start=np.array(target_start), target_end=np.array(target_end),
        )

    def _validate_protocol(self, timestamp: str) -> None:
        split_stamps = {name: set(self.transformed[name].index) for name in self.transformed}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if split_stamps[left].intersection(split_stamps[right]):
                raise AssertionError(f"timestamp overlap between {left} and {right}")
        for name, win in self.windows.items():
            if not np.all((win.target_start - win.input_start) == np.timedelta64(self.config["lookback"] * 5, "m")):
                raise AssertionError(f"non-regular input/output transition in {name}")
            if not np.all((win.target_end - win.target_start) == np.timedelta64((self.config["horizon"] - 1) * 5, "m")):
                raise AssertionError(f"non-regular target steps in {name}")
        if any(entry["split"] != "train" for entry in self.fit_log):
            raise AssertionError("a preprocessor was fit outside Train")

    def summary(self) -> dict:
        assert self.raw is not None
        return {
            "dataset": self.dataset_name,
            "raw_timestamp_range": [self.raw.attrs["raw_min"], self.raw.attrs["raw_max"]],
            "duplicate_source_timestamps": self.raw.attrs["duplicate_count"],
            "feature_count_before_masks": len(self.feature_columns),
            "model_input_feature_count": int(self.windows["train"].x.shape[-1]),
            "target_scaler_train_range": [float(self.target_scaler.data_min_[0]), float(self.target_scaler.data_max_[0])],
            "target_range_for_nrmse": self.target_range,
            "fit_log": self.fit_log,
            "splits": {
                name: {
                    "timestamp_range": [str(self.transformed[name].index.min()), str(self.transformed[name].index.max())],
                    "regular_rows": len(self.transformed[name]),
                    "source_timestamp_missing_rows": int((~self.transformed[name]["_source_timestamp_present"]).sum()),
                    "valid_windows": len(self.windows[name].x),
                    "valid_evaluation_targets": int(self.windows[name].y_raw.size),
                    "daylight_evaluation_targets": int(self.windows[name].day_mask.sum()),
                }
                for name in ("train", "validation", "test")
            },
        }


class MSDTCN(nn.Module):
    def __init__(self, input_dim: int, branch_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(input_dim, branch_channels, 3, padding=d, dilation=d), nn.GELU(), nn.Dropout(dropout))
            for d in (1, 2, 4)
        ])
        self.projection = nn.Linear(branch_channels * 3, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        encoded = [branch(x.transpose(1, 2)).transpose(1, 2) for branch in self.branches]
        return self.projection(torch.cat(encoded, dim=-1))


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, local: Tensor, global_: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([local, global_], dim=-1)))
        return gate * local + (1.0 - gate) * global_


class ContextualAggregator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        states, _ = self.gru(x)
        weights = torch.softmax(self.score(states).squeeze(-1), dim=1)
        return torch.sum(states * weights.unsqueeze(-1), dim=1)


class TimeConditionedODEFunc(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2 + 1, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: Tensor | float, z: Tensor, context: Tensor) -> Tensor:
        time_value = torch.as_tensor(t, device=z.device, dtype=z.dtype).reshape(1, 1)
        # The scalar is normalized to the fixed physical 12-hour forecast range.
        t_norm = (time_value / 12.0).expand(z.shape[0], 1)
        return self.net(torch.cat([z, context, t_norm], dim=-1))


class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        d = cfg["embedding_dim"]
        self.local = MSDTCN(input_dim, cfg["tcn_branch_channels"], d, cfg["dropout"])
        self.input_projection = nn.Linear(input_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg["transformer_heads"], dim_feedforward=d * 2,
                                           dropout=cfg["dropout"], batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg["transformer_layers"])
        self.fusion = GatedFusion(d)
        self.aggregator = ContextualAggregator(d)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local(x)
        global_ = self.transformer(self.input_projection(x))
        return self.aggregator(self.fusion(local, global_))


class TimeConditionedGFNODE(nn.Module):
    model_name = "GFNODE"

    def __init__(self, input_dim: int, cfg: dict, frequency_minutes: int = 5):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, cfg)
        d = cfg["embedding_dim"]
        self.ode_func = TimeConditionedODEFunc(d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer("time_grid", torch.arange(0, 145, dtype=torch.float32) * (frequency_minutes / 60.0))

    def forward(self, x: Tensor) -> Tensor:
        context = self.encoder(x)
        z = context
        trajectory: List[Tensor] = []
        for i in range(144):
            t, next_t = self.time_grid[i], self.time_grid[i + 1]
            h = next_t - t
            k1 = self.ode_func(t, z, context)
            k2 = self.ode_func(t + h / 2, z + h * k1 / 2, context)
            k3 = self.ode_func(t + h / 2, z + h * k2 / 2, context)
            k4 = self.ode_func(next_t, z + h * k3, context)
            z = z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            trajectory.append(self.readout(z).squeeze(-1))
        return torch.stack(trajectory, dim=1)


class DiscreteTrajectoryDecoder(nn.Module):
    model_name = "Discrete"

    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, cfg)
        d = cfg["embedding_dim"]
        # This GRUCell has virtually the same parameter count as the ODE vector field.
        self.decoder = nn.GRUCell(d, d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: Tensor) -> Tensor:
        context = self.encoder(x)
        state = context
        trajectory: List[Tensor] = []
        for _ in range(144):
            state = self.decoder(context, state)
            trajectory.append(self.readout(state).squeeze(-1))
        return torch.stack(trajectory, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(windows: Dict[str, SplitWindows], batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def loader(split: str, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(windows[split].x), torch.from_numpy(windows[split].y_scaled))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())
    return loader("train", True), loader("validation", False), loader("test", False)


def train_one(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, cfg: dict, device: torch.device) -> Tuple[dict, dict]:
    """Early stopping intentionally accepts Validation only; Test is absent by design."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale = math.inf, 0, 0
    best_state: dict | None = None
    started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.to(device, non_blocking=True))
            loss = loss_fn(pred, y.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for x, y in validation_loader:
                val_loss = loss_fn(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
                losses.append(float(val_loss.cpu()))
        value = float(np.mean(losses))
        if value < best_loss - 1e-8:
            best_loss, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No validation-best model state was recorded")
    return best_state, {"actual_epochs": epoch, "best_epoch": best_epoch, "best_validation_mse": best_loss,
                        "training_seconds": time.perf_counter() - started, "numerically_finite": True}


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for x, _ in loader:
            batches.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(batches, axis=0)


def metric_row(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, denom: float) -> dict:
    yv, pv = y[mask], pred[mask]
    if len(yv) == 0:
        return {"evaluated_targets": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "nrmse": np.nan}
    rmse = float(np.sqrt(np.mean((pv - yv) ** 2)))
    mae = float(np.mean(np.abs(pv - yv)))
    centered = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = float(1.0 - np.sum((pv - yv) ** 2) / centered) if centered > 0 else np.nan
    return {"evaluated_targets": int(len(yv)), "rmse": rmse, "mae": mae, "r2": r2,
            "nrmse": float(rmse / denom) if denom > 0 else np.nan}


def evaluate_prefixes(y_raw: np.ndarray, pred_raw: np.ndarray, day_mask: np.ndarray, denominator: float) -> List[dict]:
    rows: List[dict] = []
    for horizon in HORIZONS:
        y, pred, day = y_raw[:, :horizon], pred_raw[:, :horizon], day_mask[:, :horizon]
        for scope, mask in (("regular_full_timeline", np.ones_like(day, dtype=bool)), ("predefined_daylight", day)):
            result = metric_row(y.reshape(-1), pred.reshape(-1), mask.reshape(-1), denominator)
            result.update({"horizon": horizon, "scope": scope, "prefix_source": "same_H144_prediction"})
            rows.append(result)
    return rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)


def run_single(dataset: str, model_kind: str, seed: int, config: dict, protocol: CleanDataProtocol, device: torch.device) -> List[dict]:
    set_seed(seed)
    model_cfg, train_cfg = config["model"], config["training"]
    input_dim = protocol.windows["train"].x.shape[-1]
    model: nn.Module = (TimeConditionedGFNODE(input_dim, model_cfg) if model_kind == "GFNODE"
                         else DiscreteTrajectoryDecoder(input_dim, model_cfg))
    train_loader, validation_loader, test_loader = make_loaders(protocol.windows, train_cfg["batch_size"])
    best_state, train_info = train_one(model, train_loader, validation_loader, train_cfg, device)
    model.load_state_dict(best_state)
    run_dir = RESULTS / dataset / model_kind / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "validation_best.pt"
    torch.save({"state_dict": best_state, "model": model_kind, "seed": seed, "best_validation_mse": train_info["best_validation_mse"]}, checkpoint_path)
    scaled_pred = predict(model, test_loader, device)
    raw_pred = protocol.target_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).reshape(scaled_pred.shape).astype(np.float32)
    test = protocol.windows["test"]
    np.savez_compressed(run_dir / "test_H144_predictions_and_labels.npz", predictions=raw_pred, labels=test.y_raw,
                        daylight_mask=test.day_mask, target_start=test.target_start, target_end=test.target_end)
    rows = evaluate_prefixes(test.y_raw, raw_pred, test.day_mask, protocol.target_range)
    for row in rows:
        row.update({"dataset": dataset, "model": model_kind, "seed": seed, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "prediction_file": str((run_dir / "test_H144_predictions_and_labels.npz").relative_to(ROOT)),
                    "parameter_count": parameter_count(model), **train_info})
    write_json(run_dir / "run_metadata.json", {"dataset": dataset, "model": model_kind, "seed": seed, "device": str(device),
                                                   "parameter_count": parameter_count(model), **train_info})
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[dict]) -> Tuple[List[dict], dict]:
    frame = pd.DataFrame(rows)
    numeric = ["rmse", "mae", "r2", "nrmse"]
    summary_rows: List[dict] = []
    for keys, group in frame.groupby(["dataset", "model", "scope", "horizon"], sort=True):
        record = dict(zip(["dataset", "model", "scope", "horizon"], keys))
        record["seeds"] = ",".join(str(x) for x in sorted(group["seed"].unique()))
        record["parameter_count"] = int(group["parameter_count"].iloc[0])
        for metric in numeric:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    decision: dict = {"dataset": {}, "all_runs_numerically_finite": bool(frame["numerically_finite"].all())}
    full = summary[(summary.scope == "regular_full_timeline")]
    for dataset in sorted(frame.dataset.unique()):
        pair = full[(full.dataset == dataset) & (full.horizon == 144)].set_index("model")
        gf, disc = float(pair.loc["GFNODE", "rmse_mean"]), float(pair.loc["Discrete", "rmse_mean"])
        seed_rows = frame[(frame.dataset == dataset) & (frame.scope == "regular_full_timeline") & (frame.horizon == 144)]
        seed_pivot = seed_rows.pivot(index="seed", columns="model", values="rmse")
        improvement = (disc - gf) / disc * 100.0
        h12 = full[(full.dataset == dataset) & (full.horizon == 12)].set_index("model")
        h12_improvement = (float(h12.loc["Discrete", "rmse_mean"]) - float(h12.loc["GFNODE", "rmse_mean"])) / float(h12.loc["Discrete", "rmse_mean"]) * 100.0
        decision["dataset"][dataset] = {
            "H144_gfnode_rmse_mean": gf, "H144_discrete_rmse_mean": disc, "H144_relative_rmse_improvement_percent": improvement,
            "H12_relative_rmse_improvement_percent": h12_improvement,
            "H144_seedwise_gfnode_better": int((seed_pivot["GFNODE"] < seed_pivot["Discrete"]).sum()),
        }
    datasets = list(decision["dataset"].values())
    criteria = {
        "both_datasets_H144_better": all(d["H144_relative_rmse_improvement_percent"] > 0 for d in datasets),
        "at_least_one_H144_improves_3_percent": any(d["H144_relative_rmse_improvement_percent"] >= 3 for d in datasets),
        "other_dataset_not_worse_than_1_percent": all(d["H144_relative_rmse_improvement_percent"] >= -1 for d in datasets),
        "long_horizon_not_more_than_1pp_weaker_than_H12": all(d["H144_relative_rmse_improvement_percent"] >= d["H12_relative_rmse_improvement_percent"] - 1 for d in datasets),
        "not_single_seed_effect": all(d["H144_seedwise_gfnode_better"] >= 2 for d in datasets),
        "no_numerical_divergence": decision["all_runs_numerically_finite"],
    }
    decision["criteria"] = criteria
    decision["verdict"] = "PASS" if all(criteria.values()) else "FAIL"
    return summary_rows, decision


def write_report(config: dict, split_summaries: dict, summary_rows: List[dict], decision: dict, counts: dict) -> None:
    gf_params = next(row["parameter_count"] for row in summary_rows if row["model"] == "GFNODE")
    discrete_params = next(row["parameter_count"] for row in summary_rows if row["model"] == "Discrete")
    diff = abs(gf_params - discrete_params) / max(gf_params, discrete_params) * 100.0
    lines = [
        "# Clean time-conditioned GFNODE decision experiment",
        "", "## Fixed protocol", "",
        "- Full 5-minute clock retained; missing source rows are reindexed and marked.",
        "- KNN imputer, Isolation Forest, feature scaler, and target scaler are fit only on Train.",
        "- Windows are built independently inside Train, Validation, and Test; lookback=72 and horizon=144.",
        "- Validation MSE alone selects each checkpoint. Test is evaluated only after checkpoint selection.",
        f"- Predefined daylight rule: Global_Horizontal_Radiation >= {config['daylight_ghi_threshold']}.",
        f"- Parameters: GFNODE={gf_params:,}; Discrete={discrete_params:,}; difference={diff:.3f}%.",
        "", "## Split and sample summary", "",
        "```json", json.dumps(split_summaries, indent=2, ensure_ascii=False), "```",
        "", "## Three-seed mean ± SD (regular and daylight metrics)", "",
        "| Dataset | Model | Scope | H | RMSE mean±SD | MAE mean±SD | R² mean±SD | nRMSE mean±SD |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['scope']} | {row['horizon']} | "
                     f"{row['rmse_mean']:.5f} ± {row['rmse_sd']:.5f} | {row['mae_mean']:.5f} ± {row['mae_sd']:.5f} | "
                     f"{row['r2_mean']:.5f} ± {row['r2_sd']:.5f} | {row['nrmse_mean']:.5f} ± {row['nrmse_sd']:.5f} |")
    lines += ["", "## Pre-specified route decision", ""]
    for dataset, values in decision["dataset"].items():
        lines.append(f"- {dataset}: H144 GFNODE-vs-Discrete relative RMSE improvement = "
                     f"{values['H144_relative_rmse_improvement_percent']:.3f}%; "
                     f"seedwise wins = {values['H144_seedwise_gfnode_better']}/3.")
    lines += ["", "| Criterion | Result |", "|---|---|"]
    for criterion, passed in decision["criteria"].items():
        lines.append(f"| {criterion} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", f"**Final decision: {decision['verdict']}**", "",
             "A FAIL means do not tune ODE depth, width, step, activation, or time encoding again in this project stage. "
             "The permitted next step is to replace the ODE-centered route, not to create a GFNODE v2/v3."]
    (ROOT / "DECISION_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all() -> None:
    config = load_config()
    RESULTS.mkdir(exist_ok=True)
    device = torch.device("cuda" if config["device"] == "cuda_if_available" and torch.cuda.is_available() else "cpu")
    all_rows: List[dict] = []
    split_summaries: dict = {}
    protocols: Dict[str, CleanDataProtocol] = {}
    for dataset in config["datasets"]:
        protocol = CleanDataProtocol(config, dataset)
        protocol.load_regularized_raw()
        protocol.fit_transform()
        protocols[dataset] = protocol
        split_summaries[dataset] = protocol.summary()
    write_json(ROOT / "split_and_sample_summary.json", split_summaries)
    for dataset, protocol in protocols.items():
        for model_kind in ("GFNODE", "Discrete"):
            for seed in config["seed_values"]:
                print(f"RUN dataset={dataset} model={model_kind} seed={seed}", flush=True)
                all_rows.extend(run_single(dataset, model_kind, seed, config, protocol, device))
                write_csv(ROOT / "metrics_per_seed.csv", all_rows)
    summary_rows, decision = summarize(all_rows)
    write_csv(ROOT / "metrics_summary_mean_sd.csv", summary_rows)
    write_json(ROOT / "decision.json", decision)
    write_report(config, split_summaries, summary_rows, decision, {"runs": 12})
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()
    if args.run_all:
        run_all()
    else:
        parser.error("use --run-all")

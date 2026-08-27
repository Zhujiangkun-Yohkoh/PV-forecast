"""Submission-critical correction for the deterministic PV benchmark.

The script adds causal history Active_Power and its missingness indicator, selects
checkpoints with a validation SSE/count MSE, and evaluates H12/H48/H96/H144 on
horizon-specific origins shared exactly with persistence.  It never receives a
Test loader in the training API.
"""
from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = Path(os.environ.get("SCHEME_A_CORRECTION_RESULTS_ROOT", HERE / "results")).resolve()
CONFIG_PATH = HERE / "config.json"
METRICS_PATH = HERE / "corrected_metrics.csv"
MODEL_NAMES = (
    "Discrete recurrent decoder",
    "Inverted-variate Transformer",
    "Joint-patch Transformer",
    "Depthwise convolutional TCN",
)
DATASETS = ("Sanyo", "Hanwha", "Qcells")


class StaleArtifactError(RuntimeError):
    """Raised when a completed run is inconsistent with the active protocol."""


def artifact_reference(run_id: str) -> str:
    """Stable provenance label without committing a machine-specific absolute path."""
    return f"local_results/{run_id}/test_H144.npz"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_path(cfg: dict, dataset: str) -> Path:
    override = os.environ.get("PV_CORRECTION_DATA_ROOT")
    roots = [Path(override)] if override else []
    roots.extend((REPO / item).resolve() for item in cfg["data_root_candidates"])
    for root in roots:
        candidate = root / cfg["datasets"][dataset]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate {cfg['datasets'][dataset]} in {roots}")


@dataclass
class WindowBundle:
    x: np.ndarray
    y_scaled: np.ndarray
    y_raw: np.ndarray
    target_valid: np.ndarray
    last_power: np.ndarray
    input_start: np.ndarray
    forecast_origin: np.ndarray
    target_start: np.ndarray


class CorrectedProtocol:
    """Regular timeline with Train-only preprocessing and causal power history."""

    def __init__(self, cfg: dict, dataset: str):
        self.cfg = cfg
        self.dataset = dataset
        self.data_path = resolve_data_path(cfg, dataset)
        self.feature_columns = list(cfg["corrected_feature_columns"])
        self.original_feature_columns = list(cfg["original_feature_columns"])
        self.fit_log: list[dict] = []
        self.raw: pd.DataFrame | None = None
        self.transformed: dict[str, pd.DataFrame] = {}
        self.train_windows: WindowBundle | None = None
        self.validation_windows: WindowBundle | None = None
        self.evaluation_windows: dict[str, WindowBundle] = {}
        self.knn: KNNImputer | None = None
        self.isolation_forest: IsolationForest | None = None
        self.feature_scaler: MinMaxScaler | None = None
        self.target_scaler: MinMaxScaler | None = None
        self.train_target_min = math.nan
        self.train_target_max = math.nan
        self.train_target_range = math.nan
        self.daylight_threshold = math.nan

    def load(self) -> pd.DataFrame:
        frame = pd.read_csv(self.data_path)
        expected = [self.cfg["timestamp_column"], self.cfg["target_column"], *self.original_feature_columns]
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise AssertionError(f"Missing source columns for {self.dataset}: {missing}")
        timestamp = self.cfg["timestamp_column"]
        frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
        frame = frame.dropna(subset=[timestamp]).sort_values(timestamp)
        duplicate_count = int(frame.duplicated(timestamp).sum())
        frame = frame.drop_duplicates(timestamp, keep="last").set_index(timestamp)
        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="5min")
        regular = frame.reindex(full_index)
        regular.index.name = timestamp
        regular["_source_timestamp_present"] = regular.index.isin(frame.index)
        regular.attrs.update(
            duplicate_count=duplicate_count,
            raw_min=str(frame.index.min()),
            raw_max=str(frame.index.max()),
        )
        self.raw = regular
        return regular

    def split_raw(self, split: str) -> pd.DataFrame:
        if self.raw is None:
            raise RuntimeError("load must be called first")
        start, end = self.cfg["splits"][split]
        return self.raw.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()

    @staticmethod
    def _valid_power(series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return np.isfinite(values) & (values >= 0)

    def fit_preprocessors(self) -> None:
        train = self.split_raw("train")
        train_features = train[self.feature_columns].astype(float)
        self.knn = KNNImputer(n_neighbors=self.cfg["knn_neighbors"])
        self.knn.fit(train_features)
        self.fit_log.append({"name": "KNNImputer", "split": "train"})
        train_imputed = self.knn.transform(train_features)
        if_cfg = self.cfg["isolation_forest"]
        self.isolation_forest = IsolationForest(
            contamination=if_cfg["contamination"],
            random_state=if_cfg["random_state"],
            n_estimators=if_cfg["n_estimators"],
        )
        self.isolation_forest.fit(train_imputed)
        self.fit_log.append({"name": "IsolationForest", "split": "train"})
        train_augmented = self._augment(train, train_imputed)
        self.feature_scaler = MinMaxScaler().fit(train_augmented)
        self.fit_log.append({"name": "feature_MinMaxScaler", "split": "train"})
        target = self.cfg["target_column"]
        valid = self._valid_power(train[target])
        values = train.loc[valid, [target]].to_numpy(dtype=float)
        self.target_scaler = MinMaxScaler().fit(values)
        self.fit_log.append({"name": "target_MinMaxScaler", "split": "train"})
        self.train_target_min = float(values.min())
        self.train_target_max = float(values.max())
        self.train_target_range = self.train_target_max - self.train_target_min
        self.daylight_threshold = 0.01 * self.train_target_max

    def _augment(self, piece: pd.DataFrame, imputed: np.ndarray) -> np.ndarray:
        missing = piece[self.feature_columns].isna().to_numpy(dtype=np.float32)
        anomaly = (self.isolation_forest.predict(imputed) == -1).astype(np.float32)[:, None]
        return np.concatenate([imputed.astype(np.float32), missing, anomaly], axis=1)

    def transform(self) -> None:
        if any(item is None for item in (self.knn, self.isolation_forest, self.feature_scaler, self.target_scaler)):
            raise RuntimeError("fit_preprocessors must be called first")
        target = self.cfg["target_column"]
        for split in ("train", "validation", "test"):
            piece = self.split_raw(split)
            imputed = self.knn.transform(piece[self.feature_columns].astype(float))
            augmented = self._augment(piece, imputed)
            scaled = self.feature_scaler.transform(augmented).astype(np.float32)
            transformed = pd.DataFrame(
                scaled,
                index=piece.index,
                columns=[f"x_{i}" for i in range(scaled.shape[1])],
            )
            raw_target = pd.to_numeric(piece[target], errors="coerce").astype(float)
            transformed["_target_raw"] = raw_target
            transformed["_target_valid"] = self._valid_power(raw_target)
            transformed["_source_timestamp_present"] = piece["_source_timestamp_present"].to_numpy()
            self.transformed[split] = transformed
        self.train_windows = self._build_full_h144("train")
        self.validation_windows = self._build_full_h144("validation")
        self.evaluation_windows = {
            split: self._build_horizon_eligible(split) for split in ("validation", "test")
        }
        self.validate()

    def prepare(self) -> "CorrectedProtocol":
        self.load()
        self.fit_preprocessors()
        self.transform()
        return self

    def _window_at(self, frame: pd.DataFrame, i: int, allow_partial_tail: bool) -> tuple:
        lookback = self.cfg["lookback"]
        horizon = self.cfg["output_horizon"]
        xcols = [c for c in frame.columns if c.startswith("x_")]
        input_part = frame.iloc[i:i + lookback]
        target_part = frame.iloc[i + lookback:min(i + lookback + horizon, len(frame))]
        y_raw = np.full(horizon, np.nan, dtype=np.float32)
        y_valid = np.zeros(horizon, dtype=bool)
        available = len(target_part)
        if available:
            y_raw[:available] = target_part["_target_raw"].to_numpy(dtype=np.float32)
            y_valid[:available] = target_part["_target_valid"].to_numpy(dtype=bool)
        if not allow_partial_tail and available != horizon:
            raise AssertionError("full H144 window unexpectedly truncated")
        valid_values = np.where(y_valid, y_raw, self.train_target_min)
        y_scaled = self.target_scaler.transform(valid_values[:, None]).reshape(-1).astype(np.float32)
        origin_row = frame.iloc[i + lookback - 1]
        origin_power = float(origin_row["_target_raw"])
        origin_valid = bool(origin_row["_target_valid"])
        return (
            input_part[xcols].to_numpy(dtype=np.float32), y_scaled, y_raw, y_valid,
            origin_power if origin_valid else np.nan,
            frame.index[i].to_datetime64(), frame.index[i + lookback - 1].to_datetime64(),
            frame.index[i + lookback].to_datetime64(),
        )

    @staticmethod
    def _stack(rows: list[tuple]) -> WindowBundle:
        columns = list(zip(*rows))
        return WindowBundle(
            x=np.stack(columns[0]), y_scaled=np.stack(columns[1]), y_raw=np.stack(columns[2]),
            target_valid=np.stack(columns[3]), last_power=np.asarray(columns[4], dtype=np.float32),
            input_start=np.asarray(columns[5]), forecast_origin=np.asarray(columns[6]),
            target_start=np.asarray(columns[7]),
        )

    def _build_full_h144(self, split: str) -> WindowBundle:
        frame = self.transformed[split]
        lb, h = self.cfg["lookback"], self.cfg["output_horizon"]
        rows = []
        for i in range(len(frame) - lb - h + 1):
            row = self._window_at(frame, i, allow_partial_tail=False)
            if bool(row[3].all()):
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No complete H144 {split} windows")
        return self._stack(rows)

    def _build_horizon_eligible(self, split: str) -> WindowBundle:
        frame = self.transformed[split]
        lb, minimum_h = self.cfg["lookback"], min(self.cfg["evaluation_horizons"])
        rows = []
        for i in range(len(frame) - lb - minimum_h + 1):
            row = self._window_at(frame, i, allow_partial_tail=True)
            if np.isfinite(row[4]) and bool(row[3][:minimum_h].all()):
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No horizon-eligible {split} windows")
        return self._stack(rows)

    def validate(self) -> None:
        if [x["split"] for x in self.fit_log] != ["train"] * 4:
            raise AssertionError("preprocessor fit outside Train")
        if self.original_feature_columns + [self.cfg["target_column"]] != self.feature_columns:
            raise AssertionError("fixed feature order changed")
        if self.train_windows.x.shape[-1] != 17:
            raise AssertionError(f"expected 17 corrected inputs, got {self.train_windows.x.shape[-1]}")
        for split, bundle in {
            "train": self.train_windows,
            "validation": self.validation_windows,
            **self.evaluation_windows,
        }.items():
            if not np.all(bundle.target_start - bundle.forecast_origin == np.timedelta64(5, "m")):
                raise AssertionError(f"forecast origin mismatch in {split}")
            if not np.all(bundle.forecast_origin - bundle.input_start == np.timedelta64(355, "m")):
                raise AssertionError(f"lookback chronology mismatch in {split}")
        split_indexes = {s: set(self.transformed[s].index) for s in self.transformed}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if split_indexes[a] & split_indexes[b]:
                raise AssertionError(f"split overlap: {a}/{b}")


class MSDTCN(nn.Module):
    def __init__(self, input_dim: int, branch_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, branch_channels, 3, padding=dilation, dilation=dilation),
                nn.GELU(), nn.Dropout(dropout),
            ) for dilation in (1, 2, 4)
        ])
        self.projection = nn.Linear(branch_channels * 3, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        parts = [branch(x.transpose(1, 2)).transpose(1, 2) for branch in self.branches]
        return self.projection(torch.cat(parts, dim=-1))


class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        d = cfg["embedding_dim"]
        self.local = MSDTCN(input_dim, cfg["tcn_branch_channels"], d, cfg["dropout"])
        self.input_projection = nn.Linear(input_dim, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg["transformer_heads"], dim_feedforward=d * 2,
            dropout=cfg["dropout"], batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg["transformer_layers"])
        self.gate = nn.Linear(d * 2, d)
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.score = nn.Linear(d, 1)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local(x)
        global_ = self.transformer(self.input_projection(x))
        weight = torch.sigmoid(self.gate(torch.cat([local, global_], dim=-1)))
        states, _ = self.gru(weight * local + (1 - weight) * global_)
        attention = torch.softmax(self.score(states).squeeze(-1), dim=1)
        return torch.sum(states * attention.unsqueeze(-1), dim=1)


class DiscreteRecurrentDecoder(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        self.encoder = SharedEncoder(input_dim, cfg)
        d = cfg["embedding_dim"]
        self.decoder = nn.GRUCell(d, d)
        self.readout = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: Tensor) -> Tensor:
        context = self.encoder(x)
        state = context
        values = []
        for _ in range(144):
            state = self.decoder(context, state)
            values.append(self.readout(state).squeeze(-1))
        return torch.stack(values, dim=1)


class InvertedVariateTransformer(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.projection = nn.Linear(72, d)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], d * 2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"])
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(input_dim * d, 144))

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.encoder(self.projection(x.transpose(1, 2))))


class JointPatchTransformer(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        self.patch = cfg["patch_length"]
        self.stride = cfg["stride"]
        self.patch_count = (72 - self.patch) // self.stride + 1
        d = cfg["d_model"]
        self.projection = nn.Linear(self.patch * input_dim, d)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], d * 2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"])
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(self.patch_count * d, 144))

    def forward(self, x: Tensor) -> Tensor:
        patches = x.unfold(1, self.patch, self.stride).permute(0, 1, 3, 2)
        patches = patches.reshape(x.size(0), self.patch_count, -1)
        return self.output(self.encoder(self.projection(patches)))


class DepthwiseConvolutionalTCN(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        channels = cfg["channels"]
        layers: list[nn.Module] = [nn.Conv1d(input_dim, channels, 1), nn.GELU()]
        for _ in range(cfg["layers"]):
            layers.extend([
                nn.Conv1d(channels, channels, cfg["kernel_size"],
                          padding=cfg["kernel_size"] // 2, groups=channels),
                nn.Conv1d(channels, channels, 1), nn.GELU(),
            ])
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(channels * 72, 144)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.network(x.transpose(1, 2)).flatten(1))


def make_model(name: str, input_dim: int, cfg: dict) -> nn.Module:
    if name == MODEL_NAMES[0]:
        return DiscreteRecurrentDecoder(input_dim, cfg["discrete_recurrent_decoder"])
    if name == MODEL_NAMES[1]:
        return InvertedVariateTransformer(input_dim, cfg["inverted_variate_transformer"])
    if name == MODEL_NAMES[2]:
        return JointPatchTransformer(input_dim, cfg["joint_patch_transformer"])
    if name == MODEL_NAMES[3]:
        return DepthwiseConvolutionalTCN(input_dim, cfg["depthwise_convolutional_tcn"])
    raise KeyError(name)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def training_loader(bundle: WindowBundle, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(bundle.x), torch.from_numpy(bundle.y_scaled),
        torch.from_numpy(bundle.target_valid),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def global_masked_mse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    sse = 0.0
    count = 0
    with torch.inference_mode():
        for x, y, valid in loader:
            prediction = model(x.to(device, non_blocking=True))
            target = y.to(device, non_blocking=True)
            mask = valid.to(device, non_blocking=True)
            difference = prediction[mask] - target[mask]
            sse += float(torch.sum(difference.square()).cpu())
            count += int(mask.sum().cpu())
    if count == 0:
        raise RuntimeError("Validation has no valid targets")
    return sse / count


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
                cfg: dict, device: torch.device, run_dir: Path) -> dict:
    """Train with Validation only. Test is intentionally absent from this signature."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    start_epoch = 1
    best = math.inf
    best_epoch = 0
    stale = 0
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best_validation.pt"
    log_path = run_dir / "epochs.jsonl"
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_validation_mse"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
    started = time.perf_counter()
    stop_reason = "max_epochs"
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        model.train()
        train_sse = 0.0
        train_count = 0
        epoch_started = time.perf_counter()
        for x, y, valid in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x.to(device, non_blocking=True))
            target = y.to(device, non_blocking=True)
            mask = valid.to(device, non_blocking=True)
            difference = prediction[mask] - target[mask]
            loss = difference.square().mean()
            if not torch.isfinite(loss):
                stop_reason = "nonfinite_loss"
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            finite_gradients = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if not finite_gradients:
                stop_reason = "nonfinite_gradient"
                raise FloatingPointError("non-finite training gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            train_sse += float(torch.sum(difference.detach().square()).cpu())
            train_count += int(mask.sum().cpu())
        validation_mse = global_masked_mse(model, validation_loader, device)
        train_mse = train_sse / train_count
        record = {
            "epoch": epoch, "train_global_mse": train_mse,
            "validation_global_mse": validation_mse,
            "train_valid_target_count": train_count,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if validation_mse < best - cfg["min_delta"]:
            best, best_epoch, stale = validation_mse, epoch, 0
            torch.save({
                "epoch": epoch, "state_dict": copy.deepcopy(model.state_dict()),
                "validation_global_mse": best,
            }, best_path)
        else:
            stale += 1
        torch.save({
            "epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_validation_mse": best, "best_epoch": best_epoch, "stale": stale,
        }, last_path)
        if stale >= cfg["patience"]:
            stop_reason = "early_stopping"
            break
    if not best_path.exists():
        raise RuntimeError("No Validation-best checkpoint was saved")
    return {
        "actual_epochs": epoch, "best_epoch": best_epoch, "best_validation_mse": best,
        "training_seconds": time.perf_counter() - started, "stop_reason": stop_reason,
    }


def predict_scaled(model: nn.Module, x: np.ndarray, batch_size: int,
                   device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for (batch,) in loader:
            outputs.append(model(batch.to(device, non_blocking=True)).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(outputs), time.perf_counter() - started


def inverse_target(protocol: CorrectedProtocol, scaled: np.ndarray) -> np.ndarray:
    return protocol.target_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(scaled.shape).astype(np.float32)


def metric_values(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray,
                  denominator: float) -> dict[str, float | int]:
    mask = mask & np.isfinite(labels) & np.isfinite(predictions)
    valid = int(mask.sum())
    if valid == 0:
        return {"RMSE": math.nan, "MAE": math.nan, "R2": math.nan,
                "Bias": math.nan, "range_nRMSE": math.nan, "valid_target_count": 0}
    y = labels[mask].astype(np.float64)
    p = predictions[mask].astype(np.float64)
    error = p - y
    rmse = float(np.sqrt(np.mean(error * error)))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(error * error) / total) if total > 0 else math.nan
    return {
        "RMSE": rmse, "MAE": float(np.mean(np.abs(error))), "R2": r2,
        "Bias": float(np.mean(error)), "range_nRMSE": rmse / denominator,
        "valid_target_count": valid,
    }


def persistence_arrays(protocol: CorrectedProtocol, bundle: WindowBundle) -> tuple[np.ndarray, np.ndarray]:
    last = np.repeat(bundle.last_power[:, None], protocol.cfg["output_horizon"], axis=1).astype(np.float32)
    series = pd.to_numeric(protocol.raw[protocol.cfg["target_column"]], errors="coerce")
    starts = pd.to_datetime(bundle.target_start).to_numpy(dtype="datetime64[ns]")[:, None]
    offsets = np.arange(protocol.cfg["output_horizon"]) * np.timedelta64(5, "m")
    lookup = (starts + offsets[None, :] - np.timedelta64(1, "D")).reshape(-1)
    values = series.reindex(pd.to_datetime(lookup)).to_numpy(dtype=float).reshape(bundle.y_raw.shape)
    daily = np.where(np.isfinite(values) & (values >= 0), values, np.nan).astype(np.float32)
    return last, daily


def base_metric_row(dataset: str, model: str, seed: int | str, analysis: str,
                    horizon: int, scope: str, statistic: str, origin_count: int,
                    valid_target_count: int, source: str) -> dict:
    return {
        "dataset": dataset, "model": model, "seed": seed, "analysis": analysis,
        "horizon_steps": horizon, "horizon_minutes": horizon * 5, "scope": scope,
        "statistic": statistic, "metric": "", "value": "", "unit": "",
        "forecast_origin_count": origin_count, "valid_target_count": valid_target_count,
        "normalization_definition": "Train target max minus Train target min",
        "prediction_artifact": source,
    }


def emit_metric_rows(rows: list[dict], base: dict, values: dict, persistence: dict | None = None) -> None:
    units = {"RMSE": "kW", "MAE": "kW", "R2": "dimensionless", "Bias": "kW",
             "range_nRMSE": "dimensionless", "RMSE_skill": "dimensionless",
             "MAE_skill": "dimensionless"}
    combined = dict(values)
    if persistence is not None:
        combined["RMSE_skill"] = 1.0 - values["RMSE"] / persistence["RMSE"]
        combined["MAE_skill"] = 1.0 - values["MAE"] / persistence["MAE"]
    for name in units:
        if name not in combined:
            continue
        row = dict(base)
        row.update(metric=name, value=combined[name], unit=units[name])
        rows.append(row)


def evaluation_mask(bundle: WindowBundle, horizon: int, scope: str,
                    daylight_threshold: float, full_h144_only: bool) -> tuple[np.ndarray, np.ndarray]:
    eligible = bundle.target_valid[:, :horizon].all(axis=1) & np.isfinite(bundle.last_power)
    if full_h144_only:
        eligible &= bundle.target_valid.all(axis=1)
    point_mask = np.repeat(eligible[:, None], horizon, axis=1)
    if scope == "daylight":
        point_mask &= bundle.y_raw[:, :horizon] > daylight_threshold
    return eligible, point_mask


def evaluate_prediction(rows: list[dict], protocol: CorrectedProtocol, model: str, seed: int,
                        prediction: np.ndarray, bundle: WindowBundle, artifact: str) -> None:
    last, _ = persistence_arrays(protocol, bundle)
    for analysis, full_h144_only in (("primary_horizon_specific", False), ("secondary_h144_common", True)):
        for horizon in protocol.cfg["evaluation_horizons"]:
            for scope in ("regular_full_timeline", "daylight"):
                eligible, mask = evaluation_mask(
                    bundle, horizon, scope, protocol.daylight_threshold, full_h144_only
                )
                model_values = metric_values(bundle.y_raw[:, :horizon], prediction[:, :horizon],
                                             mask, protocol.train_target_range)
                persistence_values = metric_values(bundle.y_raw[:, :horizon], last[:, :horizon],
                                                   mask, protocol.train_target_range)
                origin_count = int(np.any(mask, axis=1).sum())
                base = base_metric_row(protocol.dataset, model, seed, analysis, horizon, scope,
                                       "per_seed", origin_count, model_values["valid_target_count"], artifact)
                emit_metric_rows(rows, base, model_values, persistence_values)


def evaluate_persistence(rows: list[dict], protocol: CorrectedProtocol, bundle: WindowBundle) -> None:
    last, _ = persistence_arrays(protocol, bundle)
    for analysis, full_h144_only in (("primary_horizon_specific", False), ("secondary_h144_common", True)):
        for horizon in protocol.cfg["evaluation_horizons"]:
            for scope in ("regular_full_timeline", "daylight"):
                _, mask = evaluation_mask(bundle, horizon, scope, protocol.daylight_threshold, full_h144_only)
                values = metric_values(bundle.y_raw[:, :horizon], last[:, :horizon], mask,
                                       protocol.train_target_range)
                origin_count = int(np.any(mask, axis=1).sum())
                base = base_metric_row(protocol.dataset, "Last-value Persistence", "DETERMINISTIC",
                                       analysis, horizon, scope, "deterministic", origin_count,
                                       values["valid_target_count"], "causal_last_observed_power")
                emit_metric_rows(rows, base, values)


def evaluate_daily_matched(rows: list[dict], protocol: CorrectedProtocol, bundle: WindowBundle,
                           predictions: dict[tuple[str, int], np.ndarray]) -> None:
    """Evaluate every method on the exact point mask available to Daily Persistence."""
    last, daily = persistence_arrays(protocol, bundle)
    for horizon in protocol.cfg["evaluation_horizons"]:
        for scope in ("regular_full_timeline", "daylight"):
            matched_mask = daily_matched_point_mask(
                bundle, daily, horizon, scope, protocol.daylight_threshold
            )
            origin_count = int(np.any(matched_mask, axis=1).sum())
            daily_values = metric_values(
                bundle.y_raw[:, :horizon], daily[:, :horizon], matched_mask,
                protocol.train_target_range,
            )
            daily_base = base_metric_row(
                protocol.dataset, "Daily Persistence", "DETERMINISTIC",
                "supplementary_daily_matched", horizon, scope, "deterministic",
                origin_count, daily_values["valid_target_count"],
                "exact_24h_lag_no_interpolation; common point mask",
            )
            emit_metric_rows(rows, daily_base, daily_values)

            last_values = metric_values(
                bundle.y_raw[:, :horizon], last[:, :horizon], matched_mask,
                protocol.train_target_range,
            )
            last_base = base_metric_row(
                protocol.dataset, "Last-value Persistence", "DETERMINISTIC",
                "supplementary_daily_matched", horizon, scope, "deterministic",
                origin_count, last_values["valid_target_count"],
                "causal_last_observed_power; Daily-matched point mask",
            )
            emit_metric_rows(rows, last_base, last_values, daily_values)

            for (model_name, seed), prediction in predictions.items():
                values = metric_values(
                    bundle.y_raw[:, :horizon], prediction[:, :horizon], matched_mask,
                    protocol.train_target_range,
                )
                run_id = (
                    f"{model_name.replace(' ', '_').replace('-', '_')}_"
                    f"{protocol.dataset}_{seed}"
                )
                base = base_metric_row(
                    protocol.dataset, model_name, seed, "supplementary_daily_matched",
                    horizon, scope, "per_seed", origin_count,
                    values["valid_target_count"], artifact_reference(run_id),
                )
                emit_metric_rows(rows, base, values, daily_values)


def daily_matched_point_mask(bundle: WindowBundle, daily: np.ndarray, horizon: int,
                             scope: str, daylight_threshold: float) -> np.ndarray:
    """One shared point mask for Daily, Last-value, and every neural forecast."""
    _, base_mask = evaluation_mask(bundle, horizon, scope, daylight_threshold, False)
    return base_mask & np.isfinite(daily[:, :horizon])


def add_persistence_skill_zero(rows: list[dict]) -> None:
    additions = []
    for row in rows:
        if (row["model"] == "Last-value Persistence" and
                row["analysis"] != "supplementary_daily_matched" and
                row["metric"] in ("RMSE", "MAE")):
            added = dict(row)
            added["metric"] = f"{row['metric']}_skill"
            added["value"] = 0.0
            added["unit"] = "dimensionless"
            additions.append(added)
    rows.extend(additions)


def aggregate_neural_rows(rows: list[dict], seeds: list[int]) -> None:
    per_seed = pd.DataFrame([row for row in rows if row["statistic"] == "per_seed"])
    keys = ["dataset", "model", "analysis", "horizon_steps", "horizon_minutes", "scope", "metric", "unit"]
    additions = []
    for key, group in per_seed.groupby(keys, dropna=False):
        if sorted(group["seed"].astype(int).tolist()) != sorted(seeds):
            raise AssertionError(f"Incomplete seeds for {key}")
        template = group.iloc[0].to_dict()
        for statistic, value in (("mean", group["value"].astype(float).mean()),
                                 ("sample_sd", group["value"].astype(float).std(ddof=1))):
            item = dict(template)
            item.update(seed="MEAN" if statistic == "mean" else "SD", statistic=statistic,
                        value=float(value), prediction_artifact="aggregation_of_three_seed_artifacts")
            additions.append(item)
    rows.extend(additions)


def add_rank_rows(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    selection = frame[
        (frame["statistic"].isin(["mean", "deterministic"])) &
        (frame["metric"] == "RMSE") &
        (frame["analysis"] == "primary_horizon_specific") &
        (frame["model"] != "Daily Persistence")
    ]
    additions = []
    for key, group in selection.groupby(["dataset", "horizon_steps", "scope"]):
        ranked = group.sort_values("value")
        for rank, (_, source) in enumerate(ranked.iterrows(), start=1):
            item = source.to_dict()
            item.update(metric="RMSE_rank", value=rank, unit="rank", statistic="rank")
            additions.append(item)
    rows.extend(additions)


def add_run_information_rows(rows: list[dict], run_information: list[dict]) -> None:
    units = {
        "parameter_count": "parameters", "actual_epochs": "epochs", "best_epoch": "epoch",
        "best_validation_mse": "scaled_MSE", "training_seconds": "s",
        "prediction_seconds": "s", "latency_mean_ms": "ms", "latency_median_ms": "ms",
        "latency_sd_ms": "ms", "latency_p5_ms": "ms", "latency_p95_ms": "ms",
        "throughput_samples_s": "samples/s", "peak_memory_mb": "MiB",
        "throughput_batch_size": "samples",
    }
    for info in run_information:
        values = {key: info[key] for key in (
            "parameter_count", "actual_epochs", "best_epoch", "best_validation_mse",
            "training_seconds", "prediction_seconds",
        )}
        values.update(info["efficiency"])
        for metric, value in values.items():
            rows.append({
                "dataset": info["dataset"], "model": info["model"], "seed": info["seed"],
                "analysis": "run_metadata", "horizon_steps": 144, "horizon_minutes": 720,
                "scope": "not_applicable", "statistic": "per_seed", "metric": metric,
                "value": value, "unit": units[metric], "forecast_origin_count": "",
                "valid_target_count": "", "normalization_definition": "not_applicable",
                "prediction_artifact": artifact_reference(info["run_id"]),
            })


def efficiency_measurement(model: nn.Module, sample: np.ndarray, device: torch.device,
                           warmup: int = 30, repeats: int = 100) -> dict:
    model.eval()
    one = torch.from_numpy(sample[:1]).to(device)
    batch_size = min(256, len(sample))
    batch = torch.from_numpy(sample[:batch_size]).to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(one)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            if device.type == "cuda": torch.cuda.synchronize()
            started = time.perf_counter()
            model(one)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(30): model(batch)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return {
        "latency_mean_ms": float(np.mean(times)), "latency_median_ms": float(np.median(times)),
        "latency_sd_ms": float(np.std(times, ddof=1)), "latency_p5_ms": float(np.percentile(times, 5)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "throughput_samples_s": float(batch_size * 30 / elapsed),
        "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else math.nan,
        "throughput_batch_size": batch_size,
    }


def _array_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise StaleArtifactError(f"STALE_ARTIFACT: {name} shape {actual.shape} != {expected.shape}")
    if np.issubdtype(actual.dtype, np.floating):
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    if not equal:
        raise StaleArtifactError(f"STALE_ARTIFACT: {name} differs from current protocol")


def validate_completed_artifact(protocol: CorrectedProtocol, model_name: str, seed: int,
                                cfg: dict, device: torch.device, run_dir: Path,
                                reforward: bool = False) -> tuple[dict, np.ndarray, nn.Module]:
    """Validate a completed artifact before any prediction is reused."""
    run_id = f"{model_name.replace(' ', '_').replace('-', '_')}_{protocol.dataset}_{seed}"
    complete = run_dir / "completed.json"
    artifact = run_dir / "test_H144.npz"
    checkpoint = run_dir / "best_validation.pt"
    missing = [str(path) for path in (complete, artifact, checkpoint) if not path.is_file()]
    if missing:
        raise StaleArtifactError(f"STALE_ARTIFACT: missing required files for {run_id}: {missing}")
    info = json.loads(complete.read_text(encoding="utf-8"))
    expected_identity = {
        "run_id": run_id, "model": model_name, "dataset": protocol.dataset, "seed": seed,
    }
    for key, expected in expected_identity.items():
        if info.get(key) != expected:
            raise StaleArtifactError(
                f"STALE_ARTIFACT: completed.json {key}={info.get(key)!r}, expected {expected!r}"
            )
    if cfg["lookback"] != 72 or cfg["output_horizon"] != 144:
        raise StaleArtifactError("STALE_ARTIFACT: active lookback/Horizon differs from 72/144")
    expected_splits = {
        "train": ["2018-04-01 00:00:00", "2018-07-15 23:55:00"],
        "validation": ["2018-07-16 00:00:00", "2018-08-07 23:55:00"],
        "test": ["2018-08-08 00:00:00", "2018-08-31 23:55:00"],
    }
    if cfg["splits"] != expected_splits:
        raise StaleArtifactError("STALE_ARTIFACT: active split dates differ from completed protocol")

    bundle = protocol.evaluation_windows["test"]
    with np.load(artifact) as saved:
        required = {"predictions", "labels", "target_valid", "forecast_origin", "target_start"}
        if not required.issubset(saved.files):
            raise StaleArtifactError(f"STALE_ARTIFACT: missing arrays {sorted(required-set(saved.files))}")
        prediction = saved["predictions"].copy()
        labels = saved["labels"].copy()
        target_valid = saved["target_valid"].copy()
        origins = saved["forecast_origin"].astype("datetime64[ns]")
        starts = saved["target_start"].astype("datetime64[ns]")
    if prediction.shape != bundle.y_raw.shape or prediction.shape[1] != cfg["output_horizon"]:
        raise StaleArtifactError(
            f"STALE_ARTIFACT: prediction shape {prediction.shape} != {bundle.y_raw.shape}"
        )
    if not np.isfinite(prediction).all():
        raise StaleArtifactError("STALE_ARTIFACT: predictions contain non-finite values")
    _array_equal("labels", labels, bundle.y_raw)
    _array_equal("target_valid", target_valid, bundle.target_valid)
    _array_equal("forecast_origin", origins, bundle.forecast_origin.astype("datetime64[ns]"))
    _array_equal("target_start", starts, bundle.target_start.astype("datetime64[ns]"))

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state.get("epoch", -1)) != int(info.get("best_epoch", -2)):
        raise StaleArtifactError("STALE_ARTIFACT: checkpoint epoch differs from completed.json")
    if not math.isclose(float(state.get("validation_global_mse", math.nan)),
                        float(info.get("best_validation_mse", math.inf)),
                        rel_tol=1e-7, abs_tol=1e-10):
        raise StaleArtifactError("STALE_ARTIFACT: checkpoint Validation MSE differs from completed.json")
    model = make_model(model_name, 17, cfg).to(device)
    try:
        model.load_state_dict(state["state_dict"], strict=True)
    except Exception as exc:
        raise StaleArtifactError(f"STALE_ARTIFACT: checkpoint is not a 17-input model: {exc}") from exc
    if parameter_count(model) != int(info.get("parameter_count", -1)):
        raise StaleArtifactError("STALE_ARTIFACT: parameter count differs from completed.json")
    if reforward:
        scaled, _ = predict_scaled(model, bundle.x, cfg["training"]["batch_size"], device)
        regenerated = inverse_target(protocol, scaled)
        if not np.allclose(regenerated, prediction, rtol=2e-5, atol=2e-5):
            delta = float(np.max(np.abs(regenerated.astype(float) - prediction.astype(float))))
            raise StaleArtifactError(
                f"STALE_ARTIFACT: best checkpoint does not reproduce saved prediction; max_abs={delta}"
            )
    return info, prediction, model


def run_one(protocol: CorrectedProtocol, model_name: str, seed: int, cfg: dict,
            device: torch.device, allow_training: bool = True) -> tuple[dict, np.ndarray, dict]:
    run_id = f"{model_name.replace(' ', '_').replace('-', '_')}_{protocol.dataset}_{seed}"
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "test_H144.npz"
    complete = run_dir / "completed.json"
    model = make_model(model_name, protocol.train_windows.x.shape[-1], cfg).to(device)
    existing = [complete.exists(), artifact.exists(), (run_dir / "best_validation.pt").exists()]
    if any(existing):
        if not all(existing):
            raise StaleArtifactError(f"STALE_ARTIFACT: partial completed run {run_id}")
        info, prediction, model = validate_completed_artifact(
            protocol, model_name, seed, cfg, device, run_dir, reforward=False
        )
        efficiency = info["efficiency"]
        return info, prediction, efficiency
    if not allow_training:
        raise StaleArtifactError(f"STALE_ARTIFACT: completed artifact is absent for {run_id}")
    train_cfg = cfg["training"]
    train_loader = training_loader(protocol.train_windows, train_cfg["batch_size"], True)
    validation_loader = training_loader(protocol.validation_windows, train_cfg["batch_size"], False)
    info = train_model(model, train_loader, validation_loader, train_cfg, device, run_dir)
    state = torch.load(run_dir / "best_validation.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    test_bundle = protocol.evaluation_windows["test"]
    scaled, prediction_seconds = predict_scaled(model, test_bundle.x, train_cfg["batch_size"], device)
    prediction = inverse_target(protocol, scaled)
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Non-finite Test predictions for {run_id}")
    efficiency = efficiency_measurement(model, test_bundle.x, device)
    info.update({
        "run_id": run_id, "model": model_name, "dataset": protocol.dataset, "seed": seed,
        "parameter_count": parameter_count(model), "prediction_seconds": prediction_seconds,
        "efficiency": efficiency,
    })
    np.savez_compressed(
        artifact, predictions=prediction, labels=test_bundle.y_raw,
        target_valid=test_bundle.target_valid, forecast_origin=test_bundle.forecast_origin,
        target_start=test_bundle.target_start, last_power=test_bundle.last_power,
    )
    complete.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info, prediction, efficiency


def write_metrics(rows: list[dict]) -> None:
    fieldnames = [
        "dataset", "model", "seed", "analysis", "horizon_steps", "horizon_minutes",
        "scope", "statistic", "metric", "value", "unit", "forecast_origin_count",
        "valid_target_count", "normalization_definition", "prediction_artifact",
    ]
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preflight(cfg: dict) -> dict[str, CorrectedProtocol]:
    protocols = {dataset: CorrectedProtocol(cfg, dataset).prepare() for dataset in DATASETS}
    expected_original = [
        "Performance_Ratio", "Weather_Temperature_Celsius", "Weather_Relative_Humidity",
        "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation",
        "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted",
    ]
    if cfg["original_feature_columns"] != expected_original:
        raise AssertionError("The audited seven-feature order changed")
    if "test" in inspect.signature(train_model).parameters:
        raise AssertionError("Test unexpectedly appears in training API")
    return protocols


def run_all() -> None:
    cfg = load_config()
    source_paths = [resolve_data_path(cfg, dataset) for dataset in DATASETS]
    source_signatures = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_paths}
    protocols = preflight(cfg)
    if cfg["device"] == "cuda_if_available" and not torch.cuda.is_available():
        raise RuntimeError("The requested 36-run experiment requires the available GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    run_information = []
    for protocol in protocols.values():
        evaluate_persistence(rows, protocol, protocol.evaluation_windows["test"])
        reference = None
        for model_name in MODEL_NAMES:
            for seed in cfg["seeds"]:
                set_seed(seed)
                info, prediction, efficiency = run_one(protocol, model_name, seed, cfg, device)
                bundle = protocol.evaluation_windows["test"]
                if reference is None:
                    reference = (bundle.y_raw.copy(), bundle.target_valid.copy(), bundle.forecast_origin.copy())
                elif not (np.array_equal(bundle.y_raw, reference[0], equal_nan=True) and
                          np.array_equal(bundle.target_valid, reference[1]) and
                          np.array_equal(bundle.forecast_origin, reference[2])):
                    raise AssertionError("Model fairness arrays differ")
                artifact = artifact_reference(info["run_id"])
                evaluate_prediction(rows, protocol, model_name, seed, prediction, bundle, artifact)
                run_information.append(info)
                print(json.dumps({"completed": len(run_information), "of": 36, "run": info["run_id"],
                                  "best_epoch": info["best_epoch"]}), flush=True)
    add_persistence_skill_zero(rows)
    aggregate_neural_rows(rows, cfg["seeds"])
    add_rank_rows(rows)
    add_run_information_rows(rows, run_information)
    write_metrics(rows)
    (RESULTS / "run_summary.json").write_text(json.dumps(run_information, indent=2), encoding="utf-8")
    for path, signature in source_signatures.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
            raise AssertionError(f"Source data changed: {path}")
    print(json.dumps({"status": "completed", "runs": len(run_information),
                      "metric_rows": len(rows), "device": str(device)}), flush=True)


def evidence_only(reforward: bool = True) -> None:
    """Recompute final evidence from completed runs without training or mutating artifacts."""
    cfg = load_config()
    protocols = preflight(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completed = sorted(RESULTS.glob("*/completed.json"))
    if len(completed) != 36:
        raise StaleArtifactError(f"STALE_ARTIFACT: expected 36 completed runs, found {len(completed)}")
    protected = [p for marker in completed for p in (
        marker, marker.parent / "test_H144.npz", marker.parent / "best_validation.pt"
    )]
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
    source_before = {
        p.data_path: (p.data_path.stat().st_size, p.data_path.stat().st_mtime_ns)
        for p in protocols.values()
    }
    rows: list[dict] = []
    run_information: list[dict] = []
    reproduced = 0
    for protocol in protocols.values():
        bundle = protocol.evaluation_windows["test"]
        evaluate_persistence(rows, protocol, bundle)
        predictions: dict[tuple[str, int], np.ndarray] = {}
        for model_name in MODEL_NAMES:
            for seed in cfg["seeds"]:
                run_id = f"{model_name.replace(' ', '_').replace('-', '_')}_{protocol.dataset}_{seed}"
                info, prediction, _ = validate_completed_artifact(
                    protocol, model_name, seed, cfg, device, RESULTS / run_id,
                    reforward=reforward,
                )
                reproduced += int(reforward)
                predictions[(model_name, seed)] = prediction
                evaluate_prediction(
                    rows, protocol, model_name, seed, prediction, bundle,
                    artifact_reference(run_id),
                )
                run_information.append(info)
        evaluate_daily_matched(rows, protocol, bundle, predictions)
    add_persistence_skill_zero(rows)
    aggregate_neural_rows(rows, cfg["seeds"])
    add_rank_rows(rows)
    add_run_information_rows(rows, run_information)
    write_metrics(rows)
    for path, signature in {**before, **source_before}.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
            raise AssertionError(f"Protected source changed during evidence audit: {path}")
    print(json.dumps({
        "status": "evidence_recomputed", "runs": len(run_information),
        "checkpoint_predictions_reproduced": reproduced,
        "metric_rows": len(rows), "device": str(device), "training_executed": False,
    }), flush=True)


def inventory_only() -> None:
    cfg = load_config()
    protocols = preflight(cfg)
    output = {}
    for name, protocol in protocols.items():
        test = protocol.evaluation_windows["test"]
        output[name] = {
            "data_path": str(protocol.data_path),
            "original_features": protocol.original_feature_columns,
            "corrected_input_dimension": int(protocol.train_windows.x.shape[-1]),
            "train_h144_windows": len(protocol.train_windows.x),
            "validation_h144_windows": len(protocol.validation_windows.x),
            "test_horizon_eligible_origins": len(test.x),
            "daylight_threshold_kw": protocol.daylight_threshold,
            "horizon_counts": {
                str(h): {
                    "origins": int((test.target_valid[:, :h].all(axis=1) & np.isfinite(test.last_power)).sum()),
                    "valid_targets": int(h * (test.target_valid[:, :h].all(axis=1) & np.isfinite(test.last_power)).sum()),
                    "daylight_targets": int(((test.y_raw[:, :h] > protocol.daylight_threshold) &
                                              np.repeat((test.target_valid[:, :h].all(axis=1) &
                                                         np.isfinite(test.last_power))[:, None], h, axis=1)).sum()),
                } for h in cfg["evaluation_horizons"]
            },
        }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--skip-reforward", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.inventory_only:
        inventory_only()
    elif arguments.evidence_only:
        evidence_only(reforward=not arguments.skip_reforward)
    else:
        run_all()

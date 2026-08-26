"""Read-only Stage B1 failure-attribution audit.

This program never trains or mutates a model.  It reads the existing B1
prepared artifact, validation predictions, checkpoints, and epoch logs, then
writes one long-form diagnostic CSV in this directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

MODELS = ("HISTORY_ONLY", "RAW_NWP", "AGE_LEAD_RELIABILITY")
SEEDS = (42, 43, 44)
HORIZONS = (12, 48, 96, 144)
QUANTILES = (0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)
OUT_COLUMNS = (
    "section", "model", "seed", "split", "scope", "horizon_steps",
    "feature", "metric", "value", "unit", "count", "source_path", "notes",
)


class Rows:
    def __init__(self) -> None:
        self.data: list[dict[str, Any]] = []

    def add(self, section: str, metric: str, value: Any, *, model: str = "",
            seed: int | str = "", split: str = "", scope: str = "",
            horizon: int | str = "", feature: str = "", unit: str = "",
            count: int | str = "", source: str | Path = "", notes: str = "") -> None:
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            value = value.item()
        self.data.append(dict(section=section, model=model, seed=seed, split=split,
                              scope=scope, horizon_steps=horizon, feature=feature,
                              metric=metric, value=value, unit=unit, count=count,
                              source_path=str(source), notes=notes))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("b1_readonly", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_values(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray,
                  target_range: float) -> dict[str, float | int]:
    y = labels[mask].astype(np.float64)
    p = predictions[mask].astype(np.float64)
    error = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "rmse_kw": rmse,
        "mae_kw": float(np.mean(np.abs(error))),
        "range_nrmse": rmse / target_range,
        "r2": 1.0 - float(np.sum(error ** 2)) / float(denominator) if denominator > 0 else math.nan,
        "valid_target_count": int(mask.sum()),
    }


def future(values: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    offsets = np.arange(1, horizon + 1, dtype=np.int64)
    return values[origins[:, None] + offsets[None, :]]


def qname(q: float) -> str:
    return "p" + (str(q * 100).rstrip("0").rstrip(".") if q else "0")


def add_distribution(rows: Rows, model: str, seed: int, horizon: int,
                     labels: np.ndarray, predictions: np.ndarray, target_min: float,
                     target_max: float, source: Path) -> None:
    y = labels[:, :horizon].astype(np.float64).ravel()
    p = predictions[:, :horizon].astype(np.float64).ravel()
    ae = np.abs(p - y)
    se = (p - y) ** 2
    for name, values, unit in (("prediction", p, "kW"), ("label", y, "kW"),
                               ("absolute_error", ae, "kW"), ("squared_error", se, "kW^2")):
        rows.add("tail_distribution", f"{name}_min", float(np.min(values)), model=model, seed=seed,
                 split="validation", horizon=horizon, unit=unit, count=len(values), source=source)
        rows.add("tail_distribution", f"{name}_max", float(np.max(values)), model=model, seed=seed,
                 split="validation", horizon=horizon, unit=unit, count=len(values), source=source)
        rows.add("tail_distribution", f"{name}_mean", float(np.mean(values)), model=model, seed=seed,
                 split="validation", horizon=horizon, unit=unit, count=len(values), source=source)
        rows.add("tail_distribution", f"{name}_sd", float(np.std(values)), model=model, seed=seed,
                 split="validation", horizon=horizon, unit=unit, count=len(values), source=source)
        for q, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
            rows.add("tail_distribution", f"{name}_{qname(q)}", float(value), model=model, seed=seed,
                     split="validation", horizon=horizon, unit=unit, count=len(values), source=source)
    ratios = {
        "prediction_negative_ratio": np.mean(p < 0),
        "prediction_outside_train_minmax_ratio": np.mean((p < target_min) | (p > target_max)),
        "prediction_above_1.25_train_max_ratio": np.mean(p > 1.25 * target_max),
        "prediction_above_1.5_train_max_ratio": np.mean(p > 1.5 * target_max),
        "prediction_above_2_train_max_ratio": np.mean(p > 2.0 * target_max),
    }
    for metric, value in ratios.items():
        rows.add("physical_range", metric, float(value), model=model, seed=seed, split="validation",
                 horizon=horizon, unit="fraction", count=len(p), source=source)
    total_sse = float(se.sum())
    ordered = np.sort(se)[::-1]
    for pct in (0.001, 0.01, 0.05):
        n = max(1, int(math.ceil(len(ordered) * pct)))
        contribution = float(ordered[:n].sum() / total_sse) if total_sse > 0 else math.nan
        rows.add("error_concentration", f"top_{pct*100:g}pct_sse_contribution", contribution,
                 model=model, seed=seed, split="validation", horizon=horizon, unit="fraction",
                 count=n, source=source)


def add_top_errors(rows: Rows, model: str, seed: int, labels: np.ndarray,
                   predictions: np.ndarray, origins_ns: np.ndarray, source: Path) -> None:
    ae = np.abs(predictions.astype(np.float64) - labels.astype(np.float64))
    flat = ae.ravel()
    indices = np.argpartition(flat, -20)[-20:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    for rank, flat_index in enumerate(indices, 1):
        window, step0 = np.unravel_index(int(flat_index), ae.shape)
        payload = {"rank": rank, "origin": str(pd.Timestamp(int(origins_ns[window]))),
                   "horizon_step": int(step0 + 1), "label_kw": float(labels[window, step0]),
                   "prediction_kw": float(predictions[window, step0]),
                   "absolute_error_kw": float(ae[window, step0])}
        rows.add("top_absolute_errors", "record", json.dumps(payload, ensure_ascii=False),
                 model=model, seed=seed, split="validation", horizon=int(step0 + 1),
                 unit="json", count=1, source=source)


def scenario_masks(data: Any, cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    origins = data["validation_origins"]
    stamps = pd.DatetimeIndex(pd.to_datetime(data["times_ns"][origins]))
    power = data["power"]
    ghi = data["ground_ghi"]
    origin_ghi = ghi[origins]
    past_change = np.array([np.nanmax(np.abs(np.diff(power[o-12:o+1]))) for o in origins])
    train_origins = data["train_origins"]
    train_change = np.array([np.nanmax(np.abs(np.diff(power[o-12:o+1]))) for o in train_origins])
    q50, q90 = np.nanquantile(train_change, [0.5, 0.9])
    masks: dict[str, np.ndarray] = {
        "daylight_origin": np.isfinite(origin_ghi) & (origin_ghi >= cfg["daylight_ghi_threshold_wm2"]),
        "night_origin": ~(np.isfinite(origin_ghi) & (origin_ghi >= cfg["daylight_ghi_threshold_wm2"])),
        "past_variation_clear_stable": past_change <= q50,
        "past_variation_medium": (past_change > q50) & (past_change < q90),
        "past_variation_high": past_change >= q90,
        "nwp_valid": data["validation_nwp_valid"].astype(bool),
        "nwp_invalid_or_fallback": ~data["validation_nwp_valid"].astype(bool),
        "fallback_cycle_1": data["validation_fallback"].astype(int) == 1,
        "no_fallback": data["validation_fallback"].astype(int) == 0,
    }
    for month in range(1, 13):
        masks[f"month_{month:02d}"] = np.asarray(stamps.month == month)
    for hour in range(24):
        masks[f"origin_hour_{hour:02d}"] = np.asarray(stamps.hour == hour)
    origin_break = np.r_[True, np.diff(origins) != 1]
    segment_ids = np.cumsum(origin_break)
    for segment in np.unique(segment_ids):
        masks[f"segment_{int(segment):03d}"] = segment_ids == segment
    return masks, segment_ids


def add_scenarios(rows: Rows, model: str, seed: int, labels: np.ndarray,
                  predictions: np.ndarray, masks: dict[str, np.ndarray], target_range: float,
                  source: Path) -> None:
    for horizon in HORIZONS:
        all_se = (predictions[:, :horizon].astype(np.float64) - labels[:, :horizon].astype(np.float64)) ** 2
        total_sse = float(all_se.sum())
        for scope, window_mask in masks.items():
            if not window_mask.any():
                continue
            point_mask = np.broadcast_to(window_mask[:, None], (len(window_mask), horizon))
            values = metric_values(labels[:, :horizon], predictions[:, :horizon], point_mask, target_range)
            for metric, value in values.items():
                rows.add("scenario_error", metric, value, model=model, seed=seed, split="validation",
                         scope=scope, horizon=horizon, unit=("kW" if metric in ("rmse_kw", "mae_kw") else ""),
                         count=int(window_mask.sum()), source=source)
            rows.add("scenario_error", "sse_contribution", float(all_se[window_mask].sum() / total_sse),
                     model=model, seed=seed, split="validation", scope=scope, horizon=horizon,
                     unit="fraction", count=int(window_mask.sum()), source=source)


def add_error_predictability(rows: Rows, model: str, seed: int, labels: np.ndarray,
                             predictions: np.ndarray, data: Any, cfg: dict[str, Any], source: Path) -> None:
    window_mae = np.mean(np.abs(predictions.astype(np.float64) - labels.astype(np.float64)), axis=1)
    origins = data["validation_origins"]
    for index, feature in enumerate(cfg["history_features"]):
        observed_at_origin = data["history_features"][origins, index].astype(np.float64)
        correlation = pd.Series(window_mae).corr(pd.Series(observed_at_origin), method="spearman")
        rows.add("error_predictability", "spearman_window_mae_vs_origin_feature", float(correlation),
                 model=model, seed=seed, split="validation", feature=feature, unit="rho",
                 count=len(window_mae), source=source,
                 notes="feature is Train-scaled value observed at forecast origin")


def add_feature_drift(rows: Rows, frame: pd.DataFrame, cfg: dict[str, Any], data: Any,
                      source: Path) -> None:
    train = (frame.index >= pd.Timestamp(cfg["splits"]["train"][0])) & (frame.index < pd.Timestamp(cfg["splits"]["train"][1]))
    validation = (frame.index >= pd.Timestamp(cfg["splits"]["validation"][0])) & (frame.index < pd.Timestamp(cfg["splits"]["validation"][1]))
    for column_index, feature in enumerate(cfg["history_features"]):
        tr = frame.loc[train, feature].to_numpy(np.float64)
        va = frame.loc[validation, feature].to_numpy(np.float64)
        trf, vaf = tr[np.isfinite(tr)], va[np.isfinite(va)]
        if len(trf) == 0 or len(vaf) == 0:
            rows.add("feature_drift", "finite_data_status", "NO_FINITE_TRAIN" if len(trf) == 0 else "NO_FINITE_VALIDATION",
                     split="train_validation", feature=feature, count=f"{len(trf)}/{len(vaf)}", source=source)
            for split, values in (("train", tr), ("validation", va)):
                rows.add("feature_drift", "missing_ratio", float(np.mean(~np.isfinite(values))), split=split,
                         feature=feature, unit="fraction", count=len(values), source=source)
            continue
        tr_min, tr_max = float(np.min(trf)), float(np.max(trf))
        center, scale = float(data["history_center"][column_index]), float(data["history_scale"][column_index])
        for split, values, finite in (("train", tr, trf), ("validation", va, vaf)):
            stats = {"missing_ratio": float(np.mean(~np.isfinite(values))), "mean": float(np.mean(finite)),
                     "sd": float(np.std(finite)), "p1": float(np.quantile(finite, 0.01)),
                     "p50": float(np.quantile(finite, 0.5)), "p99": float(np.quantile(finite, 0.99)),
                     "min": float(np.min(finite)), "max": float(np.max(finite))}
            for metric, value in stats.items():
                rows.add("feature_drift", metric, value, split=split, feature=feature,
                         unit="fraction" if metric == "missing_ratio" else "source_unit",
                         count=len(values), source=source)
        z = np.abs((vaf - center) / scale)
        rows.add("feature_drift", "validation_outside_train_minmax_ratio",
                 float(np.mean((vaf < tr_min) | (vaf > tr_max))), split="validation", feature=feature,
                 unit="fraction", count=len(vaf), source=source)
        for threshold in (3, 5, 10):
            rows.add("feature_drift", f"validation_abs_z_gt_{threshold}_ratio", float(np.mean(z > threshold)),
                     split="validation", feature=feature, unit="fraction", count=len(vaf), source=source)
        for split, indices in (("train", np.flatnonzero(train)), ("validation", np.flatnonzero(validation))):
            missing_indicator = data["history_features"][indices, 13 + column_index]
            rows.add("missingness_indicator", "indicator_mean", float(np.mean(missing_indicator)), split=split,
                     feature=feature, unit="fraction", count=len(indices), source=source)


def checkpoint_forward_check(rows: Rows, b1: Any, cfg: dict[str, Any], data: Any,
                             model_name: str, seed: int, run_dir: Path, artifact: Any) -> None:
    checkpoint = torch.load(run_dir / "best_validation.pt", map_location="cpu", weights_only=True)
    logs = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    best_log = min(logs, key=lambda row: row["validation_mse"])
    rows.add("checkpoint", "checkpoint_epoch", int(checkpoint["epoch"]), model=model_name, seed=seed,
             source=run_dir / "best_validation.pt")
    rows.add("checkpoint", "log_best_epoch", int(best_log["epoch"]), model=model_name, seed=seed,
             source=run_dir / "epochs.jsonl")
    rows.add("checkpoint", "checkpoint_validation_mse", float(checkpoint["validation_mse"]), model=model_name,
             seed=seed, source=run_dir / "best_validation.pt")
    rows.add("checkpoint", "log_best_validation_mse", float(best_log["validation_mse"]), model=model_name,
             seed=seed, source=run_dir / "epochs.jsonl")
    model = b1.ForecastModel(model_name, data["history_features"].shape[1], cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = b1.ForecastDataset(data, "validation", cfg)
    sample_indices = np.unique(np.linspace(0, len(dataset) - 1, 32, dtype=int))
    batches = [dataset[int(index)] for index in sample_indices]
    fields = [torch.stack([batch[i] for batch in batches]) for i in range(5)]
    with torch.inference_mode():
        scaled = model(*fields).numpy()
    physical = scaled * float(data["target_range"]) + float(data["target_min"])
    saved = artifact["predictions"][sample_indices]
    difference = np.abs(physical - saved)
    rows.add("checkpoint", "sample_forward_max_abs_difference", float(difference.max()), model=model_name,
             seed=seed, unit="kW", count=difference.size, source=run_dir / "best_validation.pt",
             notes="32 deterministic Validation origins; CPU re-forward")
    rows.add("checkpoint", "sample_forward_mean_abs_difference", float(difference.mean()), model=model_name,
             seed=seed, unit="kW", count=difference.size, source=run_dir / "best_validation.pt")
    del model, checkpoint, fields, batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output != Path(__file__).resolve().parent:
        raise AssertionError("output must be this audit directory")
    b1_root = args.b1_root.resolve()
    results = b1_root / "results"
    required = [results / "prepared_data.npz", b1_root / "config.json", b1_root / "run_nwp_minimal_screen.py"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("B1 local results/prepared artifact unavailable")
    cfg = json.loads((b1_root / "config.json").read_text(encoding="utf-8"))
    if cfg["splits"]["validation"][1] != "2023-01-01 00:00:00":
        raise AssertionError("unexpected Validation boundary")
    if "2023" in str(results / "prepared_data.npz"):
        raise AssertionError("unexpected 2023 path")
    source_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in required}
    b1 = load_module(b1_root / "run_nwp_minimal_screen.py")
    rows = Rows()
    data = np.load(results / "prepared_data.npz", allow_pickle=False)
    if int(data["times_ns"].max()) >= int(pd.Timestamp("2023-01-01").value):
        raise AssertionError("prepared artifact contains sealed 2023")
    target_min = float(data["target_min"]); target_range = float(data["target_range"]); target_max = target_min + target_range
    rows.add("scaling", "target_min", target_min, split="train", unit="kW", source=results / "prepared_data.npz")
    rows.add("scaling", "target_max", target_max, split="train", unit="kW", source=results / "prepared_data.npz")
    rows.add("scaling", "target_range", target_range, split="train", unit="kW", source=results / "prepared_data.npz")
    rows.add("scaling", "forward_formula", "(y_kw-target_min)/target_range", split="train", source=b1_root / "run_nwp_minimal_screen.py")
    rows.add("scaling", "inverse_formula", "scaled_prediction*target_range+target_min", split="validation", source=b1_root / "run_nwp_minimal_screen.py")

    common_labels = common_origins = common_nwp_valid = None
    masks, segment_ids = scenario_masks(data, cfg)
    history_rmse: dict[tuple[int, int], float] = {}
    for model_name in MODELS:
        for seed in SEEDS:
            run_dir = results / model_name / f"seed_{seed}"
            paths = [run_dir / name for name in ("epochs.jsonl", "best_validation.pt", "last.pt", "validation_H144.npz")]
            if not all(path.is_file() for path in paths):
                raise FileNotFoundError(f"incomplete run {model_name} seed {seed}")
            artifact_path = run_dir / "validation_H144.npz"
            artifact = np.load(artifact_path, allow_pickle=False)
            predictions = artifact["predictions"]
            labels = artifact["labels"]
            origins_ns = artifact["forecast_origin_timestamp_ns"]
            nwp_valid = artifact["nwp_valid"]
            if predictions.shape != (len(data["validation_origins"]), 144) or labels.shape != predictions.shape:
                raise AssertionError(f"shape mismatch {model_name} {seed}")
            if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
                raise AssertionError(f"non-finite prediction/label {model_name} {seed}")
            if common_labels is None:
                common_labels, common_origins, common_nwp_valid = labels.copy(), origins_ns.copy(), nwp_valid.copy()
            else:
                if not np.array_equal(common_labels, labels): raise AssertionError("labels differ across runs")
                if not np.array_equal(common_origins, origins_ns): raise AssertionError("origins differ across runs")
                if not np.array_equal(common_nwp_valid, nwp_valid): raise AssertionError("NWP-valid masks differ across runs")
            rows.add("artifact", "run_complete", True, model=model_name, seed=seed, split=str(artifact["split"]),
                     count=len(labels), source=artifact_path)
            rows.add("artifact", "prediction_shape", str(tuple(predictions.shape)), model=model_name, seed=seed, source=artifact_path)
            rows.add("artifact", "labels_equal_common", True, model=model_name, seed=seed, source=artifact_path)
            rows.add("artifact", "origins_equal_common", True, model=model_name, seed=seed, source=artifact_path)
            rows.add("artifact", "nwp_mask_equal_common", True, model=model_name, seed=seed, source=artifact_path)
            checkpoint_forward_check(rows, b1, cfg, data, model_name, seed, run_dir, artifact)
            logs = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            for log in logs:
                for metric in ("train_mse", "validation_mse", "seconds"):
                    rows.add("training_log", metric, log[metric], model=model_name, seed=seed, split="epoch",
                             horizon=log["epoch"], unit="seconds" if metric == "seconds" else "scaled_MSE",
                             source=run_dir / "epochs.jsonl")
            best = min(logs, key=lambda row: row["validation_mse"])
            rows.add("training_summary", "initial_validation_mse", logs[0]["validation_mse"], model=model_name, seed=seed, source=run_dir / "epochs.jsonl")
            rows.add("training_summary", "best_validation_mse", best["validation_mse"], model=model_name, seed=seed, horizon=best["epoch"], source=run_dir / "epochs.jsonl")
            rows.add("training_summary", "actual_epochs", len(logs), model=model_name, seed=seed, source=run_dir / "epochs.jsonl")
            for horizon in HORIZONS:
                add_distribution(rows, model_name, seed, horizon, labels, predictions, target_min, target_max, artifact_path)
                full = np.ones((len(labels), horizon), dtype=bool)
                metrics = metric_values(labels[:, :horizon], predictions[:, :horizon], full, target_range)
                for metric, value in metrics.items():
                    rows.add("validation_metrics", metric, value, model=model_name, seed=seed, split="validation",
                             scope="regular_full_timeline", horizon=horizon,
                             unit="kW" if metric in ("rmse_kw", "mae_kw") else "", count=int(full.sum()), source=artifact_path)
                if model_name == "HISTORY_ONLY": history_rmse[(seed, horizon)] = float(metrics["rmse_kw"])
            add_top_errors(rows, model_name, seed, labels, predictions, origins_ns, artifact_path)
            add_scenarios(rows, model_name, seed, labels, predictions, masks, target_range, artifact_path)
            add_error_predictability(rows, model_name, seed, labels, predictions, data, cfg, artifact_path)
            del artifact, predictions, labels, origins_ns, nwp_valid

    assert common_labels is not None and common_origins is not None
    expected_labels = future(data["power"], data["validation_origins"], 144)
    expected_origins = data["times_ns"][data["validation_origins"]]
    if not np.array_equal(common_labels, expected_labels): raise AssertionError("saved labels not from prepared origins")
    if not np.array_equal(common_origins, expected_origins): raise AssertionError("saved timestamps not from prepared origins")
    persistence = np.repeat(data["power"][data["validation_origins"]][:, None], 144, axis=1)
    if not np.isfinite(persistence).all(): raise AssertionError("persistence non-finite")
    future_ghi = future(data["ground_ghi"], data["validation_origins"], 144)
    previous = np.concatenate([data["power"][data["validation_origins"]][:, None], common_labels[:, :-1]], axis=1)
    change = np.abs(common_labels - previous)
    daylight = np.isfinite(future_ghi) & (future_ghi >= cfg["daylight_ghi_threshold_wm2"])
    high_change = daylight & (change >= float(data["high_change_threshold"]))
    persistence_rmse: dict[int, float] = {}
    for horizon in HORIZONS:
        for scope, mask in (("regular_full_timeline", np.ones((len(common_labels), horizon), bool)),
                            ("daylight", daylight[:, :horizon]), ("high_change_daylight", high_change[:, :horizon])):
            result = metric_values(common_labels[:, :horizon], persistence[:, :horizon], mask, target_range)
            for metric, value in result.items():
                rows.add("persistence", metric, value, model="PERSISTENCE_LAST", seed="NA", split="validation",
                         scope=scope, horizon=horizon, unit="kW" if metric in ("rmse_kw", "mae_kw") else "",
                         count=int(mask.sum()), source=results / "prepared_data.npz",
                         notes="last observed Active_Power at forecast origin repeated through H144")
            if scope == "regular_full_timeline": persistence_rmse[horizon] = float(result["rmse_kw"])
    for seed in SEEDS:
        for horizon in HORIZONS:
            rows.add("persistence_comparison", "history_minus_persistence_rmse", history_rmse[(seed, horizon)] - persistence_rmse[horizon],
                     model="HISTORY_ONLY", seed=seed, split="validation", horizon=horizon, unit="kW")
            rows.add("persistence_comparison", "history_relative_to_persistence_rmse_ratio", history_rmse[(seed, horizon)] / persistence_rmse[horizon],
                     model="HISTORY_ONLY", seed=seed, split="validation", horizon=horizon, unit="ratio")

    frame, pv_info = b1.read_pv_train_validation(cfg)
    add_feature_drift(rows, frame, cfg, data, Path(cfg["pv_file"]))
    for metric, value in pv_info.items(): rows.add("source_readonly", metric, value, source=cfg["pv_file"])
    rows.add("scenario_definition", "past_variation_groups", "Train p50/p90 of maximum absolute PV change over previous 12 steps", source=results / "prepared_data.npz")
    rows.add("scenario_definition", "daylight_night", f"origin GHI >= {cfg['daylight_ghi_threshold_wm2']} W/m2", source=results / "prepared_data.npz")
    rows.add("scenario_definition", "segment_count", int(segment_ids.max()), split="validation", unit="count")
    rows.add("scenario_definition", "nwp_invalid_origin_count", int((~data["validation_nwp_valid"]).sum()), split="validation", unit="count")

    code_diffs = [
        ("input_features", "B1: 13 raw history variables + 13 missing indicators + 4 time features; includes cumulative Active_Energy. Verified MEAN_ONLY: historical Active_Power, time encodings and MB channel means/validity."),
        ("target_scaling", "B1: Train min-range; verified MEAN_ONLY: Train mean/std; clean benchmark: Train-only MinMaxScaler."),
        ("split", "B1: Train 2021-03-23..2021-12-31, Validation 2022; verified TRAJECTORY_ONLY: within-2022 Jan-Aug/Sept-Oct/Nov-Dec."),
        ("horizon", "B1: H144 (12 h); verified MEAN_ONLY: H12 (1 h); clean benchmark: H144."),
        ("model_parameters", "B1 HISTORY_ONLY 683,856 reported; verified architecture uses same depthwise/pointwise ModernTCN family but input dimension and output horizon differ."),
        ("output_range", "B1 and verified implementations use an unconstrained linear output head; B1 applies one inverse min-range transform."),
        ("early_stopping", "B1: Validation scaled MSE; MEAN_ONLY: Validation physical RMSE; clean benchmark: Validation MSE."),
        ("missingness", "B1: Train median fill + explicit per-feature missing indicators; clean protocol uses Train-only KNN/IF/scalers."),
    ]
    comparison_sources = f"{b1_root / 'run_nwp_minimal_screen.py'} | {b1_root.parent / 'asoc_multirate_information_screen/run_information_screen.py'}"
    for feature, text in code_diffs:
        rows.add("code_difference", "B1_vs_verified", text, feature=feature, source=comparison_sources,
                 notes="Code-level comparison only; RMSE across different years/tasks is not compared")

    for path, before in source_stats.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != before: raise AssertionError(f"source changed: {path}")
    if any("2023" in str(row["source_path"]) for row in rows.data): raise AssertionError("2023 source accessed")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "B1_FAILURE_DIAGNOSTICS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader(); writer.writerows(rows.data)
    reread = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    if len(reread) != len(rows.data): raise AssertionError("CSV row-count mismatch")
    allowed = {Path(__file__).resolve(), csv_path.resolve(), (output / "REPORT.md").resolve()}
    actual = {path.resolve() for path in output.iterdir() if path.is_file()}
    if not actual.issubset(allowed): raise AssertionError(f"unexpected output files: {actual - allowed}")
    summary = {"rows": len(rows.data), "runs": 9, "validation_windows": len(common_labels),
               "persistence_h144_rmse": persistence_rmse[144],
               "history_h144_rmse": {str(seed): history_rmse[(seed, 144)] for seed in SEEDS},
               "source_unchanged": True, "sealed_2023_accessed": False, "training_called": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

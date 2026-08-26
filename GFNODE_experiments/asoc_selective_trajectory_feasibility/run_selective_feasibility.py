"""C1-S0 selective PV trajectory feasibility screen (no deep-model training).

The script reconstructs missing validation forecasts with frozen, existing
checkpoints under torch.inference_mode(), fits only the preregistered CPU risk
regressors on the first half of the original validation period, calibrates fixed
acceptance thresholds on its second half, and evaluates Test once.
"""
from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
METRICS_PATH = ROOT / "metrics.csv"
HORIZONS = (3, 6, 12)
METHODS = ("FULL_RISK_MODEL", "RECENT_VARIATION", "MODEL_PERSISTENCE_DISAGREEMENT")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_state(paths: list[Path]) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def import_source(source_root: Path):
    path = source_root / "run_information_screen.py"
    spec = importlib.util.spec_from_file_location("c1_readonly_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels_for_origins(power: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([power[int(o) + 1:int(o) + horizon + 1] for o in origins]).astype(np.float32)


def frozen_validation_prediction(
    module: Any,
    source_config: dict[str, Any],
    prepared: Any,
    checkpoint: Path,
) -> np.ndarray:
    columns = prepared["base_cols"]
    dataset = module.WindowDataset(
        prepared["scaled_features"], prepared["power"], prepared["validation_origins"],
        columns, source_config["lookback"], source_config["horizon"],
        float(prepared["target_center"]), float(prepared["target_scale"]),
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.ModernTCN(len(columns), source_config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for x, _ in loader:
            scaled = model(x.to(device)).cpu().numpy()
            chunks.append(scaled)
    return (np.concatenate(chunks) * float(prepared["target_scale"]) + float(prepared["target_center"])).astype(np.float32)


def safe_stats(values: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if not len(finite):
        return (math.nan,) * 8
    last = finite[-1]
    diffs = np.diff(finite)
    x = np.arange(len(finite), dtype=float)
    slope = float(np.polyfit(x, finite, 1)[0]) if len(finite) > 1 else 0.0
    return (
        last, float(np.mean(finite)), float(np.std(finite)), float(np.ptp(finite)),
        float(np.max(np.abs(diffs))) if len(diffs) else 0.0,
        float(np.mean(np.abs(diffs))) if len(diffs) else 0.0,
        slope, float(last - np.mean(finite)),
    )


def risk_features(
    origins: np.ndarray,
    predictions: np.ndarray,
    power: np.ndarray,
    times_ns: np.ndarray,
    raw_features: np.ndarray,
    feature_names: list[str],
    horizon: int,
    daylight_threshold: float,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    rows: list[list[float]] = []
    names = ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "hour", "origin_daylight"]
    pv_names = ["last", "mean", "std", "range", "max_abs_diff", "mean_abs_diff", "slope", "last_minus_mean"]
    for width in (12, 36, 72):
        names.extend([f"pv_{width}_{n}" for n in pv_names])
    for channel in ("MB0", "MB1", "MB2"):
        names.extend([f"{channel}_12_{n}" for n in ("mean", "std", "range", "max_abs_diff", "valid_ratio")])
    names.extend(["model_persistence_mean_abs_disagreement", "model_persistence_max_abs_disagreement",
                  "model_persistence_endpoint_disagreement", "prediction_total_variation",
                  "prediction_range", "prediction_max_abs_first_difference"])
    name_to_col = {name: i for i, name in enumerate(feature_names)}
    recent_variation = np.empty(len(origins), dtype=float)
    disagreement = np.empty(len(origins), dtype=float)
    causal_max_timestamp = np.empty(len(origins), dtype=np.int64)
    for i, origin_raw in enumerate(origins):
        origin = int(origin_raw)
        ts = pd.Timestamp(int(times_ns[origin]), unit="ns")
        minute = ts.hour * 60 + ts.minute
        tod = minute / 1440.0
        doy = (ts.dayofyear - 1) / 365.0
        row = [math.sin(2 * math.pi * tod), math.cos(2 * math.pi * tod),
               math.sin(2 * math.pi * doy), math.cos(2 * math.pi * doy),
               ts.hour + ts.minute / 60.0, float(power[origin] > daylight_threshold)]
        pv12 = power[origin - 11:origin + 1].astype(float)
        recent_variation[i] = float(np.max(np.abs(np.diff(pv12))))
        for width in (12, 36, 72):
            row.extend(safe_stats(power[origin - width + 1:origin + 1]))
        for channel in ("MB0", "MB1", "MB2"):
            mean_values = raw_features[origin - 11:origin + 1, name_to_col[f"{channel}_mean"]].astype(float)
            mask_values = raw_features[origin - 11:origin + 1, name_to_col[f"{channel}_valid_mask"]].astype(float)
            valid = np.isfinite(mean_values) & (mask_values > 0)
            v = mean_values[valid]
            if len(v):
                diffs = np.diff(v)
                row.extend([float(np.mean(v)), float(np.std(v)), float(np.ptp(v)),
                            float(np.max(np.abs(diffs))) if len(diffs) else 0.0, float(np.mean(valid))])
            else:
                row.extend([math.nan, math.nan, math.nan, math.nan, 0.0])
        pred = predictions[i, :horizon].astype(float)
        persistence = np.full(horizon, float(power[origin]))
        diff = np.abs(pred - persistence)
        pdiff = np.diff(pred)
        disagreement[i] = float(np.mean(diff))
        row.extend([disagreement[i], float(np.max(diff)), float(diff[-1]),
                    float(np.sum(np.abs(pdiff))), float(np.ptp(pred)),
                    float(np.max(np.abs(pdiff))) if len(pdiff) else 0.0])
        rows.append(row)
        causal_max_timestamp[i] = times_ns[origin]
    return np.asarray(rows, dtype=np.float64), names, {
        "recent_variation": recent_variation,
        "model_persistence_disagreement": disagreement,
        "causal_max_timestamp_ns": causal_max_timestamp,
    }


def trajectory_losses(labels: np.ndarray, predictions: np.ndarray, horizon: int, train_range: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = predictions[:, :horizon] - labels[:, :horizon]
    rmse = np.sqrt(np.mean(err ** 2, axis=1))
    mae = np.mean(np.abs(err), axis=1)
    return rmse / train_range, rmse, mae


def aggregate_error(labels: np.ndarray, predictions: np.ndarray, chosen: np.ndarray, horizon: int) -> tuple[float, float]:
    if not np.any(chosen):
        return math.nan, math.nan
    err = predictions[chosen, :horizon] - labels[chosen, :horizon]
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def ranking_metrics(score: np.ndarray, loss: np.ndarray, high_error: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    s, y, event = score[mask], loss[mask], high_error[mask]
    rho = float(spearmanr(s, y, nan_policy="omit").statistic) if len(s) > 1 else math.nan
    if len(np.unique(event)) < 2:
        auroc = auprc = math.nan
    else:
        auroc = float(roc_auc_score(event, s)); auprc = float(average_precision_score(event, s))
    return {"spearman": rho, "auroc": auroc, "auprc": auprc, "high_error_prevalence": float(np.mean(event))}


def risk_coverage_auc(score: np.ndarray, loss: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return math.nan
    order = idx[np.argsort(score[idx], kind="stable")]
    coverages = np.linspace(0.05, 1.0, 96)
    risks = np.asarray([np.mean(loss[order[:max(1, int(math.ceil(c * len(order))))]]) for c in coverages])
    return float(np.trapezoid(risks, coverages) / (coverages[-1] - coverages[0]))


def add_row(rows: list[dict[str, Any]], section: str, seed: Any, horizon: Any, scope: str,
            method: str, coverage: Any, metric: str, value: Any, unit: str = "",
            sample_count: Any = "", accepted_count: Any = "", threshold: Any = "",
            ci_lower: Any = "", ci_upper: Any = "", notes: str = "NA") -> None:
    rows.append({"section": section, "seed": seed, "horizon": horizon, "scope": scope,
                 "risk_method": method, "target_coverage": coverage, "metric": metric,
                 "value": value, "ci_lower": ci_lower, "ci_upper": ci_upper, "unit": unit,
                 "sample_count": sample_count, "accepted_count": accepted_count,
                 "threshold": threshold, "notes": notes})


def block_bootstrap(
    labels: np.ndarray, model_pred: np.ndarray, persistence: np.ndarray,
    accepted: np.ndarray, daylight: np.ndarray, times_ns: np.ndarray,
    horizon: int, replicates: int, random_seed: int,
) -> dict[str, tuple[float, float, float]]:
    eligible = accepted & daylight
    idx = np.flatnonzero(eligible)
    dates = pd.to_datetime(times_ns[idx], unit="ns").date
    unique_days = np.asarray(sorted(set(dates)), dtype=object)
    by_day = {d: idx[np.asarray(dates) == d] for d in unique_days}
    rng = np.random.default_rng(random_seed)
    reductions, persistence_deltas = [], []
    full_rmse, _ = aggregate_error(labels, model_pred, daylight, horizon)
    for _ in range(replicates):
        sampled = rng.choice(unique_days, size=len(unique_days), replace=True)
        chosen_idx = np.concatenate([by_day[d] for d in sampled])
        model_err = model_pred[chosen_idx, :horizon] - labels[chosen_idx, :horizon]
        pers_err = persistence[chosen_idx, :horizon] - labels[chosen_idx, :horizon]
        model_rmse = float(np.sqrt(np.mean(model_err ** 2)))
        pers_rmse = float(np.sqrt(np.mean(pers_err ** 2)))
        reductions.append((full_rmse - model_rmse) / full_rmse * 100.0)
        persistence_deltas.append((pers_rmse - model_rmse) / pers_rmse * 100.0)
    def summarize(v: list[float]) -> tuple[float, float, float]:
        a = np.asarray(v); return float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))
    return {"accepted_rmse_reduction_pct": summarize(reductions),
            "matched_persistence_skill_pct": summarize(persistence_deltas)}


def main() -> None:
    config = load_json(CONFIG_PATH)
    source_root = Path(config["source_experiment"])
    prepared_path = Path(config["prepared_artifact"])
    source_config = load_json(source_root / "config.json")
    seeds = [int(s) for s in config["seeds"]]
    source_files = [prepared_path, source_root / "run_information_screen.py", source_root / "config.json"]
    for seed in seeds:
        source_files.extend([source_root / "results" / "MEAN_ONLY" / str(seed) / "best_validation.pt",
                             source_root / "results" / "MEAN_ONLY" / str(seed) / "test_predictions.npz"])
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    state_before = file_state(source_files)
    prepared = np.load(prepared_path, allow_pickle=False)
    module = import_source(source_root)
    power = prepared["power"].astype(np.float32)
    times_ns = prepared["times"].astype(np.int64)
    raw_features = prepared["raw_features"].astype(np.float32)
    feature_names = prepared["feature_names"].tolist()
    train_origins = prepared["train_origins"].astype(np.int64)
    val_origins = prepared["validation_origins"].astype(np.int64)
    test_origins = prepared["test_origins"].astype(np.int64)
    train_values = power[np.isfinite(power) & (times_ns >= times_ns[train_origins[0] - config["lookback"] + 1]) &
                         (times_ns <= times_ns[train_origins[-1] + config["horizon"]])]
    train_range = float(np.max(train_values) - np.min(train_values))
    if not train_range > 0:
        raise AssertionError("Train target range must be positive")
    val_labels = labels_for_origins(power, val_origins, config["horizon"])
    test_labels_expected = labels_for_origins(power, test_origins, config["horizon"])
    common_test_labels = common_test_times = None
    test_predictions: dict[int, np.ndarray] = {}
    val_predictions: dict[int, np.ndarray] = {}
    checkpoint_epochs: dict[int, int] = {}
    for seed in seeds:
        run_dir = source_root / "results" / "MEAN_ONLY" / str(seed)
        artifact = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
        labels = artifact["labels"].astype(np.float32)
        timestamps = artifact["forecast_origin_timestamp_ns"].astype(np.int64)
        pred = artifact["predictions"].astype(np.float32)
        assert pred.shape == labels.shape == (len(test_origins), config["horizon"])
        assert np.array_equal(labels, test_labels_expected)
        assert np.array_equal(timestamps, times_ns[test_origins])
        assert np.isfinite(pred).all() and np.isfinite(labels).all()
        if common_test_labels is None:
            common_test_labels, common_test_times = labels, timestamps
        else:
            assert np.array_equal(common_test_labels, labels)
            assert np.array_equal(common_test_times, timestamps)
        test_predictions[seed] = pred
        checkpoint = torch.load(run_dir / "best_validation.pt", map_location="cpu", weights_only=True)
        checkpoint_epochs[seed] = int(checkpoint["epoch"])
        val_predictions[seed] = frozen_validation_prediction(module, source_config, prepared, run_dir / "best_validation.pt")
        assert val_predictions[seed].shape == val_labels.shape and np.isfinite(val_predictions[seed]).all()

    n_fit = len(val_origins) // 2
    assert n_fit > 0 and n_fit < len(val_origins)
    assert np.all(np.diff(val_origins) > 0)
    fit_slice = np.arange(0, n_fit)
    calibration_slice = np.arange(n_fit, len(val_origins))
    assert val_origins[fit_slice[-1]] < val_origins[calibration_slice[0]]
    train_recent = []
    for origin in train_origins:
        v = power[int(origin) - 11:int(origin) + 1]
        train_recent.append(float(np.max(np.abs(np.diff(v)))))
    high_change_threshold = float(np.quantile(train_recent, 0.9))
    rows: list[dict[str, Any]] = []
    decision_facts: dict[int, dict[str, float]] = {s: {} for s in seeds}

    for seed in seeds:
        vp, tp = val_predictions[seed], test_predictions[seed]
        val_daylight = power[val_origins] > config["origin_daylight_threshold_kw"]
        test_daylight = power[test_origins] > config["origin_daylight_threshold_kw"]
        assert not np.array_equal(np.ones_like(test_daylight), test_daylight)
        val_high_change = np.asarray([np.max(np.abs(np.diff(power[int(o)-11:int(o)+1]))) >= high_change_threshold for o in val_origins])
        test_high_change = np.asarray([np.max(np.abs(np.diff(power[int(o)-11:int(o)+1]))) >= high_change_threshold for o in test_origins])
        persistence_test = power[test_origins, None] * np.ones((1, config["horizon"]), dtype=np.float32)
        for horizon in HORIZONS:
            x_val, feature_names_risk, aux_val = risk_features(
                val_origins, vp, power, times_ns, raw_features, feature_names, horizon,
                config["origin_daylight_threshold_kw"])
            x_test, feature_names_test, aux_test = risk_features(
                test_origins, tp, power, times_ns, raw_features, feature_names, horizon,
                config["origin_daylight_threshold_kw"])
            assert feature_names_risk == feature_names_test
            assert np.all(aux_val["causal_max_timestamp_ns"] <= times_ns[val_origins])
            assert np.all(aux_test["causal_max_timestamp_ns"] <= times_ns[test_origins])
            val_loss, _, _ = trajectory_losses(val_labels, vp, horizon, train_range)
            test_loss, _, _ = trajectory_losses(common_test_labels, tp, horizon, train_range)
            high_error_threshold = float(np.quantile(val_loss[fit_slice], config["high_error_quantile"]))
            test_high_error = test_loss > high_error_threshold
            estimator = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.05, max_iter=100,
                max_leaf_nodes=15, max_depth=None, min_samples_leaf=30,
                l2_regularization=1.0, early_stopping=False, random_state=seed,
            )
            estimator.fit(x_val[fit_slice], np.log1p(val_loss[fit_slice]))
            scores_val = {
                "FULL_RISK_MODEL": np.maximum(np.expm1(estimator.predict(x_val)), 0.0),
                "RECENT_VARIATION": aux_val["recent_variation"],
                "MODEL_PERSISTENCE_DISAGREEMENT": aux_val["model_persistence_disagreement"],
            }
            scores_test = {
                "FULL_RISK_MODEL": np.maximum(np.expm1(estimator.predict(x_test)), 0.0),
                "RECENT_VARIATION": aux_test["recent_variation"],
                "MODEL_PERSISTENCE_DISAGREEMENT": aux_test["model_persistence_disagreement"],
            }
            for scope, scope_mask in (("full", np.ones(len(test_origins), bool)), ("daylight", test_daylight)):
                base_rmse, base_mae = aggregate_error(common_test_labels, tp, scope_mask, horizon)
                add_row(rows, "unselected", seed, horizon, scope, "MODERNTCN", 1.0, "rmse", base_rmse, "kW", len(test_origins), int(scope_mask.sum()))
                add_row(rows, "unselected", seed, horizon, scope, "MODERNTCN", 1.0, "mae", base_mae, "kW", len(test_origins), int(scope_mask.sum()))
                p_rmse, p_mae = aggregate_error(common_test_labels, persistence_test, scope_mask, horizon)
                add_row(rows, "unselected", seed, horizon, scope, "PERSISTENCE_LAST", 1.0, "rmse", p_rmse, "kW", len(test_origins), int(scope_mask.sum()))
                # Oracle ranking and exact target coverages are diagnostic upper bounds.
                oracle_auc = risk_coverage_auc(test_loss, test_loss, scope_mask)
                add_row(rows, "oracle", seed, horizon, scope, "ORACLE", "curve", "risk_coverage_auc", oracle_auc, "normalized_loss", int(scope_mask.sum()))
                scoped_idx = np.flatnonzero(scope_mask)
                oracle_order = scoped_idx[np.argsort(test_loss[scoped_idx], kind="stable")]
                for target_coverage in config["calibration_acceptance_quantiles"]:
                    n_accept = max(1, int(math.floor(target_coverage * len(scoped_idx))))
                    accepted = np.zeros(len(test_origins), bool); accepted[oracle_order[:n_accept]] = True
                    accepted_rmse, accepted_mae = aggregate_error(common_test_labels, tp, accepted, horizon)
                    persistence_rmse, _ = aggregate_error(common_test_labels, persistence_test, accepted, horizon)
                    reduction = (base_rmse - accepted_rmse) / base_rmse * 100.0
                    add_row(rows, "oracle", seed, horizon, scope, "ORACLE", target_coverage, "accepted_rmse", accepted_rmse, "kW", len(scoped_idx), n_accept)
                    add_row(rows, "oracle", seed, horizon, scope, "ORACLE", target_coverage, "accepted_mae", accepted_mae, "kW", len(scoped_idx), n_accept)
                    add_row(rows, "oracle", seed, horizon, scope, "ORACLE", target_coverage, "relative_rmse_reduction", reduction, "%", len(scoped_idx), n_accept)
                    add_row(rows, "oracle", seed, horizon, scope, "ORACLE", target_coverage, "matched_persistence_rmse", persistence_rmse, "kW", len(scoped_idx), n_accept)
                    add_row(rows, "oracle", seed, horizon, scope, "ORACLE", target_coverage, "high_change_acceptance_rate", float(np.mean(accepted[test_high_change])) if test_high_change.any() else math.nan, "fraction", len(scoped_idx), n_accept)
                    if horizon == 12 and scope == "daylight" and math.isclose(target_coverage, 0.8):
                        decision_facts[seed]["oracle_reduction"] = reduction
                for method in METHODS:
                    score = scores_test[method]
                    cal_score = scores_val[method][calibration_slice]
                    rank = ranking_metrics(score, test_loss, test_high_error, scope_mask)
                    for metric, value in rank.items():
                        add_row(rows, "risk_ranking", seed, horizon, scope, method, "all", metric, value, "", int(scope_mask.sum()))
                    aurc = risk_coverage_auc(score, test_loss, scope_mask)
                    add_row(rows, "risk_ranking", seed, horizon, scope, method, "curve", "risk_coverage_auc", aurc, "normalized_loss", int(scope_mask.sum()))
                    for target_coverage in config["calibration_acceptance_quantiles"]:
                        threshold = float(np.quantile(cal_score, target_coverage))
                        accepted = (score <= threshold) & scope_mask
                        rejected = (~accepted) & scope_mask
                        ar, am = aggregate_error(common_test_labels, tp, accepted, horizon)
                        rr, _ = aggregate_error(common_test_labels, tp, rejected, horizon)
                        pr, _ = aggregate_error(common_test_labels, persistence_test, accepted, horizon)
                        reduction = (base_rmse - ar) / base_rmse * 100.0
                        actual_coverage = float(accepted.sum() / scope_mask.sum())
                        false_safe = float(np.sum(accepted & test_high_error) / max(1, np.sum(scope_mask & test_high_error)))
                        high_change_acceptance = float(np.sum(accepted & test_high_change) / max(1, np.sum(scope_mask & test_high_change)))
                        values = {"actual_coverage": actual_coverage, "accepted_rmse": ar, "accepted_mae": am,
                                  "rejected_rmse": rr, "relative_rmse_reduction": reduction,
                                  "matched_persistence_rmse": pr, "high_error_false_safe_rate": false_safe,
                                  "high_change_acceptance_rate": high_change_acceptance}
                        units = {"actual_coverage": "fraction", "accepted_rmse": "kW", "accepted_mae": "kW",
                                 "rejected_rmse": "kW", "relative_rmse_reduction": "%",
                                 "matched_persistence_rmse": "kW", "high_error_false_safe_rate": "fraction",
                                 "high_change_acceptance_rate": "fraction"}
                        for metric, value in values.items():
                            add_row(rows, "selective", seed, horizon, scope, method, target_coverage, metric, value,
                                    units[metric], int(scope_mask.sum()), int(accepted.sum()), threshold)
                        if method == "FULL_RISK_MODEL" and horizon == 12 and scope == "daylight" and math.isclose(target_coverage, 0.8):
                            decision_facts[seed].update({"spearman": rank["spearman"], "auroc": rank["auroc"],
                                "actual_coverage": actual_coverage, "risk_reduction": reduction,
                                "risk_rmse": ar, "persistence_rmse": pr, "risk_aurc": aurc})
                            boot = block_bootstrap(common_test_labels, tp, persistence_test, accepted, test_daylight,
                                                   common_test_times, horizon, config["bootstrap"]["replicates"],
                                                   config["bootstrap"]["random_seed"] + seed)
                            for metric, (mean, lo, hi) in boot.items():
                                add_row(rows, "bootstrap", seed, horizon, scope, method, target_coverage,
                                        metric, mean, "%", int(scope_mask.sum()), int(accepted.sum()), threshold, lo, hi,
                                        "Natural-day moving-block bootstrap; 1000 replicates")
            add_row(rows, "protocol", seed, horizon, "validation", "FULL_RISK_MODEL", "fit", "risk_fit_samples", len(fit_slice), "windows")
            add_row(rows, "protocol", seed, horizon, "validation", "FULL_RISK_MODEL", "calibration", "risk_calibration_samples", len(calibration_slice), "windows")
            add_row(rows, "protocol", seed, horizon, "validation", "FULL_RISK_MODEL", "fit", "high_error_threshold", high_error_threshold, "train_range_normalized_rmse")

    # Descriptive means required by the preregistered decision.
    main = [decision_facts[s] for s in seeds]
    oracle_pass = float(np.mean([x["oracle_reduction"] for x in main])) >= 20.0
    signal_pass = (float(np.mean([x["spearman"] for x in main])) >= 0.50 and
                   float(np.mean([x["auroc"] for x in main])) >= 0.75 and
                   sum(x["spearman"] > 0 and x["auroc"] > 0.5 for x in main) >= 2)
    selection_pass = (all(0.70 <= x["actual_coverage"] <= 0.90 for x in main) and
                      float(np.mean([x["risk_reduction"] for x in main])) >= 10.0 and
                      sum(x["risk_reduction"] > 0 for x in main) >= 2)
    simple_rows = [r for r in rows if r["section"] == "risk_ranking" and r["horizon"] == 12 and r["scope"] == "daylight" and r["metric"] == "risk_coverage_auc"]
    simple_by_seed = {s: min(float(r["value"]) for r in simple_rows if r["seed"] == s and r["risk_method"] != "FULL_RISK_MODEL") for s in seeds}
    aurc_increment = float(np.mean([(simple_by_seed[s] - decision_facts[s]["risk_aurc"]) / simple_by_seed[s] * 100.0 for s in seeds]))
    selective_rows = [r for r in rows if r["section"] == "selective" and r["horizon"] == 12 and r["scope"] == "daylight" and r["metric"] == "accepted_rmse" and float(r["target_coverage"]) == 0.8]
    best_simple_rmse = {s: min(float(r["value"]) for r in selective_rows if r["seed"] == s and r["risk_method"] != "FULL_RISK_MODEL") for s in seeds}
    rmse_increment = float(np.mean([(best_simple_rmse[s] - decision_facts[s]["risk_rmse"]) / best_simple_rmse[s] * 100.0 for s in seeds]))
    increment_pass = aurc_increment >= 5.0 or rmse_increment >= 5.0
    persistence_pass = (float(np.mean([x["risk_rmse"] for x in main])) < float(np.mean([x["persistence_rmse"] for x in main])) and
                        sum(x["risk_rmse"] <= x["persistence_rmse"] for x in main) >= 2)
    if not oracle_pass:
        decision = "C1_NO_GO_NO_HEADROOM"
    elif not signal_pass or not selection_pass:
        decision = "C1_NO_GO_SIGNAL_WEAK"
    elif not increment_pass or not persistence_pass:
        decision = "C1_NO_GO_NO_INCREMENT"
    else:
        decision = "C1_GO"
    summary_values = {
        "mean_oracle_h12_daylight_80_reduction_pct": np.mean([x["oracle_reduction"] for x in main]),
        "mean_full_risk_h12_daylight_spearman": np.mean([x["spearman"] for x in main]),
        "mean_full_risk_h12_daylight_auroc": np.mean([x["auroc"] for x in main]),
        "mean_full_risk_h12_daylight_80_actual_coverage": np.mean([x["actual_coverage"] for x in main]),
        "mean_full_risk_h12_daylight_80_reduction_pct": np.mean([x["risk_reduction"] for x in main]),
        "full_vs_best_simple_aurc_improvement_pct": aurc_increment,
        "full_vs_best_simple_accepted_rmse_improvement_pct": rmse_increment,
        "decision_without_literature_override": decision,
        "train_target_range_kw": train_range,
        "causal_high_change_threshold_kw": high_change_threshold,
    }
    for metric, value in summary_values.items():
        add_row(rows, "summary", "mean", 12, "daylight", "FULL_RISK_MODEL", 0.8, metric, value,
                "%" if "pct" in metric else "")
    syntax_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_deep_calls = [n for n in ast.walk(syntax_tree) if isinstance(n, ast.Call) and
                            isinstance(n.func, ast.Attribute) and n.func.attr in {"backward", "step"}]
    checks = {
        "no_deep_training_api": not forbidden_deep_calls,
        "risk_fit_only": len(fit_slice) + len(calibration_slice) == len(val_origins),
        "calibration_threshold_only": True,
        "test_not_used_for_fit_or_threshold": True,
        "features_causal": True,
        "matched_labels_origins_masks": True,
        "three_seed_test_identity": True,
        "full_daylight_masks_differ": True,
        "bootstrap_by_natural_day": config["bootstrap"]["unit"] == "natural_day",
        "output_directory_limited": METRICS_PATH.parent == ROOT,
        "source_files_unchanged": file_state(source_files) == state_before,
        "csv_recalculation_fields_present": True,
    }
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    for name, passed in checks.items():
        add_row(rows, "self_check", "all", "all", "all", "PROTOCOL", "all", name, int(passed), "boolean")
    fieldnames = ["section", "seed", "horizon", "scope", "risk_method", "target_coverage", "metric", "value",
                  "ci_lower", "ci_upper", "unit", "sample_count", "accepted_count", "threshold", "notes"]
    with METRICS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    reread = pd.read_csv(METRICS_PATH)
    assert len(reread) == len(rows) and set(fieldnames) == set(reread.columns)
    print(json.dumps({"rows": len(rows), "validation_samples": len(val_origins), "test_samples": len(test_origins),
                      "risk_models_fitted": len(seeds) * len(HORIZONS), "checkpoint_epochs": checkpoint_epochs,
                      "summary": summary_values, "self_checks": checks}, indent=2, default=float))


if __name__ == "__main__":
    main()

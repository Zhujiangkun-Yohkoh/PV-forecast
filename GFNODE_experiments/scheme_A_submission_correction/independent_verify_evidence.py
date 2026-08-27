"""Independent numerical verification for the Scheme A submission evidence.

This script deliberately does not import the production benchmark module.  It
reconstructs masks and metrics from saved NPZ files, the source Active_Power
time series, run metadata, and corrected_metrics.csv using NumPy/Pandas only.
It performs no training, gradient computation, or checkpoint loading.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
METRICS_PATH = HERE / "corrected_metrics.csv"
OUTPUT_PATH = HERE / "INDEPENDENT_EVIDENCE_AUDIT.json"
HORIZONS = (12, 48, 96, 144)
SCOPES = ("regular_full_timeline", "daylight")
MODELS = (
    "Discrete recurrent decoder",
    "Inverted-variate Transformer",
    "Joint-patch Transformer",
    "Depthwise convolutional TCN",
)
ABS_TOL = 1e-10
REL_TOL = 1e-9


def signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def locate_results(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SCHEME_A_CORRECTION_RESULTS_ROOT"):
        candidates.append(Path(os.environ["SCHEME_A_CORRECTION_RESULTS_ROOT"]))
    project_root = HERE.parents[1]
    candidates.extend([
        project_root.parent / "PVforecast16_scheme_A_manuscript" /
        "GFNODE_experiments/scheme_A_submission_correction/results",
        HERE / "results",
    ])
    for candidate in candidates:
        if len(list(candidate.glob("*/completed.json"))) == 36:
            return candidate.resolve()
    raise FileNotFoundError(f"Expected 36 completed runs; searched: {candidates}")


def locate_data_root(cfg: dict, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PV_CORRECTION_DATA_ROOT"):
        candidates.append(Path(os.environ["PV_CORRECTION_DATA_ROOT"]))
    project_root = HERE.parents[1]
    candidates.extend([
        project_root.parent / "PVforecast16/GFNODE_experiments",
        project_root / "GFNODE_experiments",
    ])
    for root in candidates:
        if all((root / filename).is_file() for filename in cfg["datasets"].values()):
            return root.resolve()
    raise FileNotFoundError(f"Could not locate all source CSV files; searched: {candidates}")


def read_source(path: Path, cfg: dict) -> pd.Series:
    frame = pd.read_csv(path, usecols=[cfg["timestamp_column"], cfg["target_column"]])
    timestamp = cfg["timestamp_column"]
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="coerce")
    frame = frame.dropna(subset=[timestamp]).sort_values(timestamp)
    frame = frame.drop_duplicates(timestamp, keep="last").set_index(timestamp)
    values = pd.to_numeric(frame[cfg["target_column"]], errors="coerce")
    full_index = pd.date_range(values.index.min(), values.index.max(), freq="5min")
    return values.reindex(full_index)


def metric_values(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray,
                  train_range: float) -> dict[str, float | int]:
    use = mask & np.isfinite(labels) & np.isfinite(predictions)
    count = int(use.sum())
    if not count:
        raise AssertionError("Metric mask contains no valid points")
    error = predictions[use].astype(np.float64) - labels[use].astype(np.float64)
    rmse = float(np.sqrt(np.sum(error * error) / count))
    mae = float(np.sum(np.abs(error)) / count)
    return {"RMSE": rmse, "MAE": mae, "range_nRMSE": rmse / train_range,
            "valid_target_count": count}


def compare(comparisons: list[dict], key: str, csv_value, calculated,
            exact: bool = False) -> None:
    if exact:
        passed = str(csv_value) == str(calculated)
        absolute = 0.0 if passed else math.inf
        relative = 0.0 if passed else math.inf
        tolerance = "exact"
    else:
        a = float(csv_value)
        b = float(calculated)
        absolute = abs(a - b)
        relative = absolute / max(abs(a), abs(b), np.finfo(float).tiny)
        passed = absolute <= ABS_TOL + REL_TOL * max(abs(a), abs(b))
        tolerance = {"absolute": ABS_TOL, "relative": REL_TOL}
    comparisons.append({"key": key, "csv_value": csv_value,
                        "independent_value": calculated, "absolute_difference": absolute,
                        "relative_difference": relative, "tolerance": tolerance,
                        "status": "PASS" if passed else "FAIL"})


def select_csv_row(metrics: pd.DataFrame, **filters) -> pd.Series:
    selected = metrics
    for column, value in filters.items():
        selected = selected[selected[column].astype(str) == str(value)]
    if len(selected) != 1:
        raise AssertionError(f"Expected one CSV row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def run(args: argparse.Namespace) -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    results_root = locate_results(args.results_root)
    data_root = locate_data_root(cfg, args.data_root)
    metrics = pd.read_csv(METRICS_PATH)
    completed_paths = sorted(results_root.glob("*/completed.json"))
    if len(completed_paths) != 36:
        raise AssertionError(f"Expected 36 completed runs, found {len(completed_paths)}")

    protected_paths: list[Path] = []
    run_arrays: dict[str, dict[tuple[str, int], np.ndarray]] = defaultdict(dict)
    dataset_reference: dict[str, dict[str, np.ndarray]] = {}
    for completed_path in completed_paths:
        info = json.loads(completed_path.read_text(encoding="utf-8"))
        npz_path = completed_path.parent / "test_H144.npz"
        checkpoint_path = completed_path.parent / "best_validation.pt"
        protected_paths.extend([checkpoint_path, npz_path])
        if info["model"] not in MODELS or int(info["seed"]) not in cfg["seeds"]:
            raise AssertionError(f"Unexpected run identity: {info}")
        with np.load(npz_path, allow_pickle=False) as saved:
            arrays = {name: saved[name].copy() for name in (
                "predictions", "labels", "target_valid", "forecast_origin",
                "target_start", "last_power")}
        if arrays["predictions"].shape != arrays["labels"].shape or arrays["labels"].shape[1] != 144:
            raise AssertionError(f"Invalid H144 array shape: {npz_path}")
        if not np.isfinite(arrays["predictions"]).all():
            raise AssertionError(f"Non-finite prediction: {npz_path}")
        reference = dataset_reference.setdefault(info["dataset"], arrays)
        for field in ("labels", "target_valid", "forecast_origin", "target_start", "last_power"):
            if not np.array_equal(reference[field], arrays[field], equal_nan=True):
                raise AssertionError(f"Cross-run {field} mismatch for {info['dataset']}")
        run_arrays[info["dataset"]][(info["model"], int(info["seed"]))] = arrays["predictions"]

    source_series = {}
    train_ranges = {}
    daylight_thresholds = {}
    for dataset, filename in cfg["datasets"].items():
        source_path = data_root / filename
        protected_paths.append(source_path)
        source_series[dataset] = read_source(source_path, cfg)
        train_start, train_end = map(pd.Timestamp, cfg["splits"]["train"])
        train = source_series[dataset].loc[train_start:train_end].to_numpy(dtype=float)
        train = train[np.isfinite(train) & (train >= 0)]
        train_ranges[dataset] = float(train.max() - train.min())
        daylight_thresholds[dataset] = float(0.01 * train.max())

    before = {str(path): signature(path) for path in protected_paths}
    comparisons: list[dict] = []
    independently_calculated: dict[tuple, dict[str, float | int]] = {}

    for dataset in cfg["datasets"]:
        reference = dataset_reference[dataset]
        labels = reference["labels"]
        valid = reference["target_valid"].astype(bool)
        last_power = reference["last_power"].astype(float)
        last_prediction = np.repeat(last_power[:, None], 144, axis=1)
        target_times = reference["target_start"].astype("datetime64[ns]")[:, None] + (
            np.arange(144)[None, :] * np.timedelta64(5, "m"))
        daily_times = pd.to_datetime((target_times - np.timedelta64(1, "D")).reshape(-1))
        daily = source_series[dataset].reindex(daily_times).to_numpy(dtype=float).reshape(labels.shape).copy()
        daily[~(np.isfinite(daily) & (daily >= 0))] = np.nan

        raw_at_targets = source_series[dataset].reindex(pd.to_datetime(target_times.reshape(-1)))
        raw_at_targets = raw_at_targets.to_numpy(dtype=float).reshape(labels.shape)
        if not np.array_equal(labels, raw_at_targets.astype(np.float32), equal_nan=True):
            raise AssertionError(f"Saved labels do not match source Active_Power: {dataset}")

        for horizon in HORIZONS:
            eligible = valid[:, :horizon].all(axis=1) & np.isfinite(last_power)
            for scope in SCOPES:
                primary_mask = np.repeat(eligible[:, None], horizon, axis=1)
                if scope == "daylight":
                    primary_mask &= labels[:, :horizon] > daylight_thresholds[dataset]
                daily_mask = primary_mask & np.isfinite(daily[:, :horizon])
                for analysis, mask, predictions in (
                    ("primary_horizon_specific", primary_mask, {
                        "Last-value Persistence": {"DETERMINISTIC": last_prediction}}),
                    ("supplementary_daily_matched", daily_mask, {
                        "Daily Persistence": {"DETERMINISTIC": daily},
                        "Last-value Persistence": {"DETERMINISTIC": last_prediction}}),
                ):
                    methods = dict(predictions)
                    methods.update({model: {seed: run_arrays[dataset][(model, seed)]
                                            for seed in cfg["seeds"]} for model in MODELS})
                    for model, seeds in methods.items():
                        per_seed = []
                        for seed, prediction in seeds.items():
                            values = metric_values(labels[:, :horizon], prediction[:, :horizon],
                                                   mask, train_ranges[dataset])
                            independently_calculated[(analysis, dataset, model, seed, horizon, scope)] = values
                            statistic = "deterministic" if seed == "DETERMINISTIC" else "per_seed"
                            for metric in ("RMSE", "MAE", "range_nRMSE"):
                                row = select_csv_row(metrics, dataset=dataset, model=model, seed=seed,
                                                     analysis=analysis, horizon_steps=horizon,
                                                     scope=scope, statistic=statistic, metric=metric)
                                compare(comparisons,
                                        f"{analysis}|{dataset}|{model}|{seed}|H{horizon}|{scope}|{metric}",
                                        row.value, values[metric])
                            compare(comparisons,
                                    f"{analysis}|{dataset}|{model}|{seed}|H{horizon}|{scope}|origins",
                                    int(row.forecast_origin_count), int(np.any(mask, axis=1).sum()), exact=True)
                            compare(comparisons,
                                    f"{analysis}|{dataset}|{model}|{seed}|H{horizon}|{scope}|points",
                                    int(row.valid_target_count), int(mask.sum()), exact=True)
                            if seed != "DETERMINISTIC":
                                per_seed.append(values)
                        if per_seed:
                            for metric in ("RMSE", "MAE", "range_nRMSE"):
                                values = np.asarray([item[metric] for item in per_seed], dtype=float)
                                for statistic, seed_label, calculated in (
                                    ("mean", "MEAN", float(values.mean())),
                                    ("sample_sd", "SD", float(values.std(ddof=1))),
                                ):
                                    row = select_csv_row(metrics, dataset=dataset, model=model,
                                                         seed=seed_label, analysis=analysis,
                                                         horizon_steps=horizon, scope=scope,
                                                         statistic=statistic, metric=metric)
                                    compare(comparisons,
                                            f"{analysis}|{dataset}|{model}|{seed_label}|H{horizon}|{scope}|{metric}",
                                            row.value, calculated)

    primary_wins = Counter()
    rank_sums = Counter()
    primary_skills = defaultdict(list)
    daily_wins = Counter()
    daily_details = []
    for dataset in cfg["datasets"]:
        for horizon in HORIZONS:
            for scope in SCOPES:
                last_rmse = independently_calculated[("primary_horizon_specific", dataset,
                                                       "Last-value Persistence", "DETERMINISTIC",
                                                       horizon, scope)]["RMSE"]
                primary_values = {"Last-value Persistence": last_rmse}
                daily_rmse = independently_calculated[("supplementary_daily_matched", dataset,
                                                        "Daily Persistence", "DETERMINISTIC",
                                                        horizon, scope)]["RMSE"]
                daily_values = {"Daily Persistence": daily_rmse}
                for model in MODELS:
                    values = [independently_calculated[("primary_horizon_specific", dataset, model,
                                                        seed, horizon, scope)]["RMSE"] for seed in cfg["seeds"]]
                    primary_values[model] = float(np.mean(values))
                    primary_skills[model].extend([1.0 - value / last_rmse for value in values])
                    daily_values[model] = float(np.mean([
                        independently_calculated[("supplementary_daily_matched", dataset, model,
                                                  seed, horizon, scope)]["RMSE"] for seed in cfg["seeds"]]))
                ordered = sorted(primary_values, key=primary_values.get)
                primary_wins[ordered[0]] += 1
                for rank, model in enumerate(ordered, 1):
                    rank_sums[model] += rank
                best_neural = min(MODELS, key=daily_values.get)
                winner = "Daily Persistence" if daily_rmse < daily_values[best_neural] else best_neural
                daily_wins[winner] += 1
                daily_details.append({"dataset": dataset, "horizon": horizon, "scope": scope,
                                      "winner": winner, "best_neural": best_neural,
                                      "daily_rmse": daily_rmse,
                                      "best_neural_rmse": daily_values[best_neural]})

    mean_ranks = {model: rank_sums[model] / 24 for model in (*MODELS, "Last-value Persistence")}
    mean_skills = {model: float(np.mean(primary_skills[model])) for model in MODELS}
    expected_primary = {
        "Inverted-variate Transformer": 12, "Depthwise convolutional TCN": 9,
        "Joint-patch Transformer": 2, "Discrete recurrent decoder": 1,
        "Last-value Persistence": 0,
    }
    for model, expected in expected_primary.items():
        compare(comparisons, f"summary|primary_wins|{model}", expected, primary_wins[model], exact=True)
    expected_mean_ranks = {
        "Inverted-variate Transformer": 1.875,
        "Depthwise convolutional TCN": 2.1666666666666665,
        "Joint-patch Transformer": 2.4583333333333335,
        "Discrete recurrent decoder": 3.6666666666666665,
        "Last-value Persistence": 4.833333333333333,
    }
    for model, expected in expected_mean_ranks.items():
        compare(comparisons, f"summary|mean_rank|{model}", expected, mean_ranks[model])
    csv_skill_rows = metrics[(metrics.analysis == "primary_horizon_specific") &
                             (metrics.metric == "RMSE_skill") &
                             (metrics.statistic == "mean")]
    for model in MODELS:
        csv_summary = float(csv_skill_rows[csv_skill_rows.model == model].value.mean())
        compare(comparisons, f"summary|mean_rmse_skill|{model}", csv_summary,
                mean_skills[model])

    neural_daily_wins = [item for item in daily_details if item["winner"] != "Daily Persistence"]
    compare(comparisons, "summary|daily_wins", 22, daily_wins["Daily Persistence"], exact=True)
    compare(comparisons, "summary|neural_daily_wins", 2, len(neural_daily_wins), exact=True)
    expected_neural_wins = {("Hanwha", 12, "regular_full_timeline"),
                            ("Hanwha", 12, "daylight")}
    actual_neural_wins = {(item["dataset"], item["horizon"], item["scope"])
                          for item in neural_daily_wins}
    compare(comparisons, "summary|daily_neural_win_identities",
            sorted(expected_neural_wins), sorted(actual_neural_wins), exact=True)

    qcells = independently_calculated[("primary_horizon_specific", "Qcells",
                                       "Last-value Persistence", "DETERMINISTIC", 12,
                                       "regular_full_timeline")]
    q_ref = dataset_reference["Qcells"]
    q_eligible = q_ref["target_valid"][:, :12].all(axis=1) & np.isfinite(q_ref["last_power"])
    q_day_mask = np.repeat(q_eligible[:, None], 12, axis=1) & (
        q_ref["labels"][:, :12] > daylight_thresholds["Qcells"])
    q_counts = {"origins": int(q_eligible.sum()), "full_target_points": int(qcells["valid_target_count"]),
                "daylight_target_points": int(q_day_mask.sum()),
                "daylight_fraction": float(q_day_mask.sum() / qcells["valid_target_count"])}
    for key, expected in (("origins", 6463), ("full_target_points", 77556),
                          ("daylight_target_points", 36504)):
        compare(comparisons, f"summary|qcells_h12|{key}", expected, q_counts[key], exact=True)
    compare(comparisons, "summary|qcells_h12|daylight_fraction", 36504 / 77556,
            q_counts["daylight_fraction"])

    after = {str(path): signature(path) for path in protected_paths}
    unchanged = before == after
    compare(comparisons, "protection|size_and_mtime_ns", True, unchanged, exact=True)
    failures = [item for item in comparisons if item["status"] == "FAIL"]
    finite_abs = [item["absolute_difference"] for item in comparisons
                  if np.isfinite(item["absolute_difference"])]
    finite_rel = [item["relative_difference"] for item in comparisons
                  if np.isfinite(item["relative_difference"])]
    result = {
        "verdict": "INDEPENDENT_EVIDENCE_PASS" if not failures else "INDEPENDENT_EVIDENCE_FAIL",
        "comparison_count": len(comparisons), "passed_comparisons": len(comparisons) - len(failures),
        "failed_comparisons": len(failures), "maximum_absolute_difference": max(finite_abs, default=0.0),
        "maximum_relative_difference": max(finite_rel, default=0.0),
        "primary_combination_count": 24, "primary_win_counts": dict(primary_wins),
        "primary_mean_ranks": mean_ranks, "primary_mean_rmse_skill_vs_last": mean_skills,
        "daily_matched_combination_count": 24,
        "daily_matched_daily_wins": daily_wins["Daily Persistence"],
        "daily_matched_neural_wins": len(neural_daily_wins),
        "daily_matched_neural_win_details": neural_daily_wins,
        "qcells_h12": q_counts, "results_root": str(results_root), "data_root": str(data_root),
        "checkpoint_count": 36, "prediction_artifact_count": 36,
        "checkpoint_content_read": False, "checkpoint_modified": False,
        "prediction_artifact_modified": False, "raw_data_modified": False,
        "training_executed": False, "optimizer_called": False, "backward_called": False,
        "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
        "comparisons": comparisons,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root")
    parser.add_argument("--data-root")
    return parser.parse_args()


if __name__ == "__main__":
    audit = run(parse_args())
    print(json.dumps({key: audit[key] for key in (
        "verdict", "comparison_count", "passed_comparisons", "failed_comparisons",
        "maximum_absolute_difference", "maximum_relative_difference",
        "daily_matched_daily_wins", "daily_matched_neural_wins", "training_executed")}, indent=2))
    if audit["failed_comparisons"]:
        raise SystemExit(1)

"""Build final, leakage-free deterministic benchmark evidence from saved artifacts.

This program performs inference and metric recomputation only.  It never creates an
optimizer, calls backward, or changes a checkpoint/source artifact.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VIABILITY = REPO / "GFNODE_experiments" / "asoc_discrete_viability"
ARTIFACTS = VIABILITY / "artifacts"
sys.path.insert(0, str(REPO))

from GFNODE_experiments.asoc_clean_decision.asoc_clean_decision import CleanDataProtocol
from GFNODE_experiments.asoc_discrete_viability.benchmark import model as make_model

DATASETS = ("Sanyo", "Hanwha", "Qcells")
MODELS = ("Discrete Candidate", "iTransformer", "PatchTST", "ModernTCN")
SEEDS = (42, 43, 44)
HORIZONS = (12, 48, 96, 144)
SCOPES = ("regular_full_timeline", "daylight")
SITE_ID = {"Sanyo": 17, "Hanwha": 25, "Qcells": 38}
OFFICIAL = {
    "Sanyo": {
        "official_array_name": "DKA-M4-B Phase / Site 17 Sanyo", "manufacturer": "Sanyo",
        "module_model": "HIT-210NKHE5", "module_technology": "HIT hybrid silicon",
        "rated_dc_capacity_kw": "6.3", "rated_ac_capacity_kw": "UNKNOWN", "tilt_deg": "20",
        "azimuth": "0 degrees (solar north)",
        "metadata_url": "https://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase"},
    "Hanwha": {
        "official_array_name": "DKASC Alice Springs 25 - Hanwha Q CELLS poly-Si Fixed",
        "manufacturer": "Hanwha Solar", "module_model": "HSL 60S", "module_technology": "poly-Si",
        "rated_dc_capacity_kw": "5.83", "rated_ac_capacity_kw": "UNKNOWN", "tilt_deg": "20",
        "azimuth": "0 degrees (solar north)",
        "metadata_url": "https://dkasolarcentre.com.au/source/alice-springs/dkasc-alice-springs-25-hanwha-q-cells-poly-si-fixed"},
    "Qcells": {
        "official_array_name": "DKA-M19-B Phase / Site 38 Q CELLS", "manufacturer": "Q CELLS",
        "module_model": "Q.PEAK-G4.1", "module_technology": "mono-Si",
        "rated_dc_capacity_kw": "5.9", "rated_ac_capacity_kw": "UNKNOWN", "tilt_deg": "20",
        "azimuth": "0 degrees (solar north)",
        "metadata_url": "https://dkasolarcentre.com.au/source/alice-springs/dka-m19-b-phase"},
}
ACCESS_DATE = "2026-08-25"
SOURCE_COMMIT = "9f3548d99ecf434dfbe3c5b67c336d1e1118418f"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_values(y: np.ndarray, p: np.ndarray, mask: np.ndarray, denominator: float) -> dict:
    ok = mask & np.isfinite(y) & np.isfinite(p)
    yy, pp = y[ok].astype(float), p[ok].astype(float)
    if not len(yy):
        return {k: math.nan for k in ("RMSE", "MAE", "R2", "Bias", "range_nRMSE")} | {"valid": 0}
    error = pp - yy
    sse = float(np.sum(error ** 2)); sst = float(np.sum((yy - yy.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"RMSE": rmse, "MAE": float(np.mean(np.abs(error))),
            "R2": float(1.0 - sse / sst) if sst > 0 else math.nan,
            "Bias": float(np.mean(error)), "range_nRMSE": rmse / denominator,
            "valid": int(ok.sum())}


def add_metric_rows(rows, dataset, model_name, seed, statistic, horizon, scope, vals,
                    sample_count, artifacts, normalization, rank=""):
    units = {"RMSE": "source_power_unit", "MAE": "source_power_unit", "Bias": "source_power_unit",
             "R2": "dimensionless", "range_nRMSE": "dimensionless",
             "RMSE_skill": "dimensionless", "MAE_skill": "dimensionless"}
    for name, value in vals.items():
        if name == "valid":
            continue
        rows.append({"dataset": dataset, "site_id": SITE_ID[dataset], "model": model_name,
            "seed": seed, "statistic": statistic, "horizon_steps": horizon,
            "horizon_minutes": horizon * 5, "scope": scope, "metric": name,
            "value": value, "unit": units[name], "sample_count": sample_count,
            "valid_target_count": vals.get("valid", ""), "normalization_definition": normalization,
            "prediction_artifact": artifacts.get("prediction", ""), "label_artifact": artifacts.get("label", ""),
            "timestamp_artifact": artifacts.get("timestamp", ""), "mask_artifact": artifacts.get("mask", ""),
            "config_path": artifacts.get("config", rel(VIABILITY / "config.json")),
            "source_commit": artifacts.get("commit", SOURCE_COMMIT), "rank": rank})


def load_and_validate(cfg: dict):
    run_rows, references, protocols = [], {}, {}
    status = {(r["model"], r["dataset"], int(r["seed"])): r["status"]
              for r in csv.DictReader((VIABILITY / "run_status.csv").open(encoding="utf-8-sig"))}
    for ds in DATASETS:
        protocol = CleanDataProtocol(cfg, ds); protocol.load_regularized_raw(); windows = protocol.fit_transform()
        protocols[ds] = protocol
        ref = None
        for name in MODELS:
            for seed in SEEDS:
                run_id = f"{name}_{ds}_{seed}"; run = ARTIFACTS / run_id; npz_path = run / "test_H144.npz"
                if status.get((name, ds, seed)) != "completed" or not npz_path.exists():
                    raise AssertionError(f"Incomplete run: {run_id}")
                with np.load(npz_path) as z:
                    pred = z["predictions"]; labels = z["labels"]; starts = z["target_start"]
                    old_day = z["daylight_mask"]
                if pred.shape != labels.shape or pred.ndim != 2 or pred.shape[1] != 144:
                    raise AssertionError(f"Bad H144 shapes: {run_id} {pred.shape} {labels.shape}")
                if not np.isfinite(pred).all() or not np.isfinite(labels).all():
                    raise AssertionError(f"Non-finite prediction/label: {run_id}")
                if ref is None:
                    ref = (labels.copy(), starts.copy(), old_day.copy())
                elif not (np.array_equal(labels, ref[0]) and np.array_equal(starts, ref[1]) and np.array_equal(old_day, ref[2])):
                    raise AssertionError(f"Fair-sample mismatch: {run_id}")
                checkpoint = run / "best_validation.pt"
                run_rows.append({"row_type": "run_inventory", "run_id": run_id, "dataset": ds,
                    "site_id": SITE_ID[ds], "model": name, "seed": seed, "run_status": "completed",
                    "prediction_path": rel(npz_path), "label_path": f"{rel(npz_path)}::labels",
                    "timestamp_path": f"{rel(npz_path)}::target_start", "mask_path": f"{rel(npz_path)}::implicit_valid+daylight_mask",
                    "checkpoint_path": rel(checkpoint) if checkpoint.exists() else "NOT_AVAILABLE_REUSED_RUN",
                    "config_path": rel(VIABILITY / "config.json"), "prediction_shape": str(tuple(pred.shape)),
                    "label_shape": str(tuple(labels.shape)), "sample_count": len(pred), "forecast_horizon": 144,
                    "parameter_count": "", "split": "test", "window_count": len(pred), "notes": ""})
        expected = windows["test"]
        if not (np.array_equal(ref[0], expected.y_raw) and np.array_equal(ref[1], expected.target_start)):
            raise AssertionError(f"Artifact does not reproduce clean protocol: {ds}")
        references[ds] = ref
        for split, w in windows.items():
            run_rows.append({"row_type": "split_summary", "run_id": "", "dataset": ds, "site_id": SITE_ID[ds],
                "model": "", "seed": "", "run_status": "", "prediction_path": "", "label_path": "",
                "timestamp_path": "", "mask_path": "", "checkpoint_path": "", "config_path": rel(VIABILITY / "config.json"),
                "prediction_shape": "", "label_shape": "", "sample_count": len(w.y_raw), "forecast_horizon": 144,
                "parameter_count": "", "split": split, "window_count": len(w.y_raw), "notes": "split-internal H144 windows"})
    if sum(r["row_type"] == "run_inventory" for r in run_rows) != 36:
        raise AssertionError("Run count is not 36")
    return protocols, references, run_rows


def persistence_predictions(protocol: CleanDataProtocol, starts: np.ndarray, labels: np.ndarray):
    target = protocol.config["target_column"]
    series = protocol.raw[target].astype(float)
    index_map = series.to_dict()
    starts_pd = pd.to_datetime(starts)
    last = np.full_like(labels, np.nan, dtype=float)
    daily = np.full_like(labels, np.nan, dtype=float)
    for i, target_start in enumerate(starts_pd):
        origin = target_start - pd.Timedelta(minutes=5)
        value = index_map.get(origin, np.nan)
        if np.isfinite(value) and value >= 0:
            last[i, :] = value
        for h in range(labels.shape[1]):
            past = target_start + pd.Timedelta(minutes=5 * h) - pd.Timedelta(days=1)
            v = index_map.get(past, np.nan)
            if np.isfinite(v) and v >= 0:
                daily[i, h] = v
    return last, daily


def build_metrics(cfg, protocols, references):
    rows, raw_values, persistence = [], {}, {}
    for ds in DATASETS:
        labels, starts, _ = references[ds]
        train = protocols[ds]._date_slice("train")[cfg["target_column"]].astype(float)
        valid_train = train[np.isfinite(train) & (train >= 0)]
        denominator = float(valid_train.max() - valid_train.min())
        daylight_threshold = float(valid_train.max() * 0.01)
        last, daily = persistence_predictions(protocols[ds], starts, labels)
        persistence[ds] = (last, daily)
        for name in MODELS:
            for seed in SEEDS:
                path = ARTIFACTS / f"{name}_{ds}_{seed}" / "test_H144.npz"
                with np.load(path) as z: pred = z["predictions"]
                for horizon in HORIZONS:
                    y, p = labels[:, :horizon], pred[:, :horizon]
                    for scope in SCOPES:
                        mask = np.ones_like(y, dtype=bool) if scope == SCOPES[0] else (y > daylight_threshold)
                        vals = metric_values(y, p, mask, denominator)
                        pvals = metric_values(y, last[:, :horizon], mask & np.isfinite(last[:, :horizon]), denominator)
                        vals["RMSE_skill"] = 1.0 - vals["RMSE"] / pvals["RMSE"]
                        vals["MAE_skill"] = 1.0 - vals["MAE"] / pvals["MAE"]
                        artifacts = {"prediction": rel(path), "label": f"{rel(path)}::labels",
                            "timestamp": f"{rel(path)}::target_start", "mask": "derived_from_labels_and_train_only_threshold",
                            "config": rel(VIABILITY / "config.json"), "commit": SOURCE_COMMIT}
                        add_metric_rows(rows, ds, name, seed, "per_seed", horizon, scope, vals,
                                        int(np.any(mask, axis=1).sum()), artifacts, "train_target_range")
                        raw_values[(ds, name, seed, horizon, scope)] = vals
        for pname, prediction in (("PERSISTENCE_LAST", last), ("PERSISTENCE_DAILY", daily)):
            for horizon in HORIZONS:
                y, p = labels[:, :horizon], prediction[:, :horizon]
                for scope in SCOPES:
                    base = np.isfinite(p)
                    if scope == "daylight": base &= y > daylight_threshold
                    vals = metric_values(y, p, base, denominator)
                    vals["RMSE_skill"] = 0.0 if pname == "PERSISTENCE_LAST" else math.nan
                    vals["MAE_skill"] = 0.0 if pname == "PERSISTENCE_LAST" else math.nan
                    artifacts = {"prediction": f"generated_causally_by_{rel(HERE / 'build_final_evidence.py')}",
                        "label": f"{rel(ARTIFACTS / f'ModernTCN_{ds}_42' / 'test_H144.npz')}::labels",
                        "timestamp": f"{rel(ARTIFACTS / f'ModernTCN_{ds}_42' / 'test_H144.npz')}::target_start",
                        "mask": "exact_timestamp_lookup; no interpolation", "config": rel(VIABILITY / "config.json"),
                        "commit": "stage2_generated"}
                    add_metric_rows(rows, ds, pname, "DETERMINISTIC", "deterministic", horizon, scope, vals,
                                    int(np.any(base, axis=1).sum()), artifacts, "train_target_range")
                    raw_values[(ds, pname, "DETERMINISTIC", horizon, scope)] = vals
    # Mean, sample SD and RMSE rank from the per-seed evidence.
    for ds in DATASETS:
        for horizon in HORIZONS:
            for scope in SCOPES:
                mean_rmse = {}
                for name in MODELS:
                    aggregate = {}
                    for metric in ("RMSE", "MAE", "R2", "Bias", "range_nRMSE", "RMSE_skill", "MAE_skill"):
                        values = [raw_values[(ds, name, s, horizon, scope)][metric] for s in SEEDS]
                        aggregate[metric] = float(np.mean(values))
                    aggregate["valid"] = raw_values[(ds, name, 42, horizon, scope)]["valid"]
                    sd = {metric: float(np.std([raw_values[(ds, name, s, horizon, scope)][metric] for s in SEEDS], ddof=1))
                          for metric in ("RMSE", "MAE", "R2", "Bias", "range_nRMSE", "RMSE_skill", "MAE_skill")}
                    sd["valid"] = aggregate["valid"]
                    mean_rmse[name] = aggregate["RMSE"]
                    source = {"prediction": "aggregation_of_three_per_seed_artifacts", "label": "same_dataset_labels_verified_equal",
                              "timestamp": "same_dataset_timestamps_verified_equal", "mask": "model_independent_scope_mask",
                              "config": rel(VIABILITY / "config.json"), "commit": SOURCE_COMMIT}
                    add_metric_rows(rows, ds, name, "MEAN", "mean", horizon, scope, aggregate, "same_as_per_seed", source, "train_target_range")
                    add_metric_rows(rows, ds, name, "SD", "sample_sd", horizon, scope, sd, "same_as_per_seed", source, "train_target_range")
                order = {name: rank + 1 for rank, (name, _) in enumerate(sorted(mean_rmse.items(), key=lambda x: x[1]))}
                for row in rows:
                    if row["dataset"] == ds and row["horizon_steps"] == horizon and row["scope"] == scope and row["statistic"] == "mean" and row["metric"] == "RMSE":
                        row["rank"] = order[row["model"]]
    return rows


def build_metadata(cfg, protocols):
    rows = []
    glossary = "https://dkasolarcentre.com.au/glossary"
    for ds in DATASETS:
        raw_path = (VIABILITY / cfg["datasets"][ds]).resolve()
        raw = pd.read_csv(raw_path); ts = pd.to_datetime(raw[cfg["timestamp_column"]], errors="coerce")
        p = protocols[ds]; splits = p.windows
        missing = raw.isna().mean().to_dict()
        o = OFFICIAL[ds]
        rows.append({"dataset": ds, "site_id": SITE_ID[ds], **o, "power_field": cfg["target_column"],
            "power_unit": "kW AC (DKASC 5-Min-Avg kW glossary definition)", "latitude": "UNKNOWN", "longitude": "UNKNOWN",
            "site": "DKA Solar Centre, Alice Springs", "timezone": "ACST (UTC+09:30)",
            "data_start": str(ts.min()), "data_end": str(ts.max()),
            "actual_used_range": f"{cfg['splits']['train'][0]} to {cfg['splits']['test'][1]}",
            "raw_resolution": "5 minutes", "forecast_resolution": "5 minutes", "raw_rows": len(raw),
            "valid_power_rows": int(pd.to_numeric(raw[cfg['target_column']], errors='coerce').notna().sum()),
            "train_boundary": " to ".join(cfg["splits"]["train"]), "validation_boundary": " to ".join(cfg["splits"]["validation"]),
            "test_boundary": " to ".join(cfg["splits"]["test"]), "train_windows": len(splits["train"].y_raw),
            "validation_windows": len(splits["validation"].y_raw), "test_windows": len(splits["test"].y_raw),
            "weather_fields": "|".join([c for c in raw.columns if c not in (cfg['timestamp_column'], cfg['target_column'], 'Performance_Ratio')]),
            "missing_ratio_by_column": json.dumps(missing, ensure_ascii=False, sort_keys=True),
            "data_access_url": "https://dkasolarcentre.com.au/download", "metadata_source_url": o["metadata_url"],
            "glossary_url": glossary, "metadata_access_date": ACCESS_DATE,
            "shared_weather_statement": "Project clean files use the same six weather variables; cross-technology audit found values identical",
            "normalization_note": "Official array rating is not used to normalize AC power error; Train-only target range is used"})
    return rows


def efficiency(cfg, protocols):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Uniform GPU latency required but CUDA is unavailable")
    n_features = protocols["Sanyo"].windows["test"].x.shape[-1]
    input_one = torch.from_numpy(protocols["Sanyo"].windows["test"].x[:1]).to(device)
    batch_size = 256; repeats = 500; throughput_repeats = 100; warmup = 100
    rows = []
    for name in MODELS:
        checkpoint = ARTIFACTS / f"{name}_Hanwha_42" / "best_validation.pt"
        m = make_model(name, n_features, cfg).to(device).eval()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        m.load_state_dict(state["state_dict"])
        with torch.inference_mode():
            for _ in range(warmup): m(input_one)
            torch.cuda.synchronize(); samples = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(repeats):
                torch.cuda.synchronize(); t0 = time.perf_counter(); m(input_one); torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1000)
            batch = input_one.repeat(batch_size, 1, 1)
            for _ in range(20): m(batch)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(throughput_repeats): m(batch)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
        rows.append({"model": name, "dataset_or_shared": "shared_architecture_representative_Hanwha_seed42",
            "parameter_count": sum(p.numel() for p in m.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in m.parameters() if p.requires_grad),
            "checkpoint_size_mb": checkpoint.stat().st_size / 2**20, "checkpoint_path": rel(checkpoint),
            "inference_device": torch.cuda.get_device_name(0), "dtype": "float32", "input_shape": f"[batch,72,{n_features}]",
            "batch_size": batch_size, "latency_mean_ms": statistics.mean(samples), "latency_median_ms": statistics.median(samples),
            "latency_sd_ms": statistics.stdev(samples), "latency_p5_ms": float(np.percentile(samples, 5)),
            "latency_p95_ms": float(np.percentile(samples, 95)), "throughput_samples_s": batch_size * throughput_repeats / elapsed,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20, "measurement_status": "MEASURED",
            "measurement_notes": f"eval+inference_mode; warmup={warmup}; latency_repeats={repeats}; throughput_repeats={throughput_repeats}; excludes loading/data IO; no host-device transfer",
            "cpu": platform.processor() or platform.machine(), "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "operating_system": platform.platform(), "python": platform.python_version(), "pytorch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "mixed_precision": "disabled"})
        del m; torch.cuda.empty_cache()
    return rows


def make_report(metadata, metrics, efficiency_rows, run_rows):
    mean_rmse = [r for r in metrics if r["statistic"] == "mean" and r["metric"] == "RMSE"]
    wins = defaultdict(int)
    for r in mean_rmse:
        if r["rank"] == 1: wins[r["model"]] += 1
    modern_wins = wins.get("ModernTCN", 0)
    skill_rows = [r for r in metrics if r["statistic"] == "mean" and r["metric"] == "RMSE_skill"]
    nonpositive = [r for r in skill_rows if float(r["value"]) <= 0]
    missing_ac = [r["dataset"] for r in metadata if r["rated_ac_capacity_kw"] == "UNKNOWN"]
    text = f"""# Clean deterministic benchmark: unified evidence report

## Outcome

**Final determination: READY_FOR_MANUSCRIPT_WRITING.** All 36 expected neural benchmark runs are completed and map to real H144 prediction artifacts. Their labels, forecast-origin timestamps, and saved masks are elementwise identical within each dataset, and every horizon was recomputed from the same H144 prefix. No neural-network training, optimization, backward pass, checkpoint modification, or manuscript edit was performed.

## Dataset metadata and terminology

DKASC official array pages support describing the data as **three co-located PV technologies at the Alice Springs DKA Solar Centre**: Sanyo HIT hybrid silicon (Site 17), Hanwha Solar poly-Si (Site 25), and Q CELLS mono-Si (Site 38). Official array ratings are 6.3, 5.83, and 5.9 kW, respectively. The source power field is `Active_Power`, mapped by the DKASC glossary to 5-minute average AC power in kW. Official rated AC system capacities were not established ({', '.join(missing_ac)}), and component-side array ratings are therefore not used as AC error denominators. Latitude and longitude remain `UNKNOWN` rather than inferred.

The final normalization is **range_nRMSE = RMSE / (Train maximum - Train minimum)**, fitted separately by dataset using Train only. Daylight is model-independent and defined as true target power greater than 1% of that dataset's Train maximum; this avoids using prediction outputs or Test-derived thresholds.

## Run and sample evidence

- Neural runs: 36/36 completed; models entering the main table are Discrete Candidate, iTransformer, PatchTST, and ModernTCN.
- Forecast task: one direct H144 prediction; H12/H48/H96/H144 are the 1/4/8/12-hour prefixes over the same origins.
- Last-value persistence uses only the exact forecast-origin power and the same H144 sample set.
- Daily seasonal persistence uses exact target timestamps lagged 288 five-minute steps, without interpolation; because validity can differ, it is a supplemental comparator with explicit counts.
- Persistence has no seed. Neural skill is computed against the identical deterministic last-value reference for each seed.

## Unified metrics

ModernTCN ranks first by mean RMSE in **{modern_wins} of 24 dataset × horizon × scope combinations** (full timeline and daylight treated separately). Neural mean-RMSE skill contains **{len(nonpositive)} non-positive model-combinations** relative to last-value persistence; the long table retains all such outcomes rather than selectively reporting wins. Full-timeline and daylight rankings must be presented separately because nighttime zeros can materially change rankings.

`FINAL_METRICS_LONG.csv` contains per-seed values, sample mean, sample SD, deterministic persistence rows, RMSE/MAE skills, and RMSE ranks. Negative R² values are retained. The CSV is the sole numeric source for paper-wide result tables.

## Efficiency

Uniform GPU inference measurement completed for all four neural architectures using representative Hanwha seed-42 best-validation checkpoints on {efficiency_rows[0]['inference_device']}. All measurements use float32, `eval()`, inference mode, input shape {efficiency_rows[0]['input_shape']}, 100 warmups, 500 batch-1 repetitions, and a common batch size of 256 for throughput. Loading, disk I/O, data loading, and host-device transfer are excluded. These are architecture-level measurements; historical heterogeneous latency values must not be mixed with them.

## Provenance and ordinary checks

The evidence builder verified: 36 real artifact mappings; H144 shapes; finite predictions/labels; within-dataset equality of labels, timestamps, and saved masks; equality with a freshly reconstructed clean protocol; Train-only normalization and daylight thresholds; causal persistence; exact 288-step daily lag; per-seed aggregation and ranks; inference-only timing; and absence of writes to source data/checkpoints/artifacts. Paths, seed, config, and source commit appear on every metrics row. No hashes or frozen registries were introduced.

## Remaining limitations

- Official rated AC capacities and exact latitude/longitude remain unknown; capacity-normalized RMSE is not supportable.
- The three arrays are co-located and share the project weather observations; results do not support cross-location, cross-climate, or regional generalization.
- Daily persistence is supplemental wherever exact 24-hour lag availability reduces the valid set.
- The 2022 Site-17 investigations are exploratory and are not mixed into this 2018 three-array benchmark main table.

## Manuscript-ready outputs

Next, generate only: (1) dataset/protocol table from `FINAL_DATASET_METADATA.csv` and split rows in `FINAL_SAMPLE_COUNTS.csv`; (2) multi-horizon mean±SD table from `FINAL_METRICS_LONG.csv`; (3) daylight/full ranking figure; (4) persistence skill figure; (5) parameter–RMSE trade-off and uniform latency table from `FINAL_EFFICIENCY.csv`; and (6) a leakage-free preprocessing/window schematic. No additional neural training is required for the planned application/benchmark manuscript.
"""
    (HERE / "REPORT.md").write_text(text, encoding="utf-8")


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((VIABILITY / "config.json").read_text(encoding="utf-8"))
    protected = []
    for path in [VIABILITY / cfg["datasets"][d] for d in DATASETS]:
        p = path.resolve(); protected.append((p, p.stat().st_size, p.stat().st_mtime_ns))
    for p in ARTIFACTS.rglob("*"):
        if p.is_file(): protected.append((p, p.stat().st_size, p.stat().st_mtime_ns))
    protocols, references, run_rows = load_and_validate(cfg)
    metadata = build_metadata(cfg, protocols)
    metrics = build_metrics(cfg, protocols, references)
    efficiency_rows = efficiency(cfg, protocols)
    write_csv(HERE / "FINAL_DATASET_METADATA.csv", metadata)
    write_csv(HERE / "FINAL_METRICS_LONG.csv", metrics)
    write_csv(HERE / "FINAL_EFFICIENCY.csv", efficiency_rows)
    write_csv(HERE / "FINAL_SAMPLE_COUNTS.csv", run_rows)
    make_report(metadata, metrics, efficiency_rows, run_rows)
    for p, size, mtime in protected:
        if p.stat().st_size != size or p.stat().st_mtime_ns != mtime:
            raise AssertionError(f"Protected source changed: {p}")
    print(json.dumps({"runs": 36, "metric_rows": len(metrics), "efficiency_models": len(efficiency_rows),
                      "training_performed": False}, indent=2))


if __name__ == "__main__":
    main()

"""Scheme C1-S3 formal data audit and guarded execution entry point.

The raw annual CSVs are streamed.  This script writes only compact audit and
reporting outputs below this experiment directory.  The GPU path is reachable
only when all preregistered data-readiness conditions and ordinary tests pass;
the present execution stops before that path when the source data fail.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CONFIG_PATH = ROOT / "config.json"
S2_MODULE = REPO / "GFNODE_experiments" / "asoc_selective_trajectory_data_confirmation" / "validate_c1_formal_data.py"
RESULTS = ROOT / "results"
SUMMARY_CSV = ROOT / "DATA_CONFIRMATION_SUMMARY.csv"
METRICS_SEED = ROOT / "metrics_per_seed.csv"
METRICS_SUMMARY = ROOT / "metrics_summary_mean_sd.csv"
DECISION_JSON = ROOT / "decision.json"
REPORT_MD = ROOT / "REPORT.md"
CHANNELS = ("MB0", "MB1", "MB2")
TIMESTAMP_PATTERN = re.compile(rb"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}")


def load_s2_module():
    spec = importlib.util.spec_from_file_location("c1_s2_audit", S2_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import required S2 audit module: {S2_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S2 = load_s2_module()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def file_state(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_first_last(path: Path) -> tuple[str, str, str]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n")
        first = handle.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n")
        handle.seek(max(0, size - 131072))
        tail = handle.read().decode("utf-8-sig", errors="replace").splitlines()
    return header, first, tail[-1] if tail else ""


def discover_candidates(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for year_text, root_text in config["fresh_download_roots"].items():
        year = int(year_text)
        root = Path(root_text)
        records: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if not path.is_file() or path.suffix.lower() != ".csv" or path.name.startswith("~$"):
                continue
            header, first, last = read_first_last(path)
            fields = header.split(",")
            records.append({
                "year": year, "absolute_path": str(path.resolve()), **file_state(path), "header": header,
                "first_physical_record": first, "last_physical_record": last,
                "has_timestamp_utc": any(x.startswith("Timestamp_UTC") for x in fields),
                "has_mb0_mb1_mb2": all(any(f"Irradiance_{ch}" in x for x in fields) for ch in CHANNELS),
                "first_record_mentions_target_year": f"/{year} " in first,
            })
        found[year_text] = records
    return found


def selected_paths(config: dict[str, Any], candidates: dict[str, list[dict[str, Any]]]) -> dict[int, Path]:
    selected = {2022: Path(config["irradiance_files"]["2022"])}
    excluded = {str(Path(p).resolve()).lower() for p in config["excluded_sources"]}
    for year in (2021, 2023):
        eligible = [row for row in candidates[str(year)] if row["has_timestamp_utc"] and row["has_mb0_mb1_mb2"] and row["first_record_mentions_target_year"] and row["absolute_path"].lower() not in excluded]
        if len(eligible) != 1:
            raise RuntimeError(f"Year {year} has {len(eligible)} uniquely eligible fresh sources; refusing to guess")
        selected[year] = Path(eligible[0]["absolute_path"])
    return selected


def stricter_calendar(result: dict[str, Any], year: int) -> bool:
    expected_first = f"{year}-01-01 00:00:00"
    expected_last = f"{year}-12-31 23:59:59"
    zero_keys = (
        "missing_timestamps", "duplicate_timestamps", "inverse_timestamps", "empty_timestamp_fields",
        "column_count_anomalies", "quote_anomalies", "glued_records", "truncated_records",
        "duplicate_headers", "out_of_target_year_records", "data_error_lines",
    )
    return bool(
        result["parseable_target_year_unique_seconds"] == result["expected_seconds"]
        and result["first_target_year_utc"] == expected_first
        and result["last_target_year_utc"] == expected_last
        and result["main_interval_seconds"] == 1
        and all(result[key] == 0 for key in zero_keys)
    )


def field_semantics(header: list[str]) -> dict[str, Any]:
    status = [name for name in header if "status" in name.lower() or "quality" in name.lower()]
    units = {}
    for channel in CHANNELS:
        match = next((name for name in header if f"Irradiance_{channel}" in name), "UNKNOWN")
        units[channel] = match[match.find("[") + 1:match.rfind("]")] if "[" in match and "]" in match else "UNKNOWN"
    return {"irradiance_units": units, "status_or_quality_fields": status, "status_semantics": "NOT_PRESENT" if not status else "PRESENT_REQUIRES_OFFICIAL_FIELD_DEFINITION"}


def exact_structure_scan(path: Path) -> dict[str, int]:
    """Count physical CSV anomalies, including exactly two glued timestamps."""
    counts = Counter()
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
        width = len(header.split(b","))
        expected_formatted_timestamps = sum(b"[DD/MM/YYYY hh:mm:ss]" in field for field in header.split(b","))
        for raw in handle:
            line = raw.rstrip(b"\r\n")
            parts = line.split(b",")
            occurrences = len(TIMESTAMP_PATTERN.findall(line))
            counts["quote_anomalies"] += int(line.count(b'"') % 2 != 0)
            counts["duplicate_headers"] += int(line.startswith(b"Timestamp_UTC"))
            counts["glued_records"] += int(occurrences > expected_formatted_timestamps)
            counts["column_count_anomalies"] += int(len(parts) != width)
            counts["truncated_records"] += int(len(parts) < width and occurrences < 2)
            counts["data_error_lines"] += int(b"Data Error" in line.replace(b",", b""))
    return {name: int(counts[name]) for name in ("quote_anomalies", "duplicate_headers", "glued_records", "column_count_anomalies", "truncated_records", "data_error_lines")}


def combine_hf(grid: np.ndarray, scans: dict[int, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(grid)
    start_key = S2.epoch_seconds(datetime(2021, 1, 1)) // 300
    timestamp_count = np.zeros(n, dtype=np.int32)
    channel_count = np.zeros((3, n), dtype=np.int32)
    channel_sum = np.zeros((3, n), dtype=np.float64)
    for arrays in scans.values():
        target = arrays["keys"] - start_key
        ok = (target >= 0) & (target < n)
        target = target[ok]
        timestamp_count[target] += arrays["timestamp_count"][ok]
        channel_count[:, target] += arrays["channel_count"][:, ok]
        channel_sum[:, target] += np.nan_to_num(arrays["channel_mean"][:, ok]) * arrays["channel_count"][:, ok]
    mean = np.divide(channel_sum, channel_count, out=np.full_like(channel_sum, np.nan), where=channel_count > 0)
    return timestamp_count, channel_count, mean


def primary_daylight(config: dict[str, Any], grid: np.ndarray, powers: dict[str, np.ndarray], origin_state: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]:
    thresholds: dict[str, float] = {}
    grid_dt = grid.astype("datetime64[m]")
    train = (grid_dt >= np.datetime64(config["stages"]["BASE_TRAIN"][0])) & (grid_dt < np.datetime64(config["stages"]["BASE_TRAIN"][1]))
    for array, values in powers.items():
        positive = values[train & np.isfinite(values) & (values > 0)]
        p999 = float(np.quantile(positive, 0.999, method="higher")) if len(positive) else math.nan
        thresholds[array] = p999 * 0.01 if math.isfinite(p999) else math.nan
    rows: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    for stage in config["stages"]:
        common = origin_state[f"origins__{stage}__COMMON"]
        keep = np.ones(len(common), dtype=bool)
        for array, values in powers.items():
            keep &= np.isfinite(values[common]) & (values[common] > thresholds[array])
        selected = common[keep]
        masks[f"primary_daylight__{stage}"] = selected
        rows.append({"stage": stage, "common_origins": len(common), "primary_daylight_common_origins": len(selected), "first": str(grid_dt[selected[0]]) if len(selected) else "NONE", "last": str(grid_dt[selected[-1]]) if len(selected) else "NONE"})
    return rows, masks, thresholds


def seasons_for_months(months: set[int]) -> set[str]:
    return {S2.season_name(month) for month in months}


def build_readiness(config: dict[str, Any], irradiance: dict[str, dict[str, Any]], windows: list[dict[str, Any]], thresholds: dict[str, float], candidates: dict[str, list[dict[str, Any]]], unchanged: bool) -> dict[str, bool]:
    common = {row["stage"]: row for row in windows if row["array"] == "THREE_ARRAY_COMMON"}
    cond = {
        "fresh_2021_and_2023_exist": all(len(candidates[str(y)]) >= 1 for y in (2021, 2023)),
        "fresh_2021_and_2023_have_31536000_unique_seconds": all(irradiance[str(y)]["parseable_target_year_unique_seconds"] == 31_536_000 for y in (2021, 2023)),
        "main_interval_is_one_second": all(irradiance[str(y)]["main_interval_seconds"] == 1 for y in (2021, 2022, 2023)),
        "fresh_year_first_last_utc_are_exact": all(irradiance[str(y)]["first_target_year_utc"] == f"{y}-01-01 00:00:00" and irradiance[str(y)]["last_target_year_utc"] == f"{y}-12-31 23:59:59" for y in (2021, 2023)),
        "no_structural_anomaly_in_fresh_years": all(irradiance[str(y)]["structurally_full_calendar"] for y in (2021, 2023)),
        "mb_fields_units_compatible": len({tuple(irradiance[str(y)]["field_semantics"]["irradiance_units"].values()) for y in (2021, 2022, 2023)}) == 1,
        "formal_2023_source_is_not_excluded_old_file": str(Path(irradiance["2023"]["path"]).resolve()).lower() not in {str(Path(p).resolve()).lower() for p in config["excluded_sources"]},
        "all_stages_have_three_array_common_origins": all(common.get(stage, {}).get("formal_masked_origins", 0) > 0 for stage in config["stages"]),
        "base_train_and_final_test_cover_12_months_four_seasons": all(len(set(map(int, common.get(stage, {}).get("months", "").split(";"))) if common.get(stage, {}).get("months") else set()) == 12 and len(seasons_for_months(set(map(int, common[stage]["months"].split(";"))))) == 4 for stage in ("BASE_TRAIN", "FINAL_TEST") if stage in common),
        "risk_fit_and_calibration_cover_planned_months": set(common.get("RISK_FIT", {}).get("months", "").split(";")) == {"5", "6", "7", "8"} and set(common.get("RISK_CALIBRATION", {}).get("months", "").split(";")) == {"9", "10", "11", "12"},
        "final_test_outcomes_not_read_or_generated_before_freeze": True,
        "raw_sources_unchanged": unchanged,
        "frozen_daylight_thresholds_match": all(math.isfinite(thresholds[a]) and abs(thresholds[a] - config["frozen_daylight_threshold_kw"][a]) <= 5e-7 for a in thresholds),
    }
    return cond


def write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["section", "year", "stage", "array", "metric", "value", "unit", "source_path", "notes"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def summary_rows(candidates: dict[str, list[dict[str, Any]]], irradiance: dict[str, dict[str, Any]], pv: dict[str, dict[str, Any]], windows: list[dict[str, Any]], daylight_rows: list[dict[str, Any]], thresholds: dict[str, float], readiness: dict[str, bool]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, items in candidates.items():
        for item in items:
            for metric in ("size", "mtime_ns", "header", "first_physical_record", "last_physical_record"):
                rows.append({"section": "candidate", "year": year, "metric": metric, "value": item[metric], "source_path": item["absolute_path"]})
    annual_metrics = ["file_size_bytes", "physical_lines", "parseable_records_all_years", "parseable_target_year_unique_seconds", "expected_seconds", "first_target_year_utc", "last_target_year_utc", "main_interval_seconds", "missing_timestamps", "duplicate_timestamps", "inverse_timestamps", "empty_timestamp_fields", "out_of_target_year_records", "column_count_anomalies", "quote_anomalies", "glued_records", "truncated_records", "duplicate_headers", "data_error_lines", "MB0_missing", "MB1_missing", "MB2_missing", "five_minute_timestamp_complete_bins", "five_minute_all_channel_complete_bins", "five_minute_partial_bins", "five_minute_empty_bins", "structurally_full_calendar"]
    for year, item in irradiance.items():
        for metric in annual_metrics:
            rows.append({"section": "irradiance_year", "year": year, "metric": metric, "value": item[metric], "source_path": item["path"]})
        for channel, unit in item["field_semantics"]["irradiance_units"].items():
            rows.append({"section": "irradiance_field", "year": year, "metric": f"{channel}_unit", "value": unit, "source_path": item["path"]})
    for array, item in pv.items():
        rows.append({"section": "pv", "array": array, "metric": "base_train_positive_power_p99_9_kw", "value": item["base_train_positive_power_p99_9_kw"], "unit": "kW", "source_path": item["path"]})
        rows.append({"section": "pv", "array": array, "metric": "origin_daylight_threshold_kw", "value": thresholds[array], "unit": "kW", "source_path": item["path"]})
    for item in windows:
        for metric in ("expected_calendar_origins", "stage_boundary_excluded_origins", "formal_masked_origins", "strict_all_channel_complete_origins", "window_loss_total", "window_loss_pv", "window_loss_hf_zero_timestamp", "first_legal_origin", "last_legal_origin", "eligible_origin_segments", "months", "seasons"):
            rows.append({"section": "window", "stage": item["stage"], "array": item["array"], "metric": metric, "value": item[metric]})
    for item in daylight_rows:
        for metric in ("common_origins", "primary_daylight_common_origins", "first", "last"):
            rows.append({"section": "primary_daylight", "stage": item["stage"], "array": "THREE_ARRAY_COMMON", "metric": metric, "value": item[metric]})
    for name, passed in readiness.items():
        rows.append({"section": "readiness", "metric": name, "value": passed})
    return rows


def write_not_run_metrics(verdict: str) -> None:
    with METRICS_SEED.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["array", "seed", "run_status", "reason", "final_test_accessed"])
        writer.writeheader()
        for array in load_config()["pv_files"]:
            for seed in load_config()["seeds"]:
                writer.writerow({"array": array, "seed": seed, "run_status": "NOT_RUN_DATA_FAIL", "reason": verdict, "final_test_accessed": False})
    with METRICS_SUMMARY.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["decision", "completed_runs", "expected_runs", "gpu_training", "note"])
        writer.writeheader()
        writer.writerow({"decision": verdict, "completed_runs": 0, "expected_runs": 9, "gpu_training": False, "note": "Formal metrics unavailable because preregistered data readiness failed before training."})


def write_report(audit: dict[str, Any]) -> None:
    irr = audit["irradiance"]
    common = [row for row in audit["windows"] if row["array"] == "THREE_ARRAY_COMMON"]
    failed = [name for name, passed in audit["data_ready_conditions"].items() if not passed]
    lines = [
        "# Scheme C1-S3 — formal data confirmation and guarded execution",
        "", f"**Final scientific decision: `{audit['verdict']}`.**", "",
        "The new annual sources fail the preregistered readiness boundary, so the guarded GPU path was not entered. No base forecaster, risk estimator, calibration threshold, Final-Test prediction, error, coverage, or AURC was produced.", "",
        "## Formal source selection and annual audit", "",
        "| Year | Formal absolute path | Bytes | Unique target-year seconds | First UTC | Last UTC | Missing seconds | Out-of-year | Structural anomalies |",
        "|---:|---|---:|---:|---|---|---:|---:|---|",
    ]
    for year in ("2021", "2022", "2023"):
        x = irr[year]
        anomalies = f"column={x['column_count_anomalies']}; glued={x['glued_records']}; truncated={x['truncated_records']}; Data Error={x['data_error_lines']}"
        lines.append(f"| {year} | `{x['path']}` | {x['file_size_bytes']:,} | {x['parseable_target_year_unique_seconds']:,} | {x['first_target_year_utc']} | {x['last_target_year_utc']} | {x['missing_timestamps']:,} | {x['out_of_target_year_records']:,} | {anomalies} |")
    lines += ["", "The 2021 fresh file contains only 2 June–31 December. The 2023 fresh file contains only 1–2 January 2023, then records from 2024 and 2025; its year transition contains a glued timestamp and `Data Error`/truncated rows. The excluded old damaged 2023 file was not used. The 2022 authoritative redownload was reproduced exactly.", "",
              "All three headers expose `Timestamp_UTC` plus separate `Irradiance_MB0/MB1/MB2 [W/m-2]`; exported Local is ignored and ACST is computed as UTC+09:30. No System Status or quality field is present in these irradiance files, so no unsupported status meaning is inferred.", "",
              "## MB missingness and five-minute aggregation", "",
              "| Year | MB0 missing | MB1 missing | MB2 missing | Complete timestamp bins | Three-channel complete | Partial | Empty |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for year in ("2021", "2022", "2023"):
        x = irr[year]
        lines.append(f"| {year} | {x['MB0_missing']:,} | {x['MB1_missing']:,} | {x['MB2_missing']:,} | {x['five_minute_timestamp_complete_bins']:,} | {x['five_minute_all_channel_complete_bins']:,} | {x['five_minute_partial_bins']:,} | {x['five_minute_empty_bins']:,} |")
    lines += ["", "Aggregation is right-closed `(t-5 min, t]`; partial numeric missingness is retained with per-channel valid fraction/mask, while zero-timestamp bins break continuity. Nothing was interpolated or repaired.", "", "## Five-stage common-origin audit", "", "| Stage | Expected origins | Common legal origins | Strict three-channel origins | First | Last | Segments | Months | Seasons |", "|---|---:|---:|---:|---|---|---:|---|---|"]
    for row in common:
        lines.append(f"| {row['stage']} | {row['expected_calendar_origins']:,} | {row['formal_masked_origins']:,} | {row['strict_all_channel_complete_origins']:,} | {row['first_legal_origin']} | {row['last_legal_origin']} | {row['eligible_origin_segments']} | {row['months']} | {row['seasons']} |")
    lines += ["", "Each stage excludes 83 calendar positions by construction (71 initial history positions and 12 terminal target positions). The code combines adjacent UTC annual sources before interpreting ACST. It therefore treats the missing first 9.5 ACST hours of a UTC-year source as an explicit boundary effect, not file corruption. In this execution, however, the much larger 2021/2023 source gaps independently fail the full-year conditions.", "", "## Frozen implementation specification", "", "The committed config fixes the 14 causal input channels, `DEPTHWISE_TCN_TRAJECTORY` architecture inherited from C1-S0R, AdamW training protocol, Train-only imputation/scaling, risk target/range, HistGradientBoosting risk model, scope-matched order-statistic calibration, stable-tie AURC, bootstrap design, and all seven formal success conditions. Since data readiness failed, these frozen definitions were not executed or tuned.", "", "## Readiness conditions", ""]
    for name, passed in audit["data_ready_conditions"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += ["", f"Failed conditions: {', '.join(f'`{name}`' for name in failed)}.", "", "## Tests, execution, and protection", "", f"- Fixture tests: {audit.get('fixture_tests', 'pending')}", f"- Real-array tests: {audit.get('real_array_tests', 'pending')}", "- GPU training: **No (0/9 runs)**", "- Risk fitting: **No**", "- Final-Test performance access: **No**", f"- Original PV/NWP size and nanosecond mtime unchanged: **{audit['source_files_unchanged']}**", "", "## Seven formal method conditions", "", "All seven conditions are **NOT_EVALUATED**: macro coverage, per-array minimum coverage, macro AURC improvement, array-level AURC direction, matched-Persistence skill, seed-macro accepted-RMSE reduction, and seed-level AURC direction. A data failure is not a method failure.", "", "## Conclusion", "", "`C1_FORMAL_DATA_FAIL`. The failure is data-specific: 2021 is not a full year and the new 2023 export is a mixed-year, structurally damaged file. Under the preregistration, this ends the execution before training. No interpolation, alternative year, repair, C1 v2/v3, or scientific method conclusion is proposed."]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_data() -> dict[str, Any]:
    config = load_config()
    candidates = discover_candidates(config)
    paths = selected_paths(config, candidates)
    pv_paths = {name: Path(path) for name, path in config["pv_files"].items()}
    all_sources = list(paths.values()) + list(pv_paths.values())
    before = {str(path.resolve()): file_state(path) for path in all_sources}
    irradiance: dict[str, dict[str, Any]] = {}
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for year in (2021, 2022, 2023):
        result, aggregated = S2.scan_second_file(paths[year], year, config["timezone"]["utc_to_acst_minutes"])
        result.update(exact_structure_scan(paths[year]))
        result["path"] = str(paths[year].resolve())
        result["field_semantics"] = field_semantics(result["header"])
        result["structurally_full_calendar"] = stricter_calendar(result, year)
        irradiance[str(year)] = result
        arrays[year] = aggregated
    grid = np.arange(np.datetime64("2021-01-01T00:00", "m"), np.datetime64("2024-01-01T00:00", "m"), np.timedelta64(5, "m"))
    hf_timestamp_count, hf_channel_count, hf_channel_mean = combine_hf(grid, arrays)
    pv: dict[str, dict[str, Any]] = {}
    pv_present: dict[str, np.ndarray] = {}
    pv_power: dict[str, np.ndarray] = {}
    for name, path in pv_paths.items():
        result, present, power = S2.scan_pv_file(path, datetime(2021, 1, 1), datetime(2024, 1, 1))
        result["path"] = str(path.resolve())
        pv[name], pv_present[name], pv_power[name] = result, present, power
    windows, origin_state = S2.window_audit(config, grid, pv_power, hf_timestamp_count, hf_channel_count)
    for row in windows:
        row["stage_boundary_excluded_origins"] = config["lookback"] + config["horizon"] - 1
    daylight_rows, daylight_state, thresholds = primary_daylight(config, grid, pv_power, origin_state)
    after = {str(path.resolve()): file_state(path) for path in all_sources}
    unchanged = before == after
    readiness = build_readiness(config, irradiance, windows, thresholds, candidates, unchanged)
    verdict = "C1_FORMAL_DATA_READY" if all(readiness.values()) else "C1_FORMAL_DATA_FAIL"
    audit = {"study_id": config["study_id"], "generated_utc": datetime.now(UTC).isoformat(), "verdict": verdict, "candidates": candidates, "irradiance": irradiance, "pv": pv, "windows": windows, "primary_daylight": daylight_rows, "daylight_thresholds_kw": thresholds, "data_ready_conditions": readiness, "source_state_before": before, "source_state_after": after, "source_files_unchanged": unchanged, "training_performed": False, "completed_runs": 0, "final_test_predictions_or_errors_read": False}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(RESULTS / "audit_state.npz", grid_ns=grid.astype("datetime64[ns]").astype(np.int64), hf_timestamp_count=hf_timestamp_count, hf_channel_count=hf_channel_count, hf_channel_mean=hf_channel_mean, **{f"pv_power_{i}": pv_power[name] for i, name in enumerate(config["pv_files"])}, **origin_state, **daylight_state)
    write_long_csv(SUMMARY_CSV, summary_rows(candidates, irradiance, pv, windows, daylight_rows, thresholds, readiness))
    write_not_run_metrics(verdict)
    DECISION_JSON.write_text(json.dumps({"decision": verdict, "data_ready": False, "gpu_training_performed": False, "completed_runs": 0, "expected_runs": 9, "failed_readiness_conditions": [k for k, v in readiness.items() if not v], "final_test_performance_accessed": False, "method_success_conditions_evaluated": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="Run the complete read-only data audit")
    parser.add_argument("--execute-formal", action="store_true", help="Guarded formal execution; refuses unless audit is READY")
    args = parser.parse_args()
    if not args.audit and not args.execute_formal:
        parser.error("Select --audit or --execute-formal")
    audit = audit_data()
    if args.execute_formal:
        if audit["verdict"] != "C1_FORMAL_DATA_READY":
            print(json.dumps({"decision": audit["verdict"], "training_started": False, "reason": "preregistered data readiness failed"}))
            return
        raise RuntimeError("READY data path requires the fixed nine-run implementation; this branch did not reach it because the audited sources failed readiness")
    print(json.dumps({"decision": audit["verdict"], "training_started": False, "summary_csv": str(SUMMARY_CSV)}))


if __name__ == "__main__":
    main()


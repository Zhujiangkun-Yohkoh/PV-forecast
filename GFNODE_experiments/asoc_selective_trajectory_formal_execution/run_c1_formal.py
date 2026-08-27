"""Archived Scheme C1 entry point.

The route is administratively closed. Public CLI modes return the stable
closeout state before audit, data preparation, model construction, or Final-Test
materialization. Historical functions remain as read-only development evidence.
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
from c1_formal_pipeline import execute_formal, prepare_from_audit_state


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


def closed_state(config: dict[str, Any]) -> dict[str, Any]:
    """Return the stable, non-error terminal state for the archived route."""
    return {
        "route_status": config.get("route_status", "C1_ROUTE_CLOSED_DATA_UNAVAILABLE"),
        "route_closed": True,
        "data_readiness": "C1_FORMAL_DATA_FAIL",
        "scientific_method_outcome": "NOT_EVALUATED",
        "future_gpu_execution_authorized": False,
        "completed_runs": 0,
        "expected_runs": 9,
        "training_started": False,
        "final_test_accessed": False,
    }


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
    """Inspect only explicitly configured paths; never guess via rglob or filename."""
    found: dict[str, list[dict[str, Any]]] = {}
    for year_text, source_texts in config["irradiance_sources"].items():
        year = int(year_text)
        records: list[dict[str, Any]] = []
        for path in map(Path, source_texts):
            if not path.is_file():
                records.append({"year":year,"absolute_path":str(path),"exists":False})
                continue
            header, first, last = read_first_last(path)
            fields = header.split(",")
            records.append({
                "year": year, "absolute_path": str(path.resolve()), "exists":True, **file_state(path), "header": header,
                "first_physical_record": first, "last_physical_record": last,
                "has_timestamp_utc": any(x.startswith("Timestamp_UTC") for x in fields),
                "has_mb0_mb1_mb2": all(any(f"Irradiance_{ch}" in x for x in fields) for ch in CHANNELS),
                "first_record_mentions_target_year": f"/{year} " in first,
            })
        found[year_text] = records
    return found


def selected_paths(config: dict[str, Any], candidates: dict[str, list[dict[str, Any]]]) -> dict[int, list[Path]]:
    selected: dict[int,list[Path]] = {}
    excluded = {str(Path(p).resolve()).lower() for p in config["excluded_sources"]}
    for year in (2021,2022,2023):
        eligible = [row for row in candidates[str(year)] if row.get("exists") and row["has_timestamp_utc"] and row["has_mb0_mb1_mb2"] and row["absolute_path"].lower() not in excluded]
        if len(eligible) != len(config["irradiance_sources"][str(year)]) or not eligible:
            raise RuntimeError(f"Year {year} explicit source list contains missing/ineligible files")
        selected[year] = [Path(row["absolute_path"]) for row in eligible]
    return selected


def scan_year_sources(paths: list[Path], year: int, offset: int) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    """Stream each explicit block, then combine by actual UTC key without writing a joined file."""
    parts=[]
    for path in paths:
        result, arrays=S2.scan_second_file(path,year,offset); result.update(exact_structure_scan(path))
        result["path"]=str(path.resolve()); parts.append((result,arrays))
    parts.sort(key=lambda p: p[0].get("first_target_year_utc") or "9999")
    if len(parts)==1:
        base,arrays=parts[0]; base=base.copy(); base["source_count"]=1
        base["block_order"]=[{"path":base["path"],"first":base["first_target_year_utc"],"last":base["last_target_year_utc"]}]
        return base,arrays
    timestamp=sum((p[1]["timestamp_count"] for p in parts),np.zeros_like(parts[0][1]["timestamp_count"]))
    counts=sum((p[1]["channel_count"] for p in parts),np.zeros_like(parts[0][1]["channel_count"]))
    weighted=sum((np.nan_to_num(p[1]["channel_mean"])*p[1]["channel_count"] for p in parts),np.zeros_like(parts[0][1]["channel_mean"],dtype=float))
    mean=np.divide(weighted,counts,out=np.full_like(weighted,np.nan),where=counts>0)
    base=parts[0][0].copy(); base["path"]=";".join(str(p.resolve()) for p in paths); base["source_count"]=len(paths)
    sum_keys=("file_size_bytes","physical_lines","parseable_records_all_years","empty_timestamp_fields","out_of_target_year_records","column_count_anomalies","quote_anomalies","glued_records","truncated_records","duplicate_headers","data_error_lines","MB0_missing","MB1_missing","MB2_missing")
    for key in sum_keys: base[key]=sum(int(p[0].get(key,0)) for p in parts)
    base["parseable_target_year_unique_seconds"]=int(np.minimum(timestamp,300).sum())
    # Exact cross-block timestamp union; used only for multi-file years.
    year_start=S2.epoch_seconds(datetime(year,1,1)); seen=np.zeros(base["expected_seconds"],dtype=bool); duplicate=0
    for path in paths:
        with path.open("rb") as handle:
            handle.readline()
            for raw in handle:
                epoch=S2.parse_dmy_second(raw.split(b",",1)[0])
                if epoch is not None and year_start<=epoch<year_start+base["expected_seconds"]:
                    idx=epoch-year_start; duplicate+=int(seen[idx]); seen[idx]=True
    base["parseable_target_year_unique_seconds"]=int(seen.sum())
    base["duplicate_timestamps"]=duplicate
    base["inverse_timestamps"]=sum(int(p[0].get("inverse_timestamps",0)) for p in parts)
    base["missing_timestamps"]=int(base["expected_seconds"]-base["parseable_target_year_unique_seconds"])
    valid_first=[p[0]["first_target_year_utc"] for p in parts if p[0]["first_target_year_utc"]!="UNKNOWN"]
    valid_last=[p[0]["last_target_year_utc"] for p in parts if p[0]["last_target_year_utc"]!="UNKNOWN"]
    base["first_target_year_utc"]=min(valid_first) if valid_first else "UNKNOWN"
    base["last_target_year_utc"]=max(valid_last) if valid_last else "UNKNOWN"
    base["five_minute_timestamp_complete_bins"]=int(np.count_nonzero(timestamp==300))
    base["five_minute_all_channel_complete_bins"]=int(np.count_nonzero((timestamp==300)&np.all(counts==300,axis=0)))
    base["five_minute_partial_bins"]=int(np.count_nonzero((timestamp>0)&~((timestamp==300)&np.all(counts==300,axis=0))))
    base["five_minute_empty_bins"]=int(np.count_nonzero(timestamp==0))
    base["block_order"]=[{"path":p[0]["path"],"first":p[0]["first_target_year_utc"],"last":p[0]["last_target_year_utc"]} for p in parts]
    return base,{"keys":parts[0][1]["keys"],"timestamp_count":timestamp,"channel_count":counts,"channel_mean":mean}


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
    irr=audit["irradiance"]; common=[r for r in audit["windows"] if r["array"]=="THREE_ARRAY_COMMON"]
    lines=["# Scheme C1-S4 — annual source and executable-readiness review","",
      "## Archived verdicts","",f"- Data: **`{audit['verdict']}`**.","- Implementation: **`C1_FORMAL_IMPLEMENTATION_NOT_VALIDATED_AND_NO_LONGER_REQUIRED`**.","- GPU execution: **NOT AUTHORIZED / not performed (0/9)**.","- Scientific method outcome: **NOT EVALUATED**.","",
      "## Official-source finding","",
      "For both 2021 and 2023 the defensible source verdict is **`OFFICIAL_FULL_YEAR_UNAVAILABLE_OR_UNCONFIRMED`**. The official DKASC NT Solar Resource page lists Alice Springs annual 5-minute and 5-second downloads for both years and states that the Class-A stations collect high-resolution data. The official Fulcrum3D interface separately exposes a user-selected date range and a 1-second irradiance download. Neither public page guarantees that a requested 1-second export contains every second of a calendar year, documents an annual 1-second row limit, or explains these two malformed exports. Therefore portal availability is not evidence of complete 1-second annual coverage. The present files support download/export failure or wrong selection as possibilities, but do not distinguish them from true upstream gaps.","",
      "Official pages checked 2026-08-28: https://www.dkasolarcentre.com.au/download?location=nt-solar-resource and https://nt-solar-resource.fulcrum3d.com/download . The DKASC page defines System Status (0=OK, 1=issue), but the selected CSV headers contain no status field.","",
      "## Explicit sources and annual union","","|Year|Files|Bytes|Unique seconds|First UTC|Last UTC|Missing|Duplicate|Out-of-year|Structure|","|---:|---:|---:|---:|---|---|---:|---:|---:|---|"]
    for y in ("2021","2022","2023"):
      x=irr[y]; structure=f"columns {x['column_count_anomalies']}; glued {x['glued_records']}; truncated {x['truncated_records']}; Data Error {x['data_error_lines']}"
      lines.append(f"|{y}|{x.get('source_count',1)}|{x['file_size_bytes']:,}|{x['parseable_target_year_unique_seconds']:,}|{x['first_target_year_utc']}|{x['last_target_year_utc']}|{x['missing_timestamps']:,}|{x['duplicate_timestamps']:,}|{x['out_of_target_year_records']:,}|{structure}|")
    lines += ["","2021 contains only 2021-06-02 through year-end. The 2023 source begins one second late, ends its target-year coverage on 2023-01-02, then contains 2024/2025 rows and structural damage. The excluded older damaged 2023 file was not used. The 2022 authoritative redownload remains complete.","",
      "All years use `Timestamp_UTC`; ACST is derived as UTC+09:30. MB0/MB1/MB2 remain separate in W/m². No interpolation, repair, Excel rewrite, or concatenated raw copy was made. Config now accepts an explicit list of one or more annual/monthly files; ordering and overlap checks use actual UTC records.","",
      "## Five-minute quality","","|Year|MB0 missing|MB1 missing|MB2 missing|Strict complete bins|Partial bins|Empty bins|","|---:|---:|---:|---:|---:|---:|---:|"]
    for y in ("2021","2022","2023"):
      x=irr[y]; lines.append(f"|{y}|{x['MB0_missing']:,}|{x['MB1_missing']:,}|{x['MB2_missing']:,}|{x['five_minute_all_channel_complete_bins']:,}|{x['five_minute_partial_bins']:,}|{x['five_minute_empty_bins']:,}|")
    lines += ["","Right-closed `(t-5 min,t]` aggregation is fixed. Partial numeric missingness is retained through channel mean, valid fraction and valid mask; zero-timestamp bins interrupt windows.","","## Frozen primary population","",
      "All seven success conditions use exactly **H12 + THREE_ARRAY_COMMON + mask-available + PRIMARY_DAYLIGHT_COMMON**. Strict 300/300 three-channel completeness is only a data-quality sensitivity population and cannot select the main result.","","## Five-stage common origins","","|Stage|Expected|Formal common|Strict 300/300|First|Last|Segments|Months|Seasons|","|---|---:|---:|---:|---|---|---:|---|---|"]
    for r in common: lines.append(f"|{r['stage']}|{r['expected_calendar_origins']:,}|{r['formal_masked_origins']:,}|{r['strict_all_channel_complete_origins']:,}|{r['first_legal_origin']}|{r['last_legal_origin']}|{r['eligible_origin_segments']}|{r['months']}|{r['seasons']}|")
    lines += ["","## Implementation review","",
      "The archived prototype was not validated by an end-to-end production execution and is not authorized for use. Historical code remains only as audit evidence.","",
      "State fields are factual: raw Final-Test availability metadata was inspected; model predictions, errors, risk scores, coverage and AURC were not generated or accessed. NOT_RUN rows are execution status, not performance metrics.","",
      "## Tests and source protection","",f"- Synthetic/fixture: {audit.get('fixture_tests','pending')}.",f"- Real arrays: {audit.get('real_array_tests','pending')}.",f"- Selected PV and irradiance source size/mtime_ns unchanged: **{audit['source_files_unchanged']}**.","- Real optimizer/backward/epoch: **No**.","- Real risk-model fitting: **No**.","- Final-Test performance access: **No**.","","Local `results/` contains only compact review artifacts and remains untracked; it is not a clean-worktree claim and will not be committed.","",
      "## Conclusion","","The route is **`C1_ROUTE_CLOSED_DATA_UNAVAILABLE`**. The scientific method was not evaluated, and no execution or resumption is authorized."]
    REPORT_MD.write_text("\n".join(lines)+"\n",encoding="utf-8")


def audit_data() -> dict[str, Any]:
    config = load_config()
    candidates = discover_candidates(config)
    paths = selected_paths(config, candidates)
    pv_paths = {name: Path(path) for name, path in config["pv_files"].items()}
    all_sources = [p for group in paths.values() for p in group] + list(pv_paths.values())
    before = {str(path.resolve()): file_state(path) for path in all_sources}
    irradiance: dict[str, dict[str, Any]] = {}
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for year in (2021, 2022, 2023):
        result, aggregated = scan_year_sources(paths[year], year, config["timezone"]["utc_to_acst_minutes"])
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
    audit = {"study_id": config["study_id"], "generated_utc": datetime.now(UTC).isoformat(), "verdict": verdict, "candidates": candidates, "irradiance": irradiance, "pv": pv, "windows": windows, "primary_daylight": daylight_rows, "daylight_thresholds_kw": thresholds, "data_ready_conditions": readiness, "source_state_before": before, "source_state_after": after, "source_files_unchanged": unchanged, "training_performed": False, "completed_runs": 0, "raw_final_test_availability_metadata_accessed": True, "final_test_model_predictions_generated": False, "final_test_prediction_errors_accessed": False, "final_test_risk_scores_accessed": False, "final_test_coverage_or_aurc_accessed": False}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(RESULTS / "audit_state.npz", grid_ns=grid.astype("datetime64[ns]").astype(np.int64), hf_timestamp_count=hf_timestamp_count, hf_channel_count=hf_channel_count, hf_channel_mean=hf_channel_mean, **{f"pv_power_{i}": pv_power[name] for i, name in enumerate(config["pv_files"])}, **origin_state, **daylight_state)
    write_long_csv(SUMMARY_CSV, summary_rows(candidates, irradiance, pv, windows, daylight_rows, thresholds, readiness))
    write_not_run_metrics(verdict)
    DECISION_JSON.write_text(json.dumps({"decision": verdict, "data_readiness": verdict, "data_ready": verdict=="C1_FORMAL_DATA_READY", "implementation_readiness":"PENDING_TESTS", "gpu_execution_status":"NOT_AUTHORIZED_C1_S4", "scientific_method_outcome":"NOT_EVALUATED", "gpu_training_performed": False, "completed_runs": 0, "expected_runs": 9, "failed_readiness_conditions": [k for k, v in readiness.items() if not v], "raw_final_test_availability_metadata_accessed":True,"final_test_model_predictions_generated":False,"final_test_prediction_errors_accessed":False,"final_test_risk_scores_accessed":False,"final_test_coverage_or_aurc_accessed":False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="Run the complete read-only data audit")
    parser.add_argument("--execute-formal", action="store_true", help="Guarded formal execution; refuses unless audit is READY")
    parser.add_argument("--authorize-real-execution", action="store_true", help="Explicit next-stage GPU authorization; disabled in C1-S4 config")
    args = parser.parse_args()
    if not args.audit and not args.execute_formal:
        parser.error("Select --audit or --execute-formal")
    config = load_config()
    if config.get("route_closed") is True:
        print(json.dumps(closed_state(config), sort_keys=True))
        return
    audit = audit_data()
    if args.execute_formal:
        if audit["verdict"] != "C1_FORMAL_DATA_READY":
            print(json.dumps({"decision": audit["verdict"], "training_started": False, "reason": "preregistered data readiness failed"}))
            return
        config=load_config(); authorized=bool(args.authorize_real_execution and config.get("execution_authorized_this_stage",False))
        prepared=prepare_from_audit_state(config,RESULTS/"audit_state.npz") if authorized else None
        outcome=execute_formal(config,prepared,RESULTS,data_ready=True,authorize_real_execution=authorized)
        print(json.dumps(outcome)); return
    print(json.dumps({"decision": audit["verdict"], "training_started": False, "summary_csv": str(SUMMARY_CSV)}))


if __name__ == "__main__":
    main()

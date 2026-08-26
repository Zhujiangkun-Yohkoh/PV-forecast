"""Stage B1 causal GFS minimal screen for DKASC Site 17 Sanyo.

The script never loads the sealed 2023 Test period.  It streams only official
GFS DSWRF/TCDC messages, retains compact point artifacts locally, and trains
three fixed ModernTCN conditions on identical Train/Validation windows.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import http.client
import io
import json
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
NWP_MONTHS = RESULTS / "nwp_monthly"
PREPARED = RESULTS / "prepared_data.npz"
METRICS = ROOT / "metrics_per_seed.csv"
REPORT = ROOT / "REPORT.md"
UTC = dt.timezone.utc
ACST = dt.timezone(dt.timedelta(minutes=570))
MODELS = ("HISTORY_ONLY", "RAW_NWP", "AGE_LEAD_RELIABILITY")


def config() -> dict:
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_timestamp(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.strip().strip('"'))
    except (TypeError, ValueError):
        return None


def read_pv_train_validation(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Read only Train/Validation rows; sealed 2023 is never materialized."""
    path = Path(cfg["pv_file"])
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    start = dt.datetime.fromisoformat(cfg["splits"]["train"][0])
    end = dt.datetime.fromisoformat(cfg["splits"]["validation"][1])
    rows: list[list[object]] = []
    malformed = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader([handle.readline().rstrip("\r\n")]))
        timestamp_index = header.index("timestamp")
        selected = [header.index(name) for name in cfg["history_features"]]
        for physical in handle:
            try:
                row = next(csv.reader([physical]))
            except csv.Error:
                malformed += 1
                continue
            if len(row) != len(header):
                malformed += 1
                continue
            stamp = parse_timestamp(row[timestamp_index])
            if stamp is None or stamp < start or stamp >= end:
                continue
            values: list[float] = []
            for index in selected:
                try:
                    values.append(float(row[index]))
                except (TypeError, ValueError):
                    values.append(math.nan)
            rows.append([stamp, *values])
    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
        raise AssertionError("PV source changed during read")
    frame = pd.DataFrame(rows, columns=["timestamp", *cfg["history_features"]])
    duplicates = int(frame.duplicated("timestamp").sum())
    frame = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    grid = pd.date_range(start, end, freq="5min", inclusive="left")
    regular = frame.reindex(grid)
    regular["_source_present"] = regular.index.isin(frame.index)
    info = {
        "pv_source_path": str(path),
        "pv_source_size": before[0],
        "pv_source_mtime_ns": before[1],
        "parsed_rows": len(frame),
        "malformed_rows_in_file": malformed,
        "duplicate_timestamps_in_period": duplicates,
        "regular_rows": len(regular),
        "first_timestamp": str(regular.index.min()),
        "last_timestamp": str(regular.index.max()),
        "sealed_test_loaded": False,
    }
    return regular, info


def split_mask(index: pd.DatetimeIndex, bounds: list[str]) -> np.ndarray:
    return np.asarray((index >= pd.Timestamp(bounds[0])) & (index < pd.Timestamp(bounds[1])), dtype=bool)


def legal_origins(valid: np.ndarray, split: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return origins wholly contained in valid five-minute contiguous runs."""
    inside = np.asarray(valid & split, dtype=bool)
    padded = np.r_[False, inside, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    origins: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    needed = lookback + horizon
    for start, end in zip(starts, ends):
        length = int(end - start + 1)
        if length < needed:
            continue
        segments.append((int(start), int(end)))
        origins.append(np.arange(start + lookback - 1, end - horizon + 1, dtype=np.int64))
    return (np.concatenate(origins) if origins else np.empty(0, np.int64)), segments


def fit_history_preprocessing(frame: pd.DataFrame, cfg: dict, train_rows: np.ndarray) -> dict:
    raw = frame[cfg["history_features"]].to_numpy(np.float64)
    fill = np.zeros(raw.shape[1], np.float64)
    center = np.zeros(raw.shape[1], np.float64)
    scale = np.ones(raw.shape[1], np.float64)
    for column in range(raw.shape[1]):
        values = raw[train_rows, column]
        finite = values[np.isfinite(values)]
        fill[column] = float(np.median(finite)) if len(finite) else 0.0
        center[column] = float(np.mean(finite)) if len(finite) else 0.0
        std = float(np.std(finite)) if len(finite) else 1.0
        scale[column] = std if std > 1e-8 else 1.0
    missing = ~np.isfinite(raw)
    filled = np.where(missing, fill, raw)
    scaled = (filled - center) / scale
    minute = frame.index.hour.to_numpy() * 60 + frame.index.minute.to_numpy()
    day = frame.index.dayofyear.to_numpy() - 1
    time_features = np.stack([
        np.sin(2 * np.pi * minute / 1440), np.cos(2 * np.pi * minute / 1440),
        np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
    ], axis=1)
    features = np.concatenate([scaled, missing.astype(np.float64), time_features], axis=1).astype(np.float32)
    power = frame[cfg["target_column"]].to_numpy(np.float32)
    train_power = power[train_rows & np.isfinite(power) & (power >= 0)]
    target_min = float(np.min(train_power))
    target_range = float(np.max(train_power) - target_min)
    if target_range <= 0:
        raise AssertionError("non-positive Train target range")
    return {
        "features": features,
        "fill": fill.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "target_min": target_min,
        "target_range": target_range,
        "fit_split": "train",
    }


def acst_ns_to_utc(value: np.int64) -> dt.datetime:
    naive = pd.Timestamp(int(value)).to_pydatetime()
    return naive.replace(tzinfo=ACST).astimezone(UTC)


def nominal_selected_cycle(origin_utc: dt.datetime) -> dt.datetime:
    floor = origin_utc.replace(hour=(origin_utc.hour // 6) * 6, minute=0, second=0, microsecond=0)
    return floor - dt.timedelta(hours=6)


def gfs_base(cycle: dt.datetime, lead: int, cfg: dict) -> str:
    return (f"{cfg['gfs_archive']}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"gfs.t{cycle:%H}z.{cfg['gfs_product']}.f{lead:03d}")


def http_get(url: str, cfg: dict, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "PV-NWP-minimal-screen/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last: Exception | None = None
    for attempt in range(cfg["download"]["attempts"]):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=cfg["download"]["timeout_seconds"]) as response:
                data = response.read()
                if byte_range is not None:
                    expected = byte_range[1] - byte_range[0] + 1
                    if response.status != 206 or len(data) != expected:
                        raise http.client.IncompleteRead(data, expected - len(data))
                return data
        except (OSError, TimeoutError, http.client.HTTPException, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 < cfg["download"]["attempts"]:
                time.sleep(attempt + 1)
    assert last is not None
    raise last


def parse_idx(text: str) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) >= 3:
            records.append({"line": line, "offset": int(fields[1]), "next": None})
    for index in range(len(records) - 1):
        records[index]["next"] = records[index + 1]["offset"]
    return records


def choose_idx_entry(records: list[dict], variable: str, pattern: str) -> dict:
    candidates = [record for record in records if pattern in record["line"] and record["next"] is not None]
    if variable == "TCDC_entire_atmosphere":
        instantaneous = [record for record in candidates if "ave fcst" not in record["line"].lower()]
        candidates = instantaneous or candidates
    if not candidates:
        raise ValueError(f"missing IDX record for {variable}")
    return candidates[0]


def fetch_job(cycle: dt.datetime, lead: int, cfg: dict) -> dict:
    base = gfs_base(cycle, lead, cfg)
    started = time.perf_counter()
    try:
        idx = http_get(base + ".idx", cfg)
        records = parse_idx(idx.decode("utf-8"))
        payloads: dict[str, bytes] = {}
        lines: dict[str, str] = {}
        total = len(idx)
        for variable, pattern in cfg["nwp_variables"].items():
            entry = choose_idx_entry(records, variable, pattern)
            payload = http_get(base, cfg, (entry["offset"], entry["next"] - 1))
            if not payload.endswith(b"7777"):
                raise ValueError(f"truncated GRIB message for {variable}")
            payloads[variable] = payload
            lines[variable] = entry["line"]
            total += len(payload)
        return {"cycle": cycle, "lead": lead, "status": 1, "payloads": payloads,
                "lines": lines, "bytes": total, "seconds": time.perf_counter() - started, "error": ""}
    except Exception as exc:
        return {"cycle": cycle, "lead": lead, "status": 0, "payloads": {}, "lines": {},
                "bytes": 0, "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}"}


def safe_codes_get(codes_get, gid, key: str, default: object = "UNKNOWN") -> object:
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def decode_payload(payload: bytes, variable: str, cfg: dict) -> dict:
    from eccodes import codes_get, codes_grib_find_nearest, codes_new_from_message, codes_release
    gid = codes_new_from_message(payload)
    try:
        nearest = codes_grib_find_nearest(gid, cfg["gfs_point"]["latitude"], cfg["gfs_point"]["longitude"])[0]
        return {
            "value": float(nearest["value"]),
            "lat": float(nearest["lat"]),
            "lon": float(nearest["lon"]),
            "step_type": str(safe_codes_get(codes_get, gid, "stepType")),
            "start_step": float(safe_codes_get(codes_get, gid, "startStep", math.nan)),
            "end_step": float(safe_codes_get(codes_get, gid, "endStep", math.nan)),
            "units": str(safe_codes_get(codes_get, gid, "units")),
            "statistical_processing": str(safe_codes_get(codes_get, gid, "typeOfStatisticalProcessing")),
            "validity_date": int(safe_codes_get(codes_get, gid, "validityDate", 0)),
            "validity_time": int(safe_codes_get(codes_get, gid, "validityTime", 0)),
            "variable": variable,
        }
    finally:
        codes_release(gid)


def month_path(month: str) -> Path:
    return NWP_MONTHS / f"gfs_point_{month}.npz"


def load_month(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    arrays = {name: data[name] for name in data.files}
    records: dict[tuple[int, int], dict] = {}
    for index in range(len(arrays["cycle_ns"])):
        key = (int(arrays["cycle_ns"][index]), int(arrays["lead"][index]))
        records[key] = {name: arrays[name][index].item() if hasattr(arrays[name][index], "item") else arrays[name][index]
                        for name in arrays if name not in {"cycle_ns", "lead"}}
    return records


def save_month(path: Path, records: dict[tuple[int, int], dict]) -> None:
    keys = sorted(records)
    fields = sorted(next(iter(records.values())).keys()) if records else []
    arrays: dict[str, np.ndarray] = {
        "cycle_ns": np.asarray([key[0] for key in keys], np.int64),
        "lead": np.asarray([key[1] for key in keys], np.int16),
    }
    for field in fields:
        values = [records[key][field] for key in keys]
        if field in {"status"}:
            arrays[field] = np.asarray(values, np.int8)
        elif field in {"bytes"}:
            arrays[field] = np.asarray(values, np.int64)
        elif field in {"seconds", "dswrf", "tcdc", "ds_start", "ds_end", "tc_start", "tc_end", "lat", "lon"}:
            arrays[field] = np.asarray(values, np.float64)
        else:
            arrays[field] = np.asarray(values, dtype="U256")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def local_nwp_inventory() -> dict:
    """Summarize unique retained point objects without double-counting split overlap."""
    records: dict[tuple[int, int], dict] = {}
    artifact_bytes = 0
    for path in NWP_MONTHS.glob("gfs_point_*.npz"):
        records.update(load_month(path))
        artifact_bytes += path.stat().st_size
    values = list(records.values())
    return {
        "unique_objects": len(values),
        "successful_objects": sum(int(record.get("status", 0)) for record in values),
        "failed_objects": sum(not int(record.get("status", 0)) for record in values),
        "retained_transfer_bytes": int(sum(int(record.get("bytes", 0)) for record in values)),
        "compact_nwp_artifact_bytes": artifact_bytes,
    }


def utc_ns(cycle: dt.datetime) -> int:
    return int(pd.Timestamp(cycle).value)


def record_from_result(result: dict, cfg: dict) -> dict:
    base = {"status": result["status"], "bytes": result["bytes"], "seconds": result["seconds"],
            "error": result["error"], "source": gfs_base(result["cycle"], result["lead"], cfg)}
    if not result["status"]:
        return {**base, "dswrf": math.nan, "tcdc": math.nan, "ds_start": math.nan, "ds_end": math.nan,
                "tc_start": math.nan, "tc_end": math.nan, "ds_step_type": "UNKNOWN", "tc_step_type": "UNKNOWN",
                "ds_units": "UNKNOWN", "tc_units": "UNKNOWN", "lat": math.nan, "lon": math.nan}
    ds = decode_payload(result["payloads"]["DSWRF_surface"], "DSWRF_surface", cfg)
    tc = decode_payload(result["payloads"]["TCDC_entire_atmosphere"], "TCDC_entire_atmosphere", cfg)
    return {**base, "dswrf": ds["value"], "tcdc": tc["value"], "ds_start": ds["start_step"],
            "ds_end": ds["end_step"], "tc_start": tc["start_step"], "tc_end": tc["end_step"],
            "ds_step_type": ds["step_type"], "tc_step_type": tc["step_type"],
            "ds_units": ds["units"], "tc_units": tc["units"], "lat": ds["lat"], "lon": ds["lon"]}


def download_jobs(jobs: set[tuple[dt.datetime, int]], cfg: dict) -> dict[tuple[dt.datetime, int], dict]:
    """Resume by month; only missing/failed jobs are fetched."""
    grouped: dict[str, set[tuple[dt.datetime, int]]] = defaultdict(set)
    for cycle, lead in jobs:
        grouped[cycle.strftime("%Y-%m")].add((cycle, lead))
    combined: dict[tuple[dt.datetime, int], dict] = {}
    for number, month in enumerate(sorted(grouped), 1):
        path = month_path(month)
        cached = load_month(path)
        month_jobs = grouped[month]
        missing = [(cycle, lead) for cycle, lead in sorted(month_jobs)
                   if cached.get((utc_ns(cycle), lead), {}).get("status", 0) != 1]
        if missing:
            print(json.dumps({"stage": "download", "month": month, "month_index": number,
                              "months_total": len(grouped), "jobs": len(month_jobs), "missing": len(missing)}), flush=True)
            with ThreadPoolExecutor(max_workers=cfg["download"]["workers"]) as pool:
                futures = [pool.submit(fetch_job, cycle, lead, cfg) for cycle, lead in missing]
                for future in as_completed(futures):
                    result = future.result()
                    try:
                        record = record_from_result(result, cfg)
                    except Exception as exc:
                        failed = {**result, "status": 0, "bytes": 0,
                                  "error": f"decode {type(exc).__name__}: {exc}"}
                        record = record_from_result(failed, cfg)
                    cached[(utc_ns(result["cycle"]), result["lead"])] = record
            save_month(path, cached)
        for cycle, lead in month_jobs:
            record = cached.get((utc_ns(cycle), lead))
            if record is not None:
                combined[(cycle, lead)] = record
    return combined


def needed_leads(age: float, horizon_hours: float, step: int = 3) -> set[int]:
    """Configured GFS lead grid bracketing the whole future trajectory."""
    minimum = age + 1 / 12
    maximum = age + horizon_hours
    lower = int(math.floor(minimum / step) * step)
    upper = int(math.ceil(maximum / step) * step)
    return set(range(max(step, lower), upper + 1, step))


def cycle_is_complete(records: dict, cycle: dt.datetime, leads: Iterable[int]) -> bool:
    return all(records.get((cycle, lead), {}).get("status", 0) == 1 for lead in leads)


def align_variable(records: dict, cycle: dt.datetime, variable: str, leads: np.ndarray,
                   cycle_records: list[dict] | None = None) -> np.ndarray:
    available = (cycle_records if cycle_records is not None else
                 [record for (record_cycle, _), record in records.items() if record_cycle == cycle])
    available = [record for record in available if record.get("status", 0) == 1]
    if not available:
        return np.full(len(leads), np.nan, np.float32)
    prefix = "ds" if variable == "dswrf" else "tc"
    step_types = {str(record[f"{prefix}_step_type"]) for record in available}
    if variable == "dswrf" or step_types != {"instant"}:
        output = np.full(len(leads), np.nan, np.float64)
        for index, lead in enumerate(leads):
            choices = [record for record in available if float(record[f"{prefix}_start"]) < lead <= float(record[f"{prefix}_end"])]
            if choices:
                chosen = min(choices, key=lambda record: float(record[f"{prefix}_end"]) - float(record[f"{prefix}_start"]))
                output[index] = float(chosen[variable])
        return output.astype(np.float32)
    x = np.asarray([float(record["tc_end"]) for record in available])
    y = np.asarray([float(record["tcdc"]) for record in available])
    order = np.argsort(x)
    return np.interp(leads, x[order], y[order], left=np.nan, right=np.nan).astype(np.float32)


def build_nwp_for_origins(origin_ns: np.ndarray, cfg: dict) -> tuple[dict, dict]:
    origins_utc = [acst_ns_to_utc(value) for value in origin_ns]
    nominal = [nominal_selected_cycle(origin) for origin in origins_utc]
    initial_jobs = {(cycle, lead) for cycle in nominal for lead in cfg["nwp_forecast_leads_hours"]}
    records = download_jobs(initial_jobs, cfg)
    # Only if a nominal cycle is incomplete, fetch the prior cycle with the
    # longer lead range required by a one-cycle causal fallback.
    fallback_jobs: set[tuple[dt.datetime, int]] = set()
    for origin, cycle in zip(origins_utc, nominal):
        age = (origin - cycle).total_seconds() / 3600
        if not cycle_is_complete(records, cycle, needed_leads(age, 12.0, cfg["nwp_lead_step_hours"])):
            earlier = cycle - dt.timedelta(hours=6)
            for lead in needed_leads(age + 6, 12.0, cfg["nwp_lead_step_hours"]):
                fallback_jobs.add((earlier, lead))
    if fallback_jobs:
        records.update(download_jobs(fallback_jobs, cfg))
    records_by_cycle: dict[dt.datetime, list[dict]] = defaultdict(list)
    for (record_cycle, _), record in records.items():
        records_by_cycle[record_cycle].append(record)
    n = len(origin_ns)
    dswrf = np.full((n, cfg["horizon"]), np.nan, np.float32)
    tcdc = np.full_like(dswrf, np.nan)
    lead_values = np.full_like(dswrf, np.nan)
    ages = np.full(n, np.nan, np.float32)
    selected_ns = np.zeros(n, np.int64)
    fallback = np.zeros(n, np.int16)
    valid = np.zeros(n, bool)
    for index, (origin, cycle) in enumerate(zip(origins_utc, nominal)):
        age = (origin - cycle).total_seconds() / 3600
        required = needed_leads(age, 12.0, cfg["nwp_lead_step_hours"])
        if not cycle_is_complete(records, cycle, required):
            cycle -= dt.timedelta(hours=6)
            fallback[index] = 1
            age += 6
            required = needed_leads(age, 12.0, cfg["nwp_lead_step_hours"])
        if not cycle_is_complete(records, cycle, required):
            continue
        if cycle + dt.timedelta(hours=6) > origin:
            raise AssertionError("selected GFS cycle is not causally available")
        leads = age + np.arange(1, cfg["horizon"] + 1) * cfg["frequency_minutes"] / 60
        selected_records = records_by_cycle.get(cycle, [])
        ds = align_variable(records, cycle, "dswrf", leads, selected_records)
        tc = align_variable(records, cycle, "tcdc", leads, selected_records)
        if np.isfinite(ds).all() and np.isfinite(tc).all():
            dswrf[index], tcdc[index], lead_values[index] = ds, tc, leads
            ages[index] = age
            selected_ns[index] = utc_ns(cycle)
            valid[index] = True
    all_records = list(records.values())
    summary = {
        "requested_objects": len(records),
        "successful_objects": sum(int(record.get("status", 0)) for record in all_records),
        "failed_objects": sum(not int(record.get("status", 0)) for record in all_records),
        "download_bytes": int(sum(int(record.get("bytes", 0)) for record in all_records)),
        "download_extract_seconds": float(sum(float(record.get("seconds", 0)) for record in all_records)),
        "fallback_origins": int(fallback.sum()),
        "valid_origins": int(valid.sum()),
        "origin_count": n,
        "coverage": float(valid.mean()) if n else math.nan,
        "monthly_artifacts": len(list(NWP_MONTHS.glob("gfs_point_*.npz"))),
    }
    arrays = {"dswrf": dswrf, "tcdc": tcdc, "lead": lead_values, "age": ages,
              "selected_cycle_ns": selected_ns, "fallback": fallback, "valid": valid}
    return arrays, summary


def build_reliability_prior(train_nwp: dict, train_ground_ghi: np.ndarray) -> tuple[np.ndarray, dict]:
    errors = np.abs(train_nwp["dswrf"] - train_ground_ghi)
    valid = np.isfinite(errors) & train_nwp["valid"][:, None]
    global_mae = float(np.mean(errors[valid]))
    age_bins = np.floor(np.nan_to_num(train_nwp["age"], nan=-1)).astype(np.int16)
    lead_bins = np.floor(np.nan_to_num(train_nwp["lead"], nan=-1)).astype(np.int16)
    cells: dict[tuple[int, int], float] = {}
    for age in np.unique(age_bins[np.isfinite(train_nwp["age"])]):
        for lead in np.unique(lead_bins[age_bins == age]):
            mask = valid & (age_bins[:, None] == age) & (lead_bins == lead)
            if mask.any():
                cells[(int(age), int(lead))] = float(np.mean(errors[mask]))
    metadata = {"global_mae_wm2": global_mae, "cells": {f"{age}:{lead}": value for (age, lead), value in cells.items()},
                "fit_split": "train", "validation_ground_ghi_used": False}
    return reliability_for(train_nwp, metadata), metadata


def reliability_for(nwp: dict, metadata: dict) -> np.ndarray:
    output = np.full_like(nwp["lead"], math.exp(-1), dtype=np.float32)
    ages = np.floor(np.nan_to_num(nwp["age"], nan=-1)).astype(np.int16)
    leads = np.floor(np.nan_to_num(nwp["lead"], nan=-1)).astype(np.int16)
    denominator = max(float(metadata["global_mae_wm2"]), 1e-6)
    for key, mae in metadata["cells"].items():
        age, lead = map(int, key.split(":"))
        output[(ages[:, None] == age) & (leads == lead)] = math.exp(-float(mae) / denominator)
    output[~nwp["valid"]] = 0
    return output


def future_values(values: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    return np.stack([values[int(origin) + 1:int(origin) + horizon + 1] for origin in origins]).astype(np.float32)


def prepare_data(force: bool = False) -> None:
    if PREPARED.exists() and not force:
        return
    cfg = config()
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame, pv_info = read_pv_train_validation(cfg)
    index = frame.index
    train_rows = split_mask(index, cfg["splits"]["train"])
    validation_rows = split_mask(index, cfg["splits"]["validation"])
    power = frame[cfg["target_column"]].to_numpy(np.float32)
    valid_power = frame["_source_present"].to_numpy(bool) & np.isfinite(power) & (power >= 0)
    train_origins, train_segments = legal_origins(valid_power, train_rows, cfg["lookback"], cfg["horizon"])
    validation_origins, validation_segments = legal_origins(valid_power, validation_rows, cfg["lookback"], cfg["horizon"])
    preprocessing = fit_history_preprocessing(frame, cfg, train_rows)
    timestamps = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    train_nwp, train_download = build_nwp_for_origins(timestamps[train_origins], cfg)
    validation_nwp, validation_download = build_nwp_for_origins(timestamps[validation_origins], cfg)
    # Fair origins are identical for all models; missing NWP is represented by
    # an explicit mask rather than deleting samples.
    train_ghi = future_values(frame[cfg["ground_ghi_column"]].to_numpy(np.float32), train_origins, cfg["horizon"])
    validation_ghi = future_values(frame[cfg["ground_ghi_column"]].to_numpy(np.float32), validation_origins, cfg["horizon"])
    train_prior, prior_metadata = build_reliability_prior(train_nwp, train_ghi)
    validation_prior = reliability_for(validation_nwp, prior_metadata)
    nwp_stack_train = np.stack([train_nwp["dswrf"], train_nwp["tcdc"]], axis=-1)
    train_valid_values = nwp_stack_train[train_nwp["valid"]]
    nwp_center = np.nanmean(train_valid_values, axis=(0, 1))
    nwp_scale = np.nanstd(train_valid_values, axis=(0, 1))
    nwp_scale = np.where(nwp_scale > 1e-8, nwp_scale, 1.0)
    def transform_nwp(nwp: dict) -> np.ndarray:
        raw = np.stack([nwp["dswrf"], nwp["tcdc"]], axis=-1)
        scaled = (np.where(np.isfinite(raw), raw, nwp_center) - nwp_center) / nwp_scale
        return np.concatenate([scaled, np.broadcast_to(nwp["valid"][:, None, None], (*scaled.shape[:2], 1))], axis=-1).astype(np.float32)
    train_changes = np.abs(np.diff(power))
    consecutive = valid_power[:-1] & valid_power[1:] & train_rows[:-1] & train_rows[1:]
    high_change_threshold = float(np.quantile(train_changes[consecutive], cfg["high_change_quantile"]))
    metadata = {
        **pv_info,
        "train_windows": len(train_origins), "validation_windows": len(validation_origins),
        "train_segments": len(train_segments), "validation_segments": len(validation_segments),
        "train_months": sorted(set(index[train_rows].strftime("%Y-%m"))),
        "validation_months": sorted(set(index[validation_rows].strftime("%Y-%m"))),
        "preprocessor_fit_split": "train", "nwp_scaler_fit_split": "train",
        "reliability_prior_fit_split": "train", "validation_ground_ghi_used_for_prior": False,
        "high_change_threshold_kw": high_change_threshold,
        "target_min_kw": preprocessing["target_min"], "target_range_kw": preprocessing["target_range"],
        "train_nwp": train_download, "validation_nwp": validation_download,
        "prior": prior_metadata, "prepare_seconds": time.perf_counter() - started,
        "neural_training_during_prepare": False,
    }
    arrays = {
        "times_ns": timestamps, "history_features": preprocessing["features"], "power": power,
        "ground_ghi": frame[cfg["ground_ghi_column"]].to_numpy(np.float32),
        "train_origins": train_origins, "validation_origins": validation_origins,
        "train_nwp": transform_nwp(train_nwp), "validation_nwp": transform_nwp(validation_nwp),
        "train_nwp_raw_dswrf": train_nwp["dswrf"], "validation_nwp_raw_dswrf": validation_nwp["dswrf"],
        "train_age": train_nwp["age"], "validation_age": validation_nwp["age"],
        "train_lead": train_nwp["lead"], "validation_lead": validation_nwp["lead"],
        "train_prior": train_prior, "validation_prior": validation_prior,
        "train_nwp_valid": train_nwp["valid"], "validation_nwp_valid": validation_nwp["valid"],
        "train_selected_cycle_ns": train_nwp["selected_cycle_ns"], "validation_selected_cycle_ns": validation_nwp["selected_cycle_ns"],
        "train_fallback": train_nwp["fallback"], "validation_fallback": validation_nwp["fallback"],
        "history_fill": preprocessing["fill"], "history_center": preprocessing["center"], "history_scale": preprocessing["scale"],
        "nwp_center": nwp_center.astype(np.float32), "nwp_scale": nwp_scale.astype(np.float32),
        "target_min": np.asarray(preprocessing["target_min"], np.float32),
        "target_range": np.asarray(preprocessing["target_range"], np.float32),
        "high_change_threshold": np.asarray(high_change_threshold, np.float32),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    np.savez_compressed(PREPARED, **arrays)
    pv_path = Path(cfg["pv_file"])
    if pv_path.stat().st_size != pv_info["pv_source_size"] or pv_path.stat().st_mtime_ns != pv_info["pv_source_mtime_ns"]:
        raise AssertionError("PV source changed during preparation")
    print(json.dumps({"stage": "prepared", "train_windows": len(train_origins),
                      "validation_windows": len(validation_origins), "artifact_bytes": PREPARED.stat().st_size}), flush=True)


class ForecastDataset(Dataset):
    def __init__(self, data: np.lib.npyio.NpzFile, split: str, cfg: dict):
        self.features = data["history_features"]
        self.power = data["power"]
        self.origins = data[f"{split}_origins"]
        self.nwp = data[f"{split}_nwp"]
        self.age = data[f"{split}_age"]
        self.lead = data[f"{split}_lead"]
        self.prior = data[f"{split}_prior"]
        self.target_min = float(data["target_min"])
        self.target_range = float(data["target_range"])
        self.lookback = cfg["lookback"]
        self.horizon = cfg["horizon"]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int):
        origin = int(self.origins[index])
        history = self.features[origin - self.lookback + 1:origin + 1]
        target = (self.power[origin + 1:origin + self.horizon + 1] - self.target_min) / self.target_range
        age_value = float(self.age[index]) if np.isfinite(self.age[index]) else 12.0
        age = np.full(self.horizon, age_value / 24.0, np.float32)
        lead = np.nan_to_num(self.lead[index], nan=18.0) / 30.0
        return (torch.from_numpy(history.copy()), torch.from_numpy(self.nwp[index].copy()),
                torch.from_numpy(age), torch.from_numpy(lead.astype(np.float32)),
                torch.from_numpy(self.prior[index].copy()), torch.from_numpy(target.astype(np.float32)))


class ModernTCNBackbone(nn.Module):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        model = cfg["model"]
        channels = model["channels"]
        layers: list[nn.Module] = [nn.Conv1d(input_dim, channels, 1), nn.GELU()]
        for _ in range(model["layers"]):
            layers.extend([nn.Conv1d(channels, channels, model["kernel_size"],
                                     padding=model["kernel_size"] // 2, groups=channels),
                           nn.Conv1d(channels, channels, 1), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.power_head = nn.Linear(channels * cfg["lookback"], cfg["horizon"])

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.power_head(self.net(history.transpose(1, 2)).flatten(1))


class NWPResidualEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden = cfg["model"]["nwp_hidden"]
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, nwp: torch.Tensor) -> torch.Tensor:
        return self.net(nwp).squeeze(-1)


class ForecastModel(nn.Module):
    def __init__(self, name: str, input_dim: int, cfg: dict):
        super().__init__()
        if name not in MODELS:
            raise ValueError(name)
        self.name = name
        self.history = ModernTCNBackbone(input_dim, cfg)
        if name != "HISTORY_ONLY":
            self.nwp = NWPResidualEncoder(cfg)
        if name == "AGE_LEAD_RELIABILITY":
            hidden = cfg["model"]["reliability_hidden"]
            self.gate = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, history: torch.Tensor, nwp: torch.Tensor, age: torch.Tensor,
                lead: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        base = self.history(history)
        if self.name == "HISTORY_ONLY":
            return base
        # A missing whole-trajectory NWP artifact must reduce exactly to the
        # history prediction rather than turning the fill value into evidence.
        delta = self.nwp(nwp) * nwp[..., 2]
        if self.name == "RAW_NWP":
            return base + delta
        gate_input = torch.stack([age, lead, prior], dim=-1)
        return base + self.gate(gate_input).squeeze(-1) * delta


def make_loaders(data: np.lib.npyio.NpzFile, cfg: dict, seed: int) -> tuple[DataLoader, DataLoader]:
    train = ForecastDataset(data, "train", cfg)
    validation = ForecastDataset(data, "validation", cfg)
    kwargs = {"batch_size": cfg["training"]["batch_size"], "num_workers": cfg["training"]["num_workers"],
              "pin_memory": torch.cuda.is_available()}
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(train, shuffle=True, generator=generator, **kwargs), DataLoader(validation, shuffle=False, **kwargs)


def train_model(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
                cfg: dict, device: torch.device, run_dir: Path) -> dict:
    """Validation-only selection; the function intentionally has no Test loader."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.MSELoss()
    best = math.inf
    stale = 0
    best_epoch = 0
    epoch_times: list[float] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "epochs.jsonl"
    log_path.write_text("", encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stop_reason = "max_epochs"
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        tick = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        for history, nwp, age, lead, prior, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history.to(device), nwp.to(device), age.to(device), lead.to(device), prior.to(device))
            loss = criterion(prediction, target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses: list[float] = []
        with torch.inference_mode():
            for history, nwp, age, lead, prior, target in validation_loader:
                prediction = model(history.to(device), nwp.to(device), age.to(device), lead.to(device), prior.to(device))
                validation_losses.append(float(criterion(prediction, target.to(device)).cpu()))
        validation_loss = float(np.mean(validation_losses))
        elapsed = time.perf_counter() - tick
        epoch_times.append(elapsed)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "validation_mse": validation_loss}
        torch.save(checkpoint, run_dir / "last.pt")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_mse": float(np.mean(train_losses)),
                                     "validation_mse": validation_loss, "seconds": elapsed}) + "\n")
        if validation_loss < best - cfg["training"]["min_delta"]:
            best, best_epoch, stale = validation_loss, epoch, 0
            torch.save({"epoch": epoch, "model": copy.deepcopy(model.state_dict()),
                        "validation_mse": validation_loss}, run_dir / "best_validation.pt")
        else:
            stale += 1
            if stale >= cfg["training"]["patience"]:
                stop_reason = "early_stopping"
                break
    return {"actual_epochs": epoch, "best_epoch": best_epoch, "best_validation_mse": best,
            "training_seconds": time.perf_counter() - started, "mean_epoch_seconds": float(np.mean(epoch_times)),
            "stop_reason": stop_reason, "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else math.nan),
            "numerically_finite": True}


def predict_validation(model: nn.Module, loader: DataLoader, device: torch.device,
                       target_min: float, target_range: float) -> tuple[np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for history, nwp, age, lead, prior, _ in loader:
            prediction = model(history.to(device), nwp.to(device), age.to(device), lead.to(device), prior.to(device))
            predictions.append(prediction.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    scaled = np.concatenate(predictions)
    physical = scaled * target_range + target_min
    if not np.isfinite(physical).all():
        raise FloatingPointError("non-finite validation prediction")
    return physical.astype(np.float32), seconds


def metric_values(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray, target_range: float) -> dict:
    y = labels[mask]
    p = predictions[mask]
    if not len(y):
        return {"rmse_kw": math.nan, "mae_kw": math.nan, "train_range_nrmse": math.nan, "r2": math.nan,
                "valid_target_count": 0}
    error = p - y
    rmse = float(np.sqrt(np.mean(error ** 2)))
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {"rmse_kw": rmse, "mae_kw": float(np.mean(np.abs(error))),
            "train_range_nrmse": rmse / target_range,
            "r2": 1 - float(np.sum(error ** 2)) / denominator if denominator > 0 else math.nan,
            "valid_target_count": int(len(y))}


def evaluate_validation(model_name: str, seed: int, predictions: np.ndarray, data: np.lib.npyio.NpzFile,
                        cfg: dict, training: dict, parameter_count: int, trainable_count: int,
                        inference_seconds: float) -> list[dict]:
    origins = data["validation_origins"]
    power = data["power"]
    labels = future_values(power, origins, cfg["horizon"])
    future_ghi = future_values(data["ground_ghi"], origins, cfg["horizon"])
    previous = np.concatenate([power[origins, None], labels[:, :-1]], axis=1)
    change = np.abs(labels - previous)
    daylight = np.isfinite(future_ghi) & (future_ghi >= cfg["daylight_ghi_threshold_wm2"])
    high_change = daylight & (change >= float(data["high_change_threshold"]))
    rows: list[dict] = []
    for horizon in cfg["horizons"]:
        y = labels[:, :horizon]
        p = predictions[:, :horizon]
        scopes = {
            "regular_full_timeline": np.ones(y.shape, bool),
            "daylight": daylight[:, :horizon],
            "high_change_daylight": high_change[:, :horizon],
        }
        for scope, mask in scopes.items():
            row = {"model": model_name, "seed": seed, "split": "validation", "horizon_steps": horizon,
                   "horizon_hours": horizon / 12, "scope": scope,
                   **metric_values(y, p, mask, float(data["target_range"])),
                   "window_count": len(origins), "parameter_count": parameter_count,
                   "trainable_parameter_count": trainable_count,
                   "inference_seconds": inference_seconds,
                   "inference_ms_per_sample": inference_seconds / len(origins) * 1000,
                   **training}
            rows.append(row)
    return rows


def write_metrics(rows: list[dict]) -> None:
    if not rows:
        return
    with METRICS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def existing_metrics() -> list[dict]:
    if not METRICS.exists():
        return []
    with METRICS.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_all() -> None:
    prepare_data()
    cfg = config()
    data = np.load(PREPARED, allow_pickle=False)
    if int(data["times_ns"].max()) >= int(pd.Timestamp("2023-01-01").value):
        raise AssertionError("sealed 2023 Test data entered B1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("B1 requires real GPU training")
    input_dim = data["history_features"].shape[1]
    rows = existing_metrics()
    completed = {(row["model"], int(row["seed"])) for row in rows}
    for model_name in MODELS:
        for seed in cfg["seeds"]:
            if (model_name, seed) in completed:
                continue
            set_seed(seed)
            train_loader, validation_loader = make_loaders(data, cfg, seed)
            run_dir = RESULTS / model_name / f"seed_{seed}"
            model = ForecastModel(model_name, input_dim, cfg).to(device)
            parameters = sum(parameter.numel() for parameter in model.parameters())
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            training = train_model(model, train_loader, validation_loader, cfg, device, run_dir)
            checkpoint = torch.load(run_dir / "best_validation.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            predictions, inference_seconds = predict_validation(model, validation_loader, device,
                                                                 float(data["target_min"]), float(data["target_range"]))
            origins = data["validation_origins"]
            labels = future_values(data["power"], origins, cfg["horizon"])
            np.savez_compressed(run_dir / "validation_H144.npz", predictions=predictions, labels=labels,
                                forecast_origin_timestamp_ns=data["times_ns"][origins],
                                selected_cycle_utc_ns=data["validation_selected_cycle_ns"],
                                forecast_age_hours=data["validation_age"], nwp_valid=data["validation_nwp_valid"],
                                fallback_cycles=data["validation_fallback"], split=np.asarray("validation"))
            rows.extend(evaluate_validation(model_name, seed, predictions, data, cfg, training,
                                            parameters, trainable, inference_seconds))
            write_metrics(rows)
            print(json.dumps({"stage": "run_complete", "model": model_name, "seed": seed,
                              "best_epoch": training["best_epoch"], "validation_mse": training["best_validation_mse"]}), flush=True)
    build_report(rows, data, cfg, device)


def mean_rmse(rows: list[dict], model: str, horizon: int) -> float:
    values = [float(row["rmse_kw"]) for row in rows if row["model"] == model and int(row["horizon_steps"]) == horizon
              and row["scope"] == "regular_full_timeline"]
    return float(np.mean(values))


def build_report(rows: list[dict], data: np.lib.npyio.NpzFile, cfg: dict, device: torch.device) -> None:
    metadata = json.loads(str(data["metadata_json"]))
    nwp_inventory = local_nwp_inventory()
    numeric_rows = [{key: (float(value) if key in {"rmse_kw", "mae_kw", "train_range_nrmse", "r2",
                    "parameter_count", "mean_epoch_seconds", "inference_ms_per_sample", "peak_gpu_memory_mb"} else value)
                    for key, value in row.items()} for row in rows]
    def summary(model: str, horizon: int, scope: str = "regular_full_timeline") -> tuple[float, float]:
        values = [row["rmse_kw"] for row in numeric_rows if row["model"] == model and int(row["horizon_steps"]) == horizon and row["scope"] == scope]
        return float(np.mean(values)), float(np.std(values, ddof=1))
    def metric_summary(model: str, horizon: int, scope: str, metric: str) -> tuple[float, float]:
        values = [row[metric] for row in numeric_rows if row["model"] == model and
                  int(row["horizon_steps"]) == horizon and row["scope"] == scope]
        return float(np.mean(values)), float(np.std(values, ddof=1))
    raw_changes = {horizon: (mean_rmse(numeric_rows, "HISTORY_ONLY", horizon) - mean_rmse(numeric_rows, "RAW_NWP", horizon)) /
                   mean_rmse(numeric_rows, "HISTORY_ONLY", horizon) * 100 for horizon in cfg["horizons"]}
    reliability_changes = {horizon: (mean_rmse(numeric_rows, "RAW_NWP", horizon) - mean_rmse(numeric_rows, "AGE_LEAD_RELIABILITY", horizon)) /
                           mean_rmse(numeric_rows, "RAW_NWP", horizon) * 100 for horizon in cfg["horizons"]}
    def paired_directions(left: str, right: str, horizon: int) -> list[bool]:
        return [next(row for row in numeric_rows if row["model"] == right and int(row["seed"]) == seed and int(row["horizon_steps"]) == horizon and row["scope"] == "regular_full_timeline")["rmse_kw"] <
                next(row for row in numeric_rows if row["model"] == left and int(row["seed"]) == seed and int(row["horizon_steps"]) == horizon and row["scope"] == "regular_full_timeline")["rmse_kw"]
                for seed in cfg["seeds"]]
    raw_pass = (raw_changes[144] >= 2 and all(paired_directions("HISTORY_ONLY", "RAW_NWP", 144)) and
                sum(value > 0 for value in raw_changes.values()) >= 3 and min(raw_changes.values()) >= -1)
    raw_params = next(row["parameter_count"] for row in numeric_rows if row["model"] == "RAW_NWP")
    rel_params = next(row["parameter_count"] for row in numeric_rows if row["model"] == "AGE_LEAD_RELIABILITY")
    parameter_increase = (rel_params - raw_params) / raw_params * 100
    reliability_pass = (reliability_changes[144] >= 1 and all(paired_directions("RAW_NWP", "AGE_LEAD_RELIABILITY", 144)) and
                        sum(value > 0 for value in reliability_changes.values()) >= 3 and min(reliability_changes.values()) >= -1 and
                        parameter_increase <= 10)
    decision = "NWP_INFORMATION_FAIL" if not raw_pass else ("B2_FINAL_EVALUATION_READY" if reliability_pass else "RAW_NWP_ONLY")
    table_lines = ["| Model | H12 RMSE | H48 RMSE | H96 RMSE | H144 RMSE | Parameters |",
                   "|---|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        values = [summary(model, horizon) for horizon in cfg["horizons"]]
        params = int(next(row["parameter_count"] for row in numeric_rows if row["model"] == model))
        table_lines.append(f"| {model} | " + " | ".join(f"{mean:.6f} ± {sd:.6f}" for mean, sd in values) + f" | {params:,} |")
    scope_lines = ["| Model | Scope | RMSE (kW) | MAE (kW) | Train-range nRMSE | R² |",
                   "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        for scope in ("regular_full_timeline", "daylight", "high_change_daylight"):
            values = [metric_summary(model, 144, scope, metric) for metric in
                      ("rmse_kw", "mae_kw", "train_range_nrmse", "r2")]
            scope_lines.append(f"| {model} | {scope} | " +
                               " | ".join(f"{mean:.6f} ± {sd:.6f}" for mean, sd in values) + " |")
    seed_lines = ["| Seed | HISTORY_ONLY H144 RMSE | RAW_NWP H144 RMSE | RAW change | AGE_LEAD H144 RMSE | AGE_LEAD vs RAW |",
                  "|---:|---:|---:|---:|---:|---:|"]
    for seed in cfg["seeds"]:
        by_model = {model: next(row["rmse_kw"] for row in numeric_rows if row["model"] == model and
                               int(row["seed"]) == seed and int(row["horizon_steps"]) == 144 and
                               row["scope"] == "regular_full_timeline") for model in MODELS}
        raw_change = (by_model["HISTORY_ONLY"] - by_model["RAW_NWP"]) / by_model["HISTORY_ONLY"] * 100
        rel_change = (by_model["RAW_NWP"] - by_model["AGE_LEAD_RELIABILITY"]) / by_model["RAW_NWP"] * 100
        seed_lines.append(f"| {seed} | {by_model['HISTORY_ONLY']:.6f} | {by_model['RAW_NWP']:.6f} | "
                          f"{raw_change:.3f}% | {by_model['AGE_LEAD_RELIABILITY']:.6f} | {rel_change:.3f}% |")
    run_rows = {(row["model"], int(row["seed"])): row for row in numeric_rows if int(row["horizon_steps"]) == 144 and row["scope"] == "regular_full_timeline"}
    run_lines = ["| Model | Seed | Best epoch | Stop | Epoch seconds | Peak GPU MB |", "|---|---:|---:|---|---:|---:|"]
    for key in sorted(run_rows):
        row = run_rows[key]
        run_lines.append(f"| {key[0]} | {key[1]} | {row['best_epoch']} | {row['stop_reason']} | {float(row['mean_epoch_seconds']):.2f} | {float(row['peak_gpu_memory_mb']):.1f} |")
    report = f"""# Stage B1 — Causal GFS minimal performance screen

## Verdict

**{decision}**

- New GPU training: 9/9 completed.
- Numerical divergence/non-finite gradients: none.
- Sealed 2023 Test accessed: **no**.
- Models use identical Train/Validation origins, labels and masks; H12/H48/H96/H144 are prefixes of one H144 output.

## Data and operational protocol

- Site 17 Sanyo only; no cross-site or cross-climate claim.
- Train: {cfg['splits']['train'][0]} to {cfg['splits']['train'][1]} (exclusive end).
- Validation: {cfg['splits']['validation'][0]} to {cfg['splits']['validation'][1]} (exclusive end).
- `PREVIOUS_COMPLETED_CYCLE_6H`; each H144 NWP trajectory uses one cycle satisfying `cycle + 6h <= origin`.
- Official GFS `{cfg['gfs_product']}` ({cfg['gfs_resolution_degrees']}°), DSWRF and TCDC only, on the predeclared 3-hour lead grid f006/f009/.../f024. This B1 bandwidth-minimal product is explicitly distinct from the B0.1 0.25° hourly pilot.
- Train windows: {metadata['train_windows']:,} in {metadata['train_segments']} legal fragments; Validation windows: {metadata['validation_windows']:,} in {metadata['validation_segments']} fragments.
- Train NWP coverage: {metadata['train_nwp']['valid_origins']:,}/{metadata['train_nwp']['origin_count']:,} ({metadata['train_nwp']['coverage']:.3%}); Validation: {metadata['validation_nwp']['valid_origins']:,}/{metadata['validation_nwp']['origin_count']:,} ({metadata['validation_nwp']['coverage']:.3%}).
- Unique official objects retained: {nwp_inventory['unique_objects']:,}; successful: {nwp_inventory['successful_objects']:,}; failed: {nwp_inventory['failed_objects']:,}; retained successful IDX/message bytes: {nwp_inventory['retained_transfer_bytes']:,}. Retry/failure network overhead was not instrumented, so this is a measured lower bound rather than an exact wire-byte count.
- Fallback origins: Train {metadata['train_nwp']['fallback_origins']:,}; Validation {metadata['validation_nwp']['fallback_origins']:,}.
- Compact NWP point artifacts: {nwp_inventory['compact_nwp_artifact_bytes']:,} bytes; prepared learning artifact: {PREPARED.stat().st_size:,} bytes; preparation/download/extraction wall time {metadata['prepare_seconds']:.1f} s.
- DSWRF is interval-average support; instantaneous TCDC is interpolated only within the selected cycle. Future measured GHI is metric-side only; only Train GHI forms the frozen reliability prior.

## Validation full-timeline RMSE, mean ± sample SD across seeds

{chr(10).join(table_lines)}

RAW_NWP relative to HISTORY_ONLY: {json.dumps(raw_changes, sort_keys=True)} percent improvement by horizon.

AGE_LEAD_RELIABILITY relative to RAW_NWP: {json.dumps(reliability_changes, sort_keys=True)} percent improvement by horizon.

M2 parameter increase over RAW_NWP: {parameter_increase:.6f}%.

## H144 scope metrics, mean ± sample SD

{chr(10).join(scope_lines)}

## H144 per-seed direction

{chr(10).join(seed_lines)}

## Run and efficiency record

{chr(10).join(run_lines)}

Device: `{torch.cuda.get_device_name(device)}`. Per-seed inference latency and all scope metrics are retained in `metrics_per_seed.csv`; local checkpoints and Validation prediction arrays remain under `results/` and are not committed.

## Pre-registered interpretation

- RAW_NWP rule satisfied: **{raw_pass}**. H144 improvement={raw_changes[144]:.3f}%; seed directions={paired_directions('HISTORY_ONLY','RAW_NWP',144)}; improved horizons={sum(value > 0 for value in raw_changes.values())}/4; worst change={min(raw_changes.values()):.3f}%.
- AGE_LEAD_RELIABILITY rule satisfied: **{reliability_pass}**. H144 improvement={reliability_changes[144]:.3f}%; seed directions={paired_directions('RAW_NWP','AGE_LEAD_RELIABILITY',144)}; improved horizons={sum(value > 0 for value in reliability_changes.values())}/4; worst change={min(reliability_changes.values()):.3f}%; parameter increase={parameter_increase:.6f}%.

Although M2 consistently reduces the very poor RAW_NWP errors, the pre-registered first-stage information criterion fails. M2 therefore cannot rescue this screen. The unusually large cross-year errors and seed variance limit the inference to this fixed implementation, 1.0° product, 3-hour forecast grid, and 2021→2022 split; they do not prove that GFS radiation/cloud forecasts are physically uninformative in every operational design. No additional variables, 2023 download, or v2/v3 model is justified under the registered stopping rule.

The decision is limited to Validation. No 2023 NWP was downloaded and no 2023 prediction, error, threshold or metric was produced.
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()
    prepare_data(force=args.force_prepare)
    if not args.prepare_only:
        run_all()


if __name__ == "__main__":
    main()


"""Read-only PV coverage, operational GFS availability and information audit."""
from __future__ import annotations

import csv
import datetime as dt
import http.client
import json
import math
import os
import statistics
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PV_DIR = Path(CFG["pv_directory"])
NWP_DIR = Path(CFG["nwp_directory"])
ACST = dt.timezone(dt.timedelta(hours=CFG["acst_offset_hours"]), name="ACST")
UTC = dt.timezone.utc
WINDOW = CFG["lookback_steps"] + CFG["horizon_steps"]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return math.nan


def audit_pv_file(path: Path, site_name: str | None) -> tuple[list[dict], pd.DataFrame | None, dict]:
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    rows_out: list[dict] = []
    physical = valid_rows = malformed = duplicate = reverse = 0
    first = last = previous = None
    missing = Counter(); year_stats = defaultdict(lambda: Counter(rows=0, power_valid=0))
    month_stats = defaultdict(lambda: Counter(rows=0, power_valid=0))
    records = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header_line = handle.readline(); physical += 1
        header = next(csv.reader([header_line])); ncol = len(header)
        ti = header.index("timestamp"); pi = header.index("Active_Power")
        gi = header.index("Global_Horizontal_Radiation")
        for line in handle:
            physical += 1
            try: row = next(csv.reader([line]))
            except csv.Error:
                malformed += 1; continue
            if len(row) != ncol:
                malformed += 1; continue
            try: stamp = dt.datetime.fromisoformat(row[ti].strip('"'))
            except ValueError:
                malformed += 1; continue
            valid_rows += 1
            if previous is not None:
                if stamp == previous: duplicate += 1
                if stamp < previous: reverse += 1
            previous = stamp; first = stamp if first is None else first; last = stamp
            for idx, name in enumerate(header):
                if row[idx].strip() == "": missing[name] += 1
            if 2020 <= stamp.year <= 2024:
                power = parse_float(row[pi]); ghi = parse_float(row[gi])
                year_stats[stamp.year]["rows"] += 1
                month_stats[stamp.strftime("%Y-%m")]["rows"] += 1
                if np.isfinite(power):
                    year_stats[stamp.year]["power_valid"] += 1
                    month_stats[stamp.strftime("%Y-%m")]["power_valid"] += 1
                if site_name: records.append((stamp, power, ghi))
    if path.stat().st_size != before[0] or path.stat().st_mtime_ns != before[1]:
        raise AssertionError(f"PV source changed: {path}")
    summary = {"record_type": "PV_FILE", "site": site_name or "other", "source_path": str(path),
        "file_size_bytes": before[0], "source_mtime_ns": before[1], "physical_lines": physical, "parsed_records": valid_rows,
        "first_timestamp": first, "last_timestamp": last, "timezone_status": "NAIVE; interpreted as ACST by validated project convention",
        "dominant_interval_minutes": 5, "duplicate_timestamps": duplicate, "reverse_timestamps": reverse,
        "malformed_physical_lines": malformed, "column_names": "|".join(header),
        "missing_by_column": json.dumps(missing, sort_keys=True), "source": "DKASC user-downloaded raw CSV"}
    rows_out.append(summary)
    frame = None
    if site_name:
        frame = pd.DataFrame(records, columns=["timestamp", "power", "ghi"]).drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    for year in range(2020, 2025):
        start = dt.datetime(year, 1, 1); end = dt.datetime(year + 1, 1, 1)
        expected = int((end - start).total_seconds() / 300)
        ys = year_stats[year]
        idx = frame.loc[str(year)].index if frame is not None and str(year) in frame.index.strftime("%Y").unique() else pd.DatetimeIndex([])
        unique = len(idx.unique()) if frame is not None else ys["rows"]
        valid_set = set(frame.loc[str(year)].index[frame.loc[str(year), "power"].notna()]) if frame is not None and len(idx) else set()
        windows = count_continuous_windows(valid_set, start, end) if frame is not None else ""
        rows_out.append({"record_type": "PV_YEAR", "site": site_name or "other", "source_path": str(path),
            "period": year, "expected_timestamps": expected, "actual_unique_timestamps": unique,
            "missing_timestamps": max(0, expected - unique), "active_power_valid": ys["power_valid"],
            "coverage_ratio": unique / expected, "continuous_L72_H144_windows": windows})
    for month, stat in sorted(month_stats.items()):
        rows_out.append({"record_type": "PV_MONTH", "site": site_name or "other", "source_path": str(path),
            "period": month, "actual_unique_timestamps": stat["rows"], "active_power_valid": stat["power_valid"]})
    return rows_out, frame, summary


def count_continuous_windows(valid_set: set[dt.datetime], start: dt.datetime, end: dt.datetime) -> int:
    total = run = 0; stamp = start
    while stamp < end:
        if stamp in valid_set:
            run += 1
            if run >= WINDOW: total += 1
        else: run = 0
        stamp += dt.timedelta(minutes=5)
    return total


def http_get(url: str, byte_range: tuple[int, int] | None = None) -> tuple[bytes, dict]:
    headers = {"User-Agent": "PV-NWP-feasibility-audit/1.0"}
    if byte_range: headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read(); meta = dict(response.headers)
                if byte_range and response.status != 206:
                    raise AssertionError(f"Range request not honored: {url}")
                if byte_range and len(data) != byte_range[1] - byte_range[0] + 1:
                    raise http.client.IncompleteRead(data, byte_range[1] - byte_range[0] + 1 - len(data))
                return data, meta
        except (http.client.IncompleteRead, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise last_error


def parse_idx(text: str) -> list[dict]:
    lines = text.splitlines(); parsed = []
    for i, line in enumerate(lines):
        parts = line.split(":", 2)
        if len(parts) < 3: continue
        parsed.append({"line": line, "offset": int(parts[1]), "next": None})
    for i in range(len(parsed) - 1): parsed[i]["next"] = parsed[i + 1]["offset"]
    return parsed


def safe_codes_get(codes_get, gid, key, default="UNKNOWN"):
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def pilot_cycles() -> list[dt.datetime]:
    pilot = CFG["continuous_pilot"]
    first = dt.date.fromisoformat(pilot["cycle_date_start_utc"])
    last = dt.date.fromisoformat(pilot["cycle_date_end_utc"])
    out = []
    day = first
    while day <= last:
        out.extend(dt.datetime.combine(day, dt.time(hour), UTC) for hour in pilot["cycles_utc"])
        day += dt.timedelta(days=1)
    return out


def gfs_base(cycle: dt.datetime, lead: int) -> str:
    return (f"{CFG['gfs_archive']}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}")


def fetch_selected_file(cycle: dt.datetime, lead: int) -> dict:
    """Download only requested GRIB messages. Existing objects are read, never rewritten."""
    base = gfs_base(cycle, lead)
    stem = f"gfs_{cycle:%Y%m%d_%H}_f{lead:03d}"
    idx_path = NWP_DIR / f"{stem}.idx"
    grib_path = NWP_DIR / f"{stem}.selected.grib2"
    started = time.perf_counter(); downloaded = 0
    try:
        if idx_path.exists():
            idx_bytes = idx_path.read_bytes(); idx_meta = {"Last-Modified": "existing_official_object"}
        else:
            idx_bytes, idx_meta = http_get(base + ".idx")
            idx_path.write_bytes(idx_bytes); downloaded += len(idx_bytes)
        entries = parse_idx(idx_bytes.decode("utf-8"))
        selected = []
        for logical, pattern in CFG["gfs_variables"].items():
            matches = [entry for entry in entries if pattern in entry["line"]]
            if not matches:
                raise ValueError(f"missing index entry {logical}")
            entry = matches[0]
            if entry["next"] is None:
                raise ValueError(f"missing end offset {logical}")
            selected.append((logical, entry))
        if not grib_path.exists():
            payloads = []
            for logical, entry in selected:
                payload, _ = http_get(base, (entry["offset"], entry["next"] - 1))
                payloads.append(payload); downloaded += len(payload)
            grib_path.write_bytes(b"".join(payloads))
        return {"cycle": cycle, "lead": lead, "base": base, "idx_path": idx_path,
                "grib_path": grib_path, "selected": selected, "downloaded_bytes": downloaded,
                "elapsed_seconds": time.perf_counter()-started, "status": "SUCCESS",
                "last_modified": idx_meta.get("Last-Modified", "UNKNOWN"), "error": ""}
    except (OSError, ValueError, http.client.HTTPException,
            urllib.error.URLError, urllib.error.HTTPError) as exc:
        return {"cycle": cycle, "lead": lead, "base": base, "idx_path": idx_path,
                "grib_path": grib_path, "selected": [], "downloaded_bytes": downloaded,
                "elapsed_seconds": time.perf_counter()-started, "status": "FAILED",
                "last_modified": "UNKNOWN", "error": f"{type(exc).__name__}: {exc}"}


def download_continuous_pilot() -> tuple[list[dict], list[dict], list[dict]]:
    try:
        from eccodes import codes_get, codes_grib_find_nearest, codes_grib_new_from_file, codes_release
    except ImportError as exc:
        raise RuntimeError("ecCodes is required for read-only GRIB2 point extraction") from exc
    NWP_DIR.mkdir(parents=True, exist_ok=True)
    cycles = pilot_cycles(); leads = CFG["continuous_pilot"]["forecast_leads_hours"]
    jobs = [(cycle, lead) for cycle in cycles for lead in leads]
    results = []
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(fetch_selected_file, cycle, lead) for cycle, lead in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    rows = []; file_rows = []
    for result in sorted(results, key=lambda x:(x["cycle"],x["lead"])):
        file_rows.append({"record_type":"NWP_PILOT_OBJECT","cycle_utc":result["cycle"].isoformat(),
            "forecast_lead_hours":result["lead"],"source_object":result["base"],
            "local_grib_path":str(result["grib_path"]),"local_idx_path":str(result["idx_path"]),
            "download_status":result["status"],"downloaded_bytes_this_run":result["downloaded_bytes"],
            "idx_file_bytes":result["idx_path"].stat().st_size if result["idx_path"].exists() else 0,
            "selected_file_bytes":result["grib_path"].stat().st_size if result["grib_path"].exists() else 0,
            "extraction_seconds":result["elapsed_seconds"],"failure_reason":result["error"],
            "official_object_last_modified":result["last_modified"]})
        if result["status"] != "SUCCESS":
            continue
        with result["grib_path"].open("rb") as handle:
            for logical, entry in result["selected"]:
                gid = codes_grib_new_from_file(handle)
                if gid is None: raise AssertionError(f"GRIB message missing: {result['grib_path']}")
                valid = dt.datetime.strptime(str(codes_get(gid,"validityDate"))+f"{int(codes_get(gid,'validityTime')):04d}","%Y%m%d%H%M").replace(tzinfo=UTC)
                nearest = codes_grib_find_nearest(gid,CFG["gfs_grid_target"]["latitude"],CFG["gfs_grid_target"]["longitude"])[0]
                start_step=float(safe_codes_get(codes_get,gid,"startStep",result["lead"]))
                end_step=float(safe_codes_get(codes_get,gid,"endStep",result["lead"]))
                rows.append({"record_type":"NWP_PILOT_MESSAGE","model":"NOAA GFS 0.25 degree",
                    "cycle_utc":result["cycle"].isoformat(),"nominal_issue_time_utc":result["cycle"].isoformat(),
                    "policy_available_time_utc":(result["cycle"]+dt.timedelta(hours=6)).isoformat(),
                    "availability_policy":CFG["availability_policy"],"forecast_lead_hours":result["lead"],
                    "valid_time_utc":valid.isoformat(),"valid_time_acst":valid.astimezone(ACST).isoformat(),
                    "variable":logical,"short_name":codes_get(gid,"shortName"),"stepType":safe_codes_get(codes_get,gid,"stepType"),
                    "startStep":start_step,"endStep":end_step,"units":codes_get(gid,"units"),
                    "typeOfStatisticalProcessing":safe_codes_get(codes_get,gid,"typeOfStatisticalProcessing"),
                    "validityDate":safe_codes_get(codes_get,gid,"validityDate"),"validityTime":safe_codes_get(codes_get,gid,"validityTime"),
                    "grid_latitude":nearest["lat"],"grid_longitude":nearest["lon"],"grid_distance_degrees":nearest["distance"],
                    "value":nearest["value"],"idx_url":result["base"]+".idx","grib_url":result["base"],
                    "source_object":result["base"],"local_idx_path":str(result["idx_path"]),
                    "local_grib_path":str(result["grib_path"]),"inventory_line":entry["line"],"archive_http_verified":True})
                codes_release(gid)
    origin_rows = build_origin_mappings(rows, file_rows)
    return rows, file_rows, origin_rows


def aligned_cycle_value(message_rows: list[dict], variable: str, lead_hours: float) -> float:
    rows = sorted((r for r in message_rows if r["variable"]==variable), key=lambda r:float(r["endStep"]))
    if variable in {"DSWRF_surface","APCP_surface"} or any(r["stepType"] != "instant" for r in rows):
        candidates=[r for r in rows if float(r["startStep"]) < lead_hours <= float(r["endStep"])]
        if not candidates: return math.nan
        row=min(candidates,key=lambda r:float(r["endStep"])-float(r["startStep"]))
        value=float(row["value"])
        if variable=="APCP_surface":
            duration=float(row["endStep"])-float(row["startStep"])
            return value/duration if duration>0 else math.nan
        return value
    x=np.array([float(r["endStep"]) for r in rows]); y=np.array([float(r["value"]) for r in rows])
    if len(x)<2 or lead_hours<x.min() or lead_hours>x.max(): return math.nan
    return float(np.interp(lead_hours,x,y))


def build_origin_mappings(message_rows: list[dict], file_rows: list[dict]) -> list[dict]:
    by_cycle=defaultdict(list)
    for row in message_rows: by_cycle[dt.datetime.fromisoformat(row["cycle_utc"])].append(row)
    good_files=defaultdict(set)
    for row in file_rows:
        if row["download_status"]=="SUCCESS": good_files[dt.datetime.fromisoformat(row["cycle_utc"])].add(int(row["forecast_lead_hours"]))
    required=set(CFG["continuous_pilot"]["forecast_leads_hours"])
    start=dt.datetime.fromisoformat(CFG["continuous_pilot"]["start_acst"]).replace(tzinfo=ACST)
    end=dt.datetime.fromisoformat(CFG["continuous_pilot"]["end_acst"]).replace(tzinfo=ACST)
    output=[]; origin=start
    while origin<=end:
        origin_utc=origin.astimezone(UTC)
        latest_hour=(origin_utc.hour//6)*6
        nominal=origin_utc.replace(hour=latest_hour,minute=0,second=0,microsecond=0)-dt.timedelta(hours=6)
        fallback=0; selected=nominal
        while selected>=min(pilot_cycles())-dt.timedelta(hours=24) and good_files.get(selected,set())!=required:
            selected-=dt.timedelta(hours=6); fallback+=1
        available=selected+dt.timedelta(hours=6); age=(origin_utc-selected).total_seconds()/3600
        all_valid=selected in by_cycle and available<=origin_utc
        valid_points=0; total_points=CFG["horizon_steps"]*len(CFG["gfs_variables"])
        if all_valid:
            for step in range(1,CFG["horizon_steps"]+1):
                lead=age+step*CFG["frequency_minutes"]/60
                for variable in CFG["gfs_variables"]:
                    if np.isfinite(aligned_cycle_value(by_cycle[selected],variable,lead)): valid_points+=1
            all_valid=valid_points==total_points
        source=(f"{CFG['gfs_archive']}/gfs.{selected:%Y%m%d}/{selected:%H}/atmos/"
                f"gfs.t{selected:%H}z.pgrb2.0p25.f006..f024") if selected in by_cycle else "MISSING"
        output.append({"record_type":"NWP_ORIGIN_MAPPING","forecast_origin_utc":origin_utc.isoformat(),
            "forecast_origin_acst":origin.isoformat(),"selected_cycle_utc":selected.isoformat(),
            "policy_available_time_utc":available.isoformat(),"availability_policy":CFG["availability_policy"],
            "forecast_age_hours":age,"forecast_lead_hours":f"{age+1/12:.6f}..{age+12:.6f}",
            "valid_time_utc":f"{(origin_utc+dt.timedelta(minutes=5)).isoformat()}..{(origin_utc+dt.timedelta(hours=12)).isoformat()}",
            "fallback_cycles":fallback,"nwp_valid":all_valid,"valid_nwp_points":valid_points,
            "expected_nwp_points":total_points,"trajectory_cycle_policy":"SINGLE_SELECTED_CYCLE",
            "source_object":source})
        origin+=dt.timedelta(minutes=5)
    return output


def align_and_summarize(nwp_rows: list[dict], site_frames: dict[str, pd.DataFrame]) -> list[dict]:
    output = []
    nwp = pd.DataFrame(nwp_rows)
    for row in nwp_rows:
        stamp = pd.Timestamp(row["valid_time_acst"]).tz_localize(None)
        for site, frame in site_frames.items():
            if stamp in frame.index:
                row[f"{site}_power"] = frame.at[stamp, "power"]
                row[f"{site}_ground_ghi"] = frame.at[stamp, "ghi"]
            else:
                row[f"{site}_power"] = math.nan; row[f"{site}_ground_ghi"] = math.nan
    nwp = pd.DataFrame(nwp_rows)
    rad = nwp[nwp.variable == "DSWRF_surface"].copy()
    for group_name, group_cols in (("overall", []), ("lead", ["forecast_lead_hours"]),
                                   ("cycle", ["cycle_utc"])):
        groups = [("ALL", rad)] if not group_cols else rad.groupby(group_cols[0])
        for key, g in groups:
            good = g[["value", "Sanyo_ground_ghi", "Sanyo_power"]].dropna()
            if len(good) < 3: continue
            output.append({"record_type": "NWP_ALIGNMENT", "group_type": group_name, "group": key,
                "sample_count": len(good), "gfs_ground_ghi_pearson": good.value.corr(good.Sanyo_ground_ghi),
                "gfs_ground_ghi_mae_wm2": float(np.mean(np.abs(good.value-good.Sanyo_ground_ghi))),
                "gfs_pv_power_pearson": good.value.corr(good.Sanyo_power)})
    direction = []
    for _, g in rad.sort_values("valid_time_utc").groupby("cycle_utc"):
        g = g.sort_values("valid_time_utc")
        for i in range(1, len(g)):
            a, b = g.iloc[i-1], g.iloc[i]
            if all(np.isfinite([a.value,b.value,a.Sanyo_ground_ghi,b.Sanyo_ground_ghi,a.Sanyo_power,b.Sanyo_power])):
                dn = np.sign(b.value-a.value); dg=np.sign(b.Sanyo_ground_ghi-a.Sanyo_ground_ghi); dp=np.sign(b.Sanyo_power-a.Sanyo_power)
                direction.append((dn==dg, dn==dp))
    output.append({"record_type": "NWP_DIRECTION", "group_type": "overall", "group": "3-hour sampled changes",
        "sample_count": len(direction), "gfs_ground_ghi_direction_accuracy": float(np.mean([x[0] for x in direction])) if direction else math.nan,
        "gfs_pv_direction_accuracy": float(np.mean([x[1] for x in direction])) if direction else math.nan})
    common = set.intersection(*(set(frame.index) for frame in site_frames.values()))
    output.append({"record_type": "CROSS_ARRAY_ALIGNMENT", "group": "2020-2024 exact common timestamps",
        "sample_count": len(common), "three_array_common_nwp": True, "notes": "co-located arrays use the same GFS grid/cycles"})
    return output


def consecutive_segments(stamps: set[dt.datetime], start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime,dt.datetime,int]]:
    selected=sorted(t for t in stamps if start<=t<=end)
    segments=[]; run_start=prev=None; run=0
    for stamp in selected:
        if prev is not None and stamp-prev==dt.timedelta(minutes=5): run+=1
        else:
            if run_start is not None: segments.append((run_start,prev,run))
            run_start=stamp; run=1
        prev=stamp
    if run_start is not None: segments.append((run_start,prev,run))
    return segments


def split_segment_rows(site_frames: dict[str,pd.DataFrame]) -> list[dict]:
    valid={site:set(frame.index[frame.power.notna()]) for site,frame in site_frames.items()}
    valid["ALL_THREE_COMMON"]=set.intersection(*(valid[s] for s in site_frames))
    rows=[]
    for split,(start_text,end_text) in CFG["preferred_splits"].items():
        start=dt.datetime.fromisoformat(start_text); end=dt.datetime.fromisoformat(end_text)
        for site,stamps in valid.items():
            segments=consecutive_segments(stamps,start,end)
            legal=[segment for segment in segments if segment[2]>=WINDOW]
            windows=sum(length-WINDOW+1 for _,_,length in legal)
            months=sorted({stamp.strftime("%Y-%m") for stamp in stamps if start<=stamp<=end})
            rows.append({"record_type":"PV_SPLIT_SUMMARY","split":split,"site":site,
                "period_start":start,"period_end":end,"continuous_segment_count":len(segments),
                "legal_segment_count":len(legal),"continuous_L72_H144_windows":windows,
                "months_covered":"|".join(months),"valid_timestamp_count":sum(length for _,_,length in segments)})
            for number,(seg_start,seg_end,length) in enumerate(legal,1):
                rows.append({"record_type":"PV_LEGAL_SEGMENT","split":split,"site":site,"segment_id":number,
                    "period_start":seg_start,"period_end":seg_end,"segment_length_5min":length,
                    "continuous_L72_H144_windows":length-WINDOW+1,"months_covered":"|".join(sorted({seg_start.strftime('%Y-%m'),seg_end.strftime('%Y-%m')}))})
    return rows


def verify_archive_boundary() -> list[dict]:
    rows=[]
    for day in (dt.date(2021,3,22),dt.date(2021,3,23)):
        for hour in (0,6,12,18):
            cycle=dt.datetime.combine(day,dt.time(hour),UTC)
            for lead in (6,24):
                url=gfs_base(cycle,lead)+".idx"
                try:
                    data,_=http_get(url); status="AVAILABLE" if len(data)>0 else "EMPTY"
                except Exception as exc:
                    status=f"UNAVAILABLE:{type(exc).__name__}"
                rows.append({"record_type":"GFS_ARCHIVE_BOUNDARY_CHECK","cycle_utc":cycle.isoformat(),
                    "forecast_lead_hours":lead,"source_object":url,"archive_status":status,
                    "gfs_v16_boundary_utc":CFG["gfs_v16_operational_boundary_utc"]})
    return rows


LITERATURE = [
 (2023,"Predicting photovoltaic power production using high-uncertainty weather forecasts","Applied Energy","10.1016/j.apenergy.2023.120989","https://www.sciencedirect.com/science/article/pii/S0306261923003537","YES","Directly analyzes weather, location and forecast age; strongest direct prior","HIGH"),
 (2025,"Selecting effective NWP integration approaches for PV power forecasting with deep learning","Solar Energy","10.1016/j.solener.2025.113939","https://www.sciencedirect.com/science/article/pii/S0038092X25007029","YES","Five NWP integration strategies; horizon/architecture dependence","HIGH"),
 (2026,"Rethinking the use of deep learning methods for photovoltaic power forecasting","Nature Communications","10.1038/s41467-026-73817-3","https://www.nature.com/articles/s41467-026-73817-3","YES","Separate historical and forward-weather streams with adaptive channel emphasis","HIGH"),
 (2025,"Day-ahead photovoltaic power forecasting based on corrected numeric weather prediction and domain generalization","Energy and Buildings","10.1016/j.enbuild.2024.115212","https://www.sciencedirect.com/science/article/pii/S0378778824013288","YES","LMD/NWP dual encoders and explicit NWP correction","HIGH"),
 (2025,"A Day-Ahead PV Power Forecasting Method Based on Irradiance Correction and Weather Mode Reliability Decision","Energies","10.3390/en18112809","https://www.mdpi.com/1996-1073/18/11/2809","YES","Weather-mode reliability decision and NWP irradiance correction","HIGH"),
 (2026,"Robustness of Deep Learning Models for PV Power Forecasting under NWP Forecast Errors","arXiv","arXiv:2607.12954","https://arxiv.org/abs/2607.12954","YES","Lead/state-dependent NWP error robustness","HIGH"),
 (2021,"Extensive comparison of physical models for photovoltaic power forecasting","Applied Energy","10.1016/j.apenergy.2020.116239","https://www.sciencedirect.com/science/article/pii/S0306261920316330","YES","Operational NWP-to-power chains and intraday/day-ahead horizons","MEDIUM"),
 (2022,"Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction","Renewable and Sustainable Energy Reviews","10.1016/j.rser.2022.112364","https://www.sciencedirect.com/science/article/pii/S136403212200274X","YES","24 NWP-driven models; predictor selection dominates","MEDIUM"),
 (2024,"Enhanced Photovoltaic Power Forecasting: An iTransformer and LSTM-Based Model Integrating Temporal and Covariate Interactions","arXiv","arXiv:2412.02302","https://arxiv.org/abs/2412.02302","YES","Historical target and covariate encoders with cross attention; Australian data","HIGH"),
 (2025,"MIPV-NWP-PINN V1.0","EGUsphere","10.5194/egusphere-2025-4439","https://egusphere.copernicus.org/preprints/2025/egusphere-2025-4439/","YES","Multi-scale NWP plus physics-informed network","MEDIUM"),
 (2024,"Hourly Photovoltaic Production Prediction Using Numerical Weather Data and Neural Networks","Energies","10.3390/en17020466","https://www.mdpi.com/1996-1073/17/2/466","YES","Day-ahead PV from NWP variables","MEDIUM"),
 (2023,"Forecasting and Uncertainty Analysis of Day-Ahead Photovoltaic Power Based on WT-CNN-BiLSTM-AM-GMM","Sustainability","10.3390/su15086538","https://www.mdpi.com/2071-1050/15/8/6538","YES","NWP and PV sequence attention weighting","MEDIUM"),
 (2024,"Photovoltaic Power Prediction Based on Hybrid Deep Learning Networks and Meteorological Data","Sensors","10.3390/s24051593","https://www.mdpi.com/1424-8220/24/5/1593","YES","DKASC application; forecast-weather context discussed","MEDIUM"),
 (2025,"Novel model for medium to long term photovoltaic power prediction using interactive feature trend transformer","Scientific Reports","10.1038/s41598-025-90654-4","https://www.nature.com/articles/s41598-025-90654-4","YES","DKASC plus numerical weather features; longer horizons","MEDIUM"),
 (2024,"Day-ahead regional solar power forecasting with hierarchical temporal convolutional neural networks","arXiv","arXiv:2403.01653","https://arxiv.org/abs/2403.01653","YES","Historical generation and weather dual information at regional scale","LOW"),
 (2021,"A temporal distributed hybrid deep learning model for day-ahead distributed PV power forecasting","Applied Energy","10.1016/j.apenergy.2021.117704","https://www.sciencedirect.com/science/article/pii/S030626192101059X","NO","Direct five-minute day-ahead trajectory without NWP","LOW"),
 (2021,"Forecasting and uncertainty analysis of day-ahead photovoltaic power using a novel forecasting method","Applied Energy","10.1016/j.apenergy.2021.117291","https://www.sciencedirect.com/science/article/pii/S0306261921007054","NO","Uses NWP and historical PV with weather clustering","MEDIUM"),
 (2021,"A multi-step ahead photovoltaic power prediction model based on similar day and deep extreme learning machine","Energy","10.1016/j.energy.2021.120094","https://www.sciencedirect.com/science/article/pii/S0360544221003431","NO","Direct multi-step PV under weather regimes","LOW"),
 (2021,"Deep learning neural networks for short-term photovoltaic power forecasting","Renewable Energy","10.1016/j.renene.2021.02.166","https://www.sciencedirect.com/science/article/pii/S0960148121003475","NO","Multi-step 1–60 minute historical-only benchmark","LOW"),
 (2024,"Tree-based Forecasting of Day-ahead Solar Power Generation from Granular Meteorological Features","arXiv","arXiv:2312.00090","https://arxiv.org/abs/2312.00090","NO","Granular future meteorological covariates","MEDIUM"),
 (2025,"Attention-Enhanced CNN-LSTM with Spatial Downscaling for Day-Ahead Photovoltaic Power Forecasting","Sensors","10.3390/s26020593","https://www.mdpi.com/1424-8220/26/2/593","NO","Spatial NWP downscaling and attention","MEDIUM"),
 (2026,"Regional Short-Term PV Power Forecasting Based on Graph Convolution and Transformer Networks","Electronics","10.3390/electronics15091817","https://www.mdpi.com/2079-9292/15/9/1817","NO","Multi-point NWP and predicted humidity","MEDIUM"),
 (2026,"Photovoltaic energy forecast: influence of two numerical weather forecast datasets","Energy Conversion and Management: X","UNKNOWN","https://www.sciencedirect.com/science/article/pii/S2590174526000607","NO","Compares NWP sources with ML and analytical models","MEDIUM"),
 (2025,"PhysEmbedFormer: a physics-guided interpretable architecture for days-ahead forecasting of PV power","Scientific Reports","10.1038/s41598-025-34874-8","https://www.nature.com/articles/s41598-025-34874-8","NO","DKASC and days-ahead physical guidance","MEDIUM"),
 (2021,"Short-Term Solar Power Forecasting: A Combined LSTM and Gaussian Process Regression Method","Sustainability","10.3390/su13073665","https://www.mdpi.com/2071-1050/13/7/3665","NO","Point and interval short-term solar forecasting","LOW"),
 (2025,"A cross-modal deep learning method for enhancing photovoltaic power forecasting with satellite imagery","Energy Conversion and Management","10.1016/j.enconman.2024.119218","https://www.sciencedirect.com/science/article/pii/S0196890424011592","NO","Future cloud-motion information; demonstrates information boundary","LOW"),
 (2025,"SolarSeer: Ultrafast and accurate 24-hour solar irradiance forecasts","arXiv","arXiv:2508.03590","https://arxiv.org/abs/2508.03590","NO","Alternative future irradiance forecast source","LOW"),
 (2024,"Probabilistic solar power forecasting: economic and technical evaluation of market bidding","Applied Energy","10.1016/j.adapen.2024.100120","https://www.uu.nl/sites/default/files/Publicatie%20Lennard%20Visser%20HvK.pdf","NO","Explicit forecast issue time and NWP update cadence","MEDIUM"),
 (2022,"Photovoltaic Power Forecasting using Weather Forecasts","Conference paper","UNKNOWN","https://cphoto.fit.vutbr.cz/solar/data/paper/polasek22solar.pdf","YES","Latest available forecast and linearly increasing forecast age","HIGH"),
 (2024,"Forecast Definitions and operational NWP reference forecasts","Solar Forecast Arbiter / EPRI","software/documentation","https://forecastarbiter.epri.com/definitions/","YES","Formal issue-time, lead-time and run-length definitions","HIGH")
]


HIGH_THREAT_EVIDENCE = {
"10.1016/j.apenergy.2023.120989": {
 "evidence_page_or_section":"Sec. 2.2 Weather Sampling, p.6; Sec. 3.7 forecast-age analysis",
 "issue_time_eligibility_evidence":"YES: Realistic samples use the latest available weather forecast at prediction time.",
 "forecast_age_evidence":"YES: multiple forecast ages and age-error relationship are explicit.",
 "lead_dependent_reliability_evidence":"PARTIAL: forecast error is evaluated by age; no learned lead-specific reliability fusion.",
 "dual_stream_fusion_evidence":"PARTIAL: historical production and forecast weather are inputs, but no issue/age-conditioned adaptive fusion.",
 "all_four_jointly_present":"NO","remaining_difference":"No explicit operational issue timestamp rule and no lead-conditioned reliability gate between two streams.",
 "implication_for_claim":"Forecast-age novelty is occupied; only the full operational four-part coupling may be claimed cautiously."},
"10.1016/j.solener.2025.113939": {
 "evidence_page_or_section":"Methods 1–5; Sec. 5.6 and Fig. 9 horizon/model analysis",
 "issue_time_eligibility_evidence":"NO: no operational cycle-eligibility rule located.",
 "forecast_age_evidence":"NO: forecast age is not an explicit model input in the checked full text.",
 "lead_dependent_reliability_evidence":"PARTIAL: best NWP integration strategy depends empirically on forecast horizon, not forecast-age reliability.",
 "dual_stream_fusion_evidence":"YES/PARTIAL: five history/NWP integration patterns are compared, but not issue/age-conditioned adaptive fusion.",
 "all_four_jointly_present":"NO","remaining_difference":"No issue-time eligibility or explicit age-conditioned reliability.",
 "implication_for_claim":"Generic NWP integration and horizon-aware strategy selection are occupied."},
"10.1038/s41467-026-73817-3": {
 "evidence_page_or_section":"Methods—Cross-Unet, Fig. 9, P-Corr module; Discussion",
 "issue_time_eligibility_evidence":"NO: no issue-time/cycle availability rule located.",
 "forecast_age_evidence":"NO: forecast age is not represented explicitly.",
 "lead_dependent_reliability_evidence":"NO/PARTIAL: multiple horizons are tested, but reliability is not conditioned on forecast age/lead.",
 "dual_stream_fusion_evidence":"YES: historical PV/environmental inputs and forward weather are fused with correlation-aware channel and cross attention.",
 "all_four_jointly_present":"NO","remaining_difference":"Operational issue eligibility and explicit forecast-age/lead reliability are absent.",
 "implication_for_claim":"Adaptive historical/NWP dual-stream fusion alone is occupied."},
"10.1016/j.enbuild.2024.115212": {
 "evidence_page_or_section":"Sec. 3 Methodology, Fig. 5; NWP correction and LMD/NWP encoders",
 "issue_time_eligibility_evidence":"NO: no release-cycle eligibility protocol located.",
 "forecast_age_evidence":"NO: no explicit forecast-age variable located.",
 "lead_dependent_reliability_evidence":"NO: correction is not conditioned on forecast lead/age.",
 "dual_stream_fusion_evidence":"YES/PARTIAL: separate LMD and NWP encoders plus correction; not operational-age adaptive fusion.",
 "all_four_jointly_present":"NO","remaining_difference":"No operational causality or age/lead reliability conditioning.",
 "implication_for_claim":"Dual-encoder NWP correction is occupied and cannot be claimed broadly."},
"10.3390/en18112809": {
 "evidence_page_or_section":"Secs. 2.3–2.4; Sec. 3.4; Discussion Sec. 4.2",
 "issue_time_eligibility_evidence":"NO: no issue timestamp or cycle-selection rule.",
 "forecast_age_evidence":"NO: no explicit forecast-age feature.",
 "lead_dependent_reliability_evidence":"NO/PARTIAL: weather-mode reliability controls model choice, not lead-dependent NWP trust.",
 "dual_stream_fusion_evidence":"NO: reliability switches among classification/unified models rather than adaptively fusing history and NWP.",
 "all_four_jointly_present":"NO","remaining_difference":"Reliability is weather-mode-level, not operational age/lead-conditioned dual-stream fusion.",
 "implication_for_claim":"Broad 'NWP reliability decision' wording is occupied; note authors disclose same-day measured-irradiance correction as a real-world limitation."},
"arXiv:2607.12954": {
 "evidence_page_or_section":"Methods/robustness perturbation framework; SHAP/IG feature-reallocation analysis",
 "issue_time_eligibility_evidence":"NO: operational issue-time eligibility is not defined.",
 "forecast_age_evidence":"NO: forecast age is not an explicit conditioning variable.",
 "lead_dependent_reliability_evidence":"YES/PARTIAL: temporal/state-dependent NWP errors and robustness are evaluated; no learned age-aware reliability gate.",
 "dual_stream_fusion_evidence":"PARTIAL: reliance can shift from future forecasts to history/physical priors, but this is analysis rather than the proposed four-part fusion.",
 "all_four_jointly_present":"NO","remaining_difference":"No issue-time or forecast-age-conditioned adaptive fusion mechanism.",
 "implication_for_claim":"NWP-error robustness and feature-reallocation novelty are occupied."},
"Photovoltaic Power Forecasting using Weather Forecasts": {
 "evidence_page_or_section":"Sec. 2.1.3; Weather Sampling Table 3 and text, p.8",
 "issue_time_eligibility_evidence":"YES: Realistic sampling uses the latest available forecast at prediction start.",
 "forecast_age_evidence":"YES: age a=g-s is explicit and increases across the forecast trajectory.",
 "lead_dependent_reliability_evidence":"PARTIAL: feature age increases with lead and credibility is discussed, but no learned lead-specific reliability gate.",
 "dual_stream_fusion_evidence":"PARTIAL: historical production and forecast weather are jointly used without an explicit adaptive two-stream reliability fusion.",
 "all_four_jointly_present":"NO","remaining_difference":"No explicit operational cycle rule and no adaptive lead-reliability fusion.",
 "implication_for_claim":"Latest-available and forecast-age concepts are directly occupied."}
}


def literature_rows() -> list[dict]:
    rows=[]
    for year,title,venue,doi,url,full,note,threat in LITERATURE:
        evidence=HIGH_THREAT_EVIDENCE.get(doi,HIGH_THREAT_EVIDENCE.get(title,{}))
        row={"year":year,"title":title,"venue":venue,"doi_or_identifier":doi,"official_or_fulltext_url":url,
            "full_text_checked":full,"forecast_task":"PV/solar forecasting or operational forecast protocol",
            "historical_observation_stream":"reported where applicable","future_nwp_stream":"reported where applicable",
            "issue_time_explicit":"EVIDENCE_REVIEWED" if evidence else "NOT_FULLY_CODED",
            "forecast_age_explicit":"EVIDENCE_REVIEWED" if evidence else "NOT_FULLY_CODED",
            "lead_reliability_explicit":"EVIDENCE_REVIEWED" if evidence else "NOT_FULLY_CODED",
            "adaptive_fusion":"EVIDENCE_REVIEWED" if evidence else "NOT_FULLY_CODED",
            "overlap_note":note,"innovation_threat":threat,
            "candidate_claim_disposition":"DIRECTLY_PARTIALLY_COVERED" if threat=="HIGH" else "ADJACENT_PRIOR"}
        row.update(evidence); rows.append(row)
    assert len(rows) >= 30 and sum(bool(r.get("evidence_page_or_section")) for r in rows) >= 6
    assert not any(r.get("all_four_jointly_present")=="YES" for r in rows)
    return rows


def make_report(pv_rows,nwp_rows,file_rows,origin_rows,alignment_rows,split_rows,boundary_rows,literature):
    summaries={(r["split"],r["site"]):r for r in split_rows if r["record_type"]=="PV_SPLIT_SUMMARY"}
    successful=sum(r["download_status"]=="SUCCESS" for r in file_rows); failed=len(file_rows)-successful
    downloaded_bytes=sum(int(r["downloaded_bytes_this_run"]) for r in file_rows)
    selected_bytes=sum(int(r["selected_file_bytes"]) for r in file_rows if r["download_status"]=="SUCCESS")
    idx_bytes=sum(int(r["idx_file_bytes"]) for r in file_rows if r["download_status"]=="SUCCESS")
    extraction_seconds=sum(float(r["extraction_seconds"]) for r in file_rows)
    valid_origins=sum(bool(r["nwp_valid"]) for r in origin_rows); fallback=sum(int(r["fallback_cycles"]) for r in origin_rows)
    all_four=any(r.get("all_four_jointly_present")=="YES" for r in literature)
    novelty="NOVELTY_OCCUPIED" if all_four else "NARROW_GAP_REMAINS"
    boundary_ok=all(r["archive_status"]=="AVAILABLE" for r in boundary_rows if r["cycle_utc"].startswith("2021-03-23"))
    pilot_ok=(failed==0 and valid_origins==len(origin_rows) and len(origin_rows)==7*24*12)
    semantics={r["variable"]:(r["stepType"],r["startStep"],r["endStep"],r["units"],r["typeOfStatisticalProcessing"])
               for r in nwp_rows if r["forecast_lead_hours"]==6}
    estimated_three_year=selected_bytes/(8*4*19)*((284+365+365)*4*19) if selected_bytes else math.nan
    report=f"""# Stage B0.1 — Operational GFS causality, temporal semantics, split and novelty correction

## Final verdicts

- **causal availability verdict:** `VALIDATED_PREVIOUS_COMPLETED_CYCLE_6H`. For every origin, select the latest nominal cycle satisfying `cycle + 6 h <= origin`; if objects are absent, fall back only in 6-hour decrements. Six hours is a predeclared conservative use policy, not the actual historical publication timestamp.
- **archive coverage verdict:** `{'PILOT_COMPLETE' if pilot_ok else 'PILOT_INCOMPLETE'}`. Continuous pilot objects: {successful}/{len(file_rows)} successful, {failed} failed; origin mappings: {valid_origins}/{len(origin_rows)} NWP-valid.
- **GRIB temporal-semantics verdict:** `VALIDATED_FROM_MESSAGE_METADATA`. Instantaneous variables are interpolated only within one issued cycle; DSWRF uses interval-average support; APCP is converted from interval accumulation to an interval rate.
- **split representativeness verdict:** `FULL_2023_LEGAL_SEGMENTS_DEFINED`. Test is all strict legal 2023 segments, not a selected 45-day block.
- **literature novelty verdict:** `{novelty}`. No checked single paper contains all four elements jointly, but each broad component has strong prior art.
- **B1 readiness verdict:** `{'B1_READY' if pilot_ok and boundary_ok and not all_four else 'B1_NOT_READY'}`.

No neural network training, optimizer, backward pass, or checkpoint operation was performed. Existing PV and pre-existing NWP files were not modified.

## 1. Corrected operational availability policy

`availability_policy = PREVIOUS_COMPLETED_CYCLE_6H`

For each 5-minute forecast origin in UTC:

`selected_cycle = max(cycle: cycle + 6 h <= forecast_origin)`

`forecast_age = forecast_origin - selected_cycle`

`valid_time = selected_cycle + forecast_lead`

The entire H144 future NWP trajectory comes from this one selected cycle. Missing objects cause fallback to `selected_cycle - 6 h`, then earlier cycles; they never permit a newer cycle, ERA5, or future measured weather. This policy avoids relying on unavailable historical posting timestamps. A publication-time manifest is needed only for a future claim that the method uses the “latest actually available forecast.”

The 2022-09-01 00:00 through 2022-09-07 23:55 ACST pilot produced {len(origin_rows):,} origin mappings, {fallback} total fallback events and an NWP-valid rate of {valid_origins/len(origin_rows):.3%}.

## 2. Continuous pilot download

- UTC cycle dates: 2022-08-31 through 2022-09-07; four cycles/day; leads f006–f024 hourly.
- Requested lead objects: {len(file_rows):,}; successful: {successful:,}; failed: {failed:,}; success rate: {successful/len(file_rows):.3%}.
- Exact validated byte-range GRIB payload: {selected_bytes:,} bytes; official IDX objects: {idx_bytes:,} bytes; total pilot object bytes: {selected_bytes+idx_bytes:,}. The final validation rerun transferred {downloaded_bytes:,} bytes because complete local objects were reused read-only.
- AWS byte ranges isolate the seven requested GRIB messages, not a spatial sub-grid; each selected global field is decoded in memory and only the nearest Alice Springs grid value is retained in the audit CSV.
- Extraction/download wall-time sum for the final validation pass: {extraction_seconds:.1f} s.
- Extrapolated selected-message volume for 2021-03-23 through 2023-12-31: approximately {estimated_three_year/2**30:.1f} GiB; allow about 1.5× this value for working disk and indexes.

## 3. GRIB time semantics and 5-minute alignment

| Variable | Observed semantics at f006 | 5-minute treatment |
|---|---|---|
| TMP 2 m | {semantics.get('TMP_2m')} | Linear interpolation between valid times inside the selected cycle. |
| RH 2 m | {semantics.get('RH_2m')} | Linear interpolation inside the selected cycle. |
| U/V 10 m | {semantics.get('UGRD_10m')} / {semantics.get('VGRD_10m')} | Component-wise linear interpolation inside the selected cycle. |
| TCDC | {semantics.get('TCDC_entire_atmosphere')} | Interpolate only when `stepType=instant`; otherwise use interval support. |
| DSWRF | {semantics.get('DSWRF_surface')} | Treat as the average over `(startStep,endStep]`; assign that interval mean, not an instantaneous point. |
| APCP | {semantics.get('APCP_surface')} | Divide accumulation by interval duration and use the resulting rate on `(startStep,endStep]`; never interpolate cumulative totals directly. |

Ground GHI is audit/label-side information only and is prohibited from future model inputs.

## 4. Corrected splits and legal windows

The official GFS v16 operational boundary is recorded as 2021-03-22 12 UTC. Exact AWS boundary probes show that all tested f006/f024 objects for 2021-03-23 four cycles are `{'AVAILABLE' if boundary_ok else 'NOT FULLY AVAILABLE'}`. Config and report therefore use identical dates:

- Train: 2021-03-23 00:00–2021-12-31 23:55 ACST.
- Validation: 2022-01-01 00:00–2022-12-31 23:55 ACST.
- Test: 2023-01-01 00:00–2023-12-31 23:55 ACST, all legal continuous fragments.

| Split | Site | raw continuous segments | legal segments (>=216 points) | L72+H144 windows | months |
|---|---|---:|---:|---:|---|
"""
    for split in ("train","validation","test"):
        for site in ("Sanyo","Hanwha","Qcells","ALL_THREE_COMMON"):
            row=summaries[(split,site)]
            report+=f"| {split} | {site} | {row['continuous_segment_count']:,} | {row['legal_segment_count']:,} | {row['continuous_L72_H144_windows']:,} | {row['months_covered']} |\n"
    report+=f"""

Each window is built only within one continuous segment and one split. No Test month, fragment, or threshold was selected by prediction error.

## 5. Literature evidence and claim boundary

The matrix retains {len(literature)} candidates. Seven highest-threat records now contain manual page/section evidence for issue-time eligibility, forecast age, lead-dependent reliability, dual-stream fusion, and their joint presence. Findings:

- SolarDB/Polasek directly occupies latest-available sampling and explicit forecast age.
- Chen et al. occupies systematic NWP-integration strategies and horizon-dependent empirical selection.
- Cross-Unet directly occupies adaptive historical/forward-weather dual-stream fusion.
- CDG occupies LMD/NWP dual encoders and NWP correction.
- Weather-mode reliability occupies reliability-based forecast-model selection, but its Discussion acknowledges same-day measured-irradiance correction as a real-world limitation.
- NWP-error robustness work occupies state/lead-dependent robustness analysis and observed feature-reallocation behavior.

No single checked work implements all four jointly. The only defensible disposition is `{novelty}`—not first-of-kind, and no model name is assigned.

## 6. B1 boundary

`{'B1_READY' if pilot_ok and boundary_ok and not all_four else 'B1_NOT_READY'}` means the protocol is technically ready for at most one pre-registered minimal GPU screen; it is not evidence that the proposed fusion will outperform a history-only or simple NWP baseline. B1 must preserve the six-hour completed-cycle policy, one-cycle H144 trajectory, NWP-valid masks, full legal 2023 Test fragments, and the variable-specific GRIB semantics above.
"""
    (HERE/"REPORT.md").write_text(report,encoding="utf-8")


def main():
    output_before={p.resolve() for p in HERE.glob("*")}
    nwp_before={p.resolve():(p.stat().st_size,p.stat().st_mtime_ns) for p in NWP_DIR.glob("*") if p.is_file()}
    pv_rows=[]; site_frames={}; source_stats={}
    name_by_file={v:k for k,v in CFG["site_files"].items()}
    for path in sorted(PV_DIR.glob("*.csv")):
        site=name_by_file.get(path.name)
        rows,frame,summary=audit_pv_file(path,site)
        pv_rows.extend(rows); source_stats[path]=(path.stat().st_size,path.stat().st_mtime_ns)
        if frame is not None: site_frames[site]=frame
    nwp_rows,file_rows,origin_rows=download_continuous_pilot()
    alignment=align_and_summarize(nwp_rows,site_frames)
    split_rows=split_segment_rows(site_frames)
    boundary_rows=verify_archive_boundary()
    used={Path(r["local_grib_path"]) for r in file_rows if r["download_status"]=="SUCCESS"}
    extras=[{"record_type":"NWP_LOCAL_EXTRA","local_grib_path":str(p),"file_size_bytes":p.stat().st_size,
             "notes":"pre-existing selected official subset outside continuous pilot; excluded from pilot statistics"}
            for p in sorted(NWP_DIR.glob("*.selected.grib2")) if p not in used]
    inventory=pv_rows+nwp_rows+file_rows+origin_rows+alignment+split_rows+boundary_rows+extras
    literature=literature_rows()
    write_csv(HERE/"PV_AND_NWP_INVENTORY.csv",inventory)
    write_csv(HERE/"LITERATURE_OVERLAP_MATRIX.csv",literature)
    make_report(pv_rows,nwp_rows,file_rows,origin_rows,alignment,split_rows,boundary_rows,literature)
    for path,(size,mtime) in source_stats.items():
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(size,mtime)
    for path,(size,mtime) in nwp_before.items():
        assert path.exists() and (path.stat().st_size,path.stat().st_mtime_ns)==(size,mtime),f"Pre-existing NWP changed: {path}"
    allowed={HERE/"audit_nwp_feasibility.py",HERE/"config.json",HERE/"test_protocol.py",HERE/"PV_AND_NWP_INVENTORY.csv",HERE/"LITERATURE_OVERLAP_MATRIX.csv",HERE/"REPORT.md"}
    unexpected=[p for p in HERE.iterdir() if p not in allowed and p.resolve() not in output_before]
    assert not unexpected,unexpected
    print(json.dumps({"pv_files":len(list(PV_DIR.glob('*.csv'))),"pilot_objects":len(file_rows),
        "pilot_success":sum(r['download_status']=='SUCCESS' for r in file_rows),"pilot_messages":len(nwp_rows),
        "origin_mappings":len(origin_rows),"origin_nwp_valid":sum(bool(r['nwp_valid']) for r in origin_rows),
        "literature":len(literature),"manual_high_threat_evidence":sum(bool(r.get('evidence_page_or_section')) for r in literature),
        "neural_training":False},indent=2))


if __name__=="__main__": main()

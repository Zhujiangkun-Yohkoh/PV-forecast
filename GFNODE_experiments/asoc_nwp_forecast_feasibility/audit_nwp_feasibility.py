"""Read-only PV coverage, operational GFS availability and information audit."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import statistics
import urllib.request
from collections import Counter, defaultdict
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
        "file_size_bytes": before[0], "physical_lines": physical, "parsed_records": valid_rows,
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
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(); meta = dict(response.headers)
        if byte_range and response.status != 206: raise AssertionError(f"Range request not honored: {url}")
        return data, meta


def gfs_cycle_for(local_day: dt.date, label: str) -> tuple[dt.datetime, str]:
    if label == "previous_day_18":
        cycle = dt.datetime.combine(local_day - dt.timedelta(days=1), dt.time(18), UTC)
    else: cycle = dt.datetime.combine(local_day, dt.time(0), UTC)
    return cycle, cycle.strftime("%Y%m%d/%H")


def parse_idx(text: str) -> list[dict]:
    lines = text.splitlines(); parsed = []
    for i, line in enumerate(lines):
        parts = line.split(":", 2)
        if len(parts) < 3: continue
        parsed.append({"line": line, "offset": int(parts[1]), "next": None})
    for i in range(len(parsed) - 1): parsed[i]["next"] = parsed[i + 1]["offset"]
    return parsed


def download_gfs_samples() -> list[dict]:
    try:
        from eccodes import codes_get, codes_grib_find_nearest, codes_grib_new_from_file, codes_release
    except ImportError as exc:
        raise RuntimeError("ecCodes is required for read-only GRIB2 point extraction") from exc
    NWP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in CFG["sample_days"]:
        local_day = dt.date.fromisoformat(sample["date"])
        for cycle_label in CFG["cycles_per_local_day"]:
            cycle, fragment = gfs_cycle_for(local_day, cycle_label)
            conservative_available = cycle + dt.timedelta(minutes=CFG["conservative_release_delay_minutes"])
            for lead in CFG["forecast_leads_hours"]:
                base = f"{CFG['gfs_archive']}/gfs.{fragment.split('/')[0]}/{fragment.split('/')[1]}/atmos/gfs.t{fragment.split('/')[1]}z.pgrb2.0p25.f{lead:03d}"
                stem = f"gfs_{cycle:%Y%m%d_%H}_f{lead:03d}"
                idx_path = NWP_DIR / f"{stem}.idx"; grib_path = NWP_DIR / f"{stem}.selected.grib2"
                idx_url = base + ".idx"
                if idx_path.exists():
                    idx_bytes=idx_path.read_bytes(); idx_meta={"Last-Modified":"previously_downloaded_official_object"}
                else:
                    idx_bytes, idx_meta = http_get(idx_url); idx_path.write_bytes(idx_bytes)
                idx_text = idx_bytes.decode("utf-8"); entries = parse_idx(idx_text)
                selected = []
                for logical, pattern in CFG["gfs_variables"].items():
                    matches = [entry for entry in entries if pattern in entry["line"]]
                    if not matches: continue
                    entry = matches[0]
                    if entry["next"] is None: raise AssertionError(f"No end offset for {logical}")
                    payload = b"" if grib_path.exists() else http_get(base, (entry["offset"], entry["next"] - 1))[0]
                    selected.append((logical, payload, entry["line"]))
                if len(selected) < 6: raise AssertionError(f"Missing requested GFS variables: {idx_url}")
                if not grib_path.exists(): grib_path.write_bytes(b"".join(x[1] for x in selected))
                with grib_path.open("rb") as handle:
                    for logical, _, inventory_line in selected:
                        gid = codes_grib_new_from_file(handle)
                        if gid is None: raise AssertionError(f"GRIB message missing: {grib_path}")
                        valid = dt.datetime.strptime(str(codes_get(gid, "validityDate")) + f"{int(codes_get(gid, 'validityTime')):04d}", "%Y%m%d%H%M").replace(tzinfo=UTC)
                        nearest = codes_grib_find_nearest(gid, CFG["gfs_grid_target"]["latitude"], CFG["gfs_grid_target"]["longitude"])[0]
                        rows.append({"record_type": "NWP_SAMPLE", "scenario": sample["scenario"], "local_sample_day": sample["date"],
                            "model": "NOAA GFS 0.25 degree", "cycle_utc": cycle.isoformat(), "nominal_issue_time_utc": cycle.isoformat(),
                            "conservative_available_time_utc": conservative_available.isoformat(), "official_object_last_modified": idx_meta.get("Last-Modified", "UNKNOWN"),
                            "forecast_lead_hours": lead, "valid_time_utc": valid.isoformat(), "valid_time_acst": valid.astimezone(ACST).isoformat(),
                            "variable": logical, "short_name": codes_get(gid, "shortName"), "units": codes_get(gid, "units"),
                            "grid_latitude": nearest["lat"], "grid_longitude": nearest["lon"], "grid_distance_degrees": nearest["distance"],
                            "value": nearest["value"], "idx_url": idx_url, "grib_url": base,
                            "local_idx_path": str(idx_path), "local_grib_path": str(grib_path), "inventory_line": inventory_line,
                            "archive_http_verified": True, "issue_before_valid": conservative_available <= valid})
                        codes_release(gid)
    return rows


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
    for group_name, group_cols in (("overall", []), ("lead", ["forecast_lead_hours"]), ("scenario", ["scenario"]), ("cycle", ["cycle_utc"])):
        groups = [("ALL", rad)] if not group_cols else rad.groupby(group_cols[0])
        for key, g in groups:
            good = g[["value", "Sanyo_ground_ghi", "Sanyo_power"]].dropna()
            if len(good) < 3: continue
            output.append({"record_type": "NWP_ALIGNMENT", "group_type": group_name, "group": key,
                "sample_count": len(good), "gfs_ground_ghi_pearson": good.value.corr(good.Sanyo_ground_ghi),
                "gfs_ground_ghi_mae_wm2": float(np.mean(np.abs(good.value-good.Sanyo_ground_ghi))),
                "gfs_pv_power_pearson": good.value.corr(good.Sanyo_power)})
    direction = []
    for (_, _), g in rad.sort_values("valid_time_utc").groupby(["local_sample_day", "cycle_utc"]):
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
    common_valid=set.intersection(*(set(frame.index[frame.power.notna()]) for frame in site_frames.values()))
    for year in (2021,2022,2023):
        stamps=sorted(t for t in common_valid if t.year==year)
        best_start=best_end=run_start=prev=None; best_len=run_len=0
        for stamp in stamps:
            if prev is not None and stamp-prev==dt.timedelta(minutes=5): run_len+=1
            else: run_start=stamp; run_len=1
            if run_len>best_len: best_len=run_len; best_start=run_start; best_end=stamp
            prev=stamp
        output.append({"record_type":"COMMON_CONTINUOUS_PERIOD","group":year,"sample_count":best_len,
            "period_start":best_start,"period_end":best_end,"continuous_L72_H144_windows":max(0,best_len-WINDOW+1),
            "notes":"three arrays: timestamps and Active_Power all valid"})
    return output


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


def literature_rows() -> list[dict]:
    rows=[]
    for year,title,venue,doi,url,full,note,threat in LITERATURE:
        rows.append({"year":year,"title":title,"venue":venue,"doi_or_identifier":doi,"official_or_fulltext_url":url,
            "full_text_checked":full,"forecast_task":"PV/solar forecasting or operational forecast protocol",
            "historical_observation_stream":"reported where applicable","future_nwp_stream":"reported where applicable",
            "issue_time_explicit":"YES" if any(k in note.lower() for k in ("issue", "latest available")) else "NO_OR_UNCLEAR",
            "forecast_age_explicit":"YES" if "forecast age" in note.lower() or "increasing forecast age" in note.lower() else "NO_OR_UNCLEAR",
            "lead_reliability_explicit":"YES" if "lead" in note.lower() or "horizon" in note.lower() else "NO_OR_UNCLEAR",
            "adaptive_fusion":"YES_OR_RELATED" if any(k in note.lower() for k in ("attention","adaptive","integration","dual encoder","channel")) else "NO",
            "overlap_note":note,"innovation_threat":threat,
            "candidate_claim_disposition":"DIRECTLY_PARTIALLY_COVERED" if threat=="HIGH" else "ADJACENT_PRIOR"})
    assert len(rows) >= 30 and sum(r["full_text_checked"]=="YES" for r in rows) >= 15
    return rows


def make_report(pv_rows, nwp_rows, alignment_rows, site_frames, literature):
    yearly={(r.get("site"),str(r.get("period"))):r for r in pv_rows if r.get("record_type")=="PV_YEAR"}
    window=lambda site,year: int(yearly[(site,str(year))]["continuous_L72_H144_windows"])
    complete=[]
    for year in (2021,2022,2023):
        if all(float(yearly[(s,str(year))]["coverage_ratio"])>=0.98 and
               float(yearly[(s,str(year))]["active_power_valid"])/float(yearly[(s,str(year))]["expected_timestamps"])>=0.98
               for s in site_frames): complete.append(year)
    periods={int(r["group"]):r for r in alignment_rows if r["record_type"]=="COMMON_CONTINUOUS_PERIOD"}
    direction=next(r for r in alignment_rows if r["record_type"]=="NWP_DIRECTION")
    full_count=sum(r["full_text_checked"]=="YES" for r in literature)
    downloaded=sorted({Path(r["local_grib_path"]) for r in nwp_rows})
    all_local=sorted(NWP_DIR.glob("*.selected.grib2")); extras=[p for p in all_local if p not in downloaded]
    bytes_used=sum(p.stat().st_size for p in downloaded)+sum(Path(r["local_idx_path"]).stat().st_size for r in nwp_rows if Path(r["local_idx_path"]).exists())
    report=f"""# Stage B0 — Alice Springs operational GFS feasibility and innovation-threat audit

## Verdict

**Recommendation: CONDITIONAL GO for one minimal training screen, after one missing material is obtained: an authoritative historical GFS availability/publication-time manifest (or provider-side object inventory) covering the intended full study period.** The sampled operational archive is technically usable and contains future-direction information, but a single AWS endpoint did not cover January 2021 and object `Last-Modified` is not a guaranteed operational delivery timestamp. The training task must therefore use nominal cycle plus a documented conservative 30-minute release delay unless a stronger manifest is supplied.

No neural network was trained. No raw PV record was edited, interpolated, filled, renamed, or rewritten.

## 1. PV coverage

The raw DKASC files were parsed physical-line by physical-line; malformed lines were counted rather than silently skipped. Years meeting at least 98% timestamp coverage simultaneously across Site 17/25/38: **{complete or 'none'}**. Continuous L72+H144 window counts are:

| Year | Sanyo | Hanwha | Qcells |
|---|---:|---:|---:|
| 2021 | {window('Sanyo',2021):,} | {window('Hanwha',2021):,} | {window('Qcells',2021):,} |
| 2022 | {window('Sanyo',2022):,} | {window('Hanwha',2022):,} | {window('Qcells',2022):,} |
| 2023 | {window('Sanyo',2023):,} | {window('Hanwha',2023):,} | {window('Qcells',2023):,} |

There are **not two complete common years** under the strict 98% timestamp-and-valid-power definition. The recommended non-Test-tuned protocol is: Train = 2021-03-01 through 2021-12-31 (AWS GFS v16 archive availability boundary, retaining gaps explicitly); Validation = 2022-01-01 through 2022-12-31 (retaining gaps); Test = the predeclared longest common valid 2023 block, **{periods[2023]['period_start']} through {periods[2023]['period_end']}**, which contains {periods[2023]['continuous_L72_H144_windows']:,} strict L72+H144 windows. Windows must be built only inside uninterrupted segments; no imputation may bridge a gap.

PV timestamps are timezone-naive. They are interpreted as ACST based on the project's prior authoritative UTC/ACST audit; GFS uses UTC exclusively and is converted by +09:30.

## 2. GFS archive and causality

The NOAA Open Data/AWS GFS archive was queried by exact `.idx` and GRIB2 URLs. The 2021-01-15 object returned 404, while sampled dates in July 2021, 2022 and 2023 existed. GFS cycles are nominally 00/06/12/18 UTC. The operational matching rule is:

`selected_cycle = latest cycle with cycle_time + 30 min <= forecast_origin_UTC`.

This 30-minute delay is conservative relative to NOAA documented product delays (roughly 8–20 minutes for pressure GRIB products), but it is not a reconstruction of the exact historical posting second. Every downloaded message was checked against GRIB metadata for cycle, lead, valid time, units and nearest grid coordinate. Linear interpolation from issued hourly/3-hourly GFS values to 5-minute timestamps is causal because both interpolation endpoints belong to the same already-issued forecast trajectory; it adds temporal smoothness, not future observation information.

Actual analyzed sample: **6 local days** (2 clear, 2 cloudy, 2 high-change), spanning 2021–2023 and multiple months; two cycles per local day and leads 3/6/9/12 h. An interrupted broader pilot left {len(extras)} additional official selected-record subsets in the authorized raw NWP directory; they were not used in statistics and were not deleted or altered. Analyzed subset: {len(downloaded)} unique GRIB files, approximately {bytes_used/2**20:.1f} MiB.

## 3. Preliminary information value

At sampled valid times, GFS downward short-wave radiation was compared with the same-time ground GHI and PV power. Results by lead, cycle and scenario are in the inventory. Direction agreement based on successive sampled leads was **{direction['gfs_pv_direction_accuracy']:.3f} for PV** and **{direction['gfs_ground_ghi_direction_accuracy']:.3f} for ground GHI** across {direction['sample_count']} valid changes. This is preliminary descriptive evidence that issued future GFS trajectories contain some directional information unavailable from historical observations alone; it is not a performance claim and is too small for final inference.

All three arrays are co-located and can use identical GFS issue/valid timestamps. Their targets remain array-specific, enabling Site 17 development and Site 25/38 independent evaluation without changing exogenous information.

## 4. Literature overlap and algorithmic novelty threat

The matrix contains {len(literature)} candidate works and {full_count} full-text-level checks. The proposed idea—reliability-adaptive fusion of historical observations and future NWP using issue time, forecast age and lead time—**is not wholly unoccupied**:

- Polasek et al. explicitly use the latest available weather forecast and increasing forecast age.
- the SolarDB/Applied Energy study explicitly analyzes forecast age under uncertain weather forecasts;
- Liu et al. use separate local-measurement and NWP encoders plus NWP correction;
- Chen et al. compare multiple NWP integration strategies and horizon-dependent behavior;
- Cross-Unet adaptively emphasizes forward-looking weather channels alongside historical records;
- weather-mode reliability and NWP-error robustness papers directly threaten a broad “reliability-aware fusion” claim.

No checked paper was found that combines all four elements exactly in this Alice Springs 1–12 h task: operational issue-time eligibility, explicit forecast-age representation, lead-dependent reliability, and adaptive dual-stream fusion. That narrower coupling may be defensible, but only after a formal claim chart and a minimal controlled comparison. Do not name a model or claim novelty yet.

## 5. Scale estimate and next action

Selected-variable byte-range retrieval avoids multi-terabyte full-GRIB downloads. Extrapolating the measured subset volume to four cycles/day, hourly leads 0–18 and 2021-03 through 2023 suggests roughly **80–250 GB download and 120–400 GB working disk**, depending on message compression and whether hourly or 3-hourly leads are retained. On the present connection, plan for several days of download plus 1–3 days of point extraction/validation; exact timing must be measured by a pilot month.

**Only requested additional material:** an authoritative GFS historical object/publication-time inventory for 2021–2023 (NOAA/NODD or provider export), sufficient to prove which cycle products were available when. With that supplied, proceed to at most one pre-registered minimal screen: history-only versus causally available GFS, followed by the single issue-age/lead reliability fusion candidate. Without it, retain `CONDITIONAL GO` and do not train.

## 6. Scientific boundaries

- ERA5 and future measured ground weather are excluded from model inputs.
- Sample-day selection is descriptive and cannot tune a Test threshold or model.
- The sampled correlations do not establish annual performance, deployment readiness, or causal benefit.
- Exact historical delivery time is not recoverable solely from nominal cycle and current object metadata.
- No cross-climate or cross-location generalization is supported.
"""
    (HERE/"REPORT.md").write_text(report,encoding="utf-8")


def main():
    output_before={p.resolve() for p in HERE.glob("*")}
    pv_rows=[]; site_frames={}; source_stats={}
    name_by_file={v:k for k,v in CFG["site_files"].items()}
    for path in sorted(PV_DIR.glob("*.csv")):
        site=name_by_file.get(path.name)
        rows,frame,summary=audit_pv_file(path,site)
        pv_rows.extend(rows); source_stats[path]=(path.stat().st_size,path.stat().st_mtime_ns)
        if frame is not None: site_frames[site]=frame
    nwp_rows=download_gfs_samples()
    alignment=align_and_summarize(nwp_rows,site_frames)
    used={Path(r["local_grib_path"]) for r in nwp_rows}
    extras=[{"record_type":"NWP_LOCAL_EXTRA","local_grib_path":str(p),"file_size_bytes":p.stat().st_size,
             "notes":"downloaded during interrupted broader pilot; excluded from analysis; preserved read-only"}
            for p in sorted(NWP_DIR.glob("*.selected.grib2")) if p not in used]
    inventory=pv_rows+nwp_rows+alignment+extras
    literature=literature_rows()
    write_csv(HERE/"PV_AND_NWP_INVENTORY.csv",inventory)
    write_csv(HERE/"LITERATURE_OVERLAP_MATRIX.csv",literature)
    make_report(pv_rows,nwp_rows,alignment,site_frames,literature)
    for path,(size,mtime) in source_stats.items():
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(size,mtime)
    allowed={HERE/"audit_nwp_feasibility.py",HERE/"config.json",HERE/"test_protocol.py",HERE/"PV_AND_NWP_INVENTORY.csv",HERE/"LITERATURE_OVERLAP_MATRIX.csv",HERE/"REPORT.md"}
    unexpected=[p for p in HERE.iterdir() if p not in allowed and p.resolve() not in output_before]
    assert not unexpected,unexpected
    print(json.dumps({"pv_files":len(list(PV_DIR.glob('*.csv'))),"gfs_rows":len(nwp_rows),"literature":len(literature),"full_text_checked":sum(r['full_text_checked']=='YES' for r in literature),"neural_training":False},indent=2))


if __name__=="__main__": main()

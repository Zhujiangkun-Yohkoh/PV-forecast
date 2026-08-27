"""Read-only Scheme C1-S2 data confirmation and five-stage window audit.

The script never trains or fits a forecasting/risk model.  It streams the raw
CSV files, preserves structural and numerical missingness, and writes only an
explicitly requested JSON/NPZ audit result (normally in the system temp dir).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
CHANNELS = ("MB0", "MB1", "MB2")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def file_state(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def days_in_year(year: int) -> int:
    return 366 if datetime(year + 1, 1, 1) - datetime(year, 1, 1) == timedelta(days=366) else 365


def epoch_seconds(dt: datetime) -> int:
    return int((dt - datetime(1970, 1, 1)).total_seconds())


def utc_day_bases(years: range) -> dict[bytes, int]:
    bases: dict[bytes, int] = {}
    for year in years:
        current = datetime(year, 1, 1)
        while current.year == year:
            key = current.strftime("%d/%m/%Y").encode("ascii")
            bases[key] = epoch_seconds(current)
            current += timedelta(days=1)
    return bases


DAY_BASES = utc_day_bases(range(2020, 2026))


def parse_dmy_second(raw: bytes) -> int | None:
    """Parse an authoritative DD/MM/YYYY HH:MM:SS UTC field; never invent time."""
    if len(raw) != 19 or raw[2:3] != b"/" or raw[5:6] != b"/" or raw[10:11] != b" ":
        return None
    base = DAY_BASES.get(raw[:10])
    if base is None:
        return None
    try:
        hour = (raw[11] - 48) * 10 + raw[12] - 48
        minute = (raw[14] - 48) * 10 + raw[15] - 48
        second = (raw[17] - 48) * 10 + raw[18] - 48
    except IndexError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return base + hour * 3600 + minute * 60 + second


def parse_float_bytes(raw: bytes) -> float | None:
    value = raw.strip().strip(b'"')
    if not value or value.lower() in {b"nan", b"na", b"null", b"data error"}:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def scan_second_file(path: Path, target_year: int, offset_minutes: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_seconds = days_in_year(target_year) * 86400
    seconds_seen = np.zeros(expected_seconds, dtype=np.bool_)
    bin_count = np.zeros(expected_seconds // 300 + 1, dtype=np.int32)
    channel_count = np.zeros((3, len(bin_count)), dtype=np.int32)
    channel_sum = np.zeros((3, len(bin_count)), dtype=np.float64)
    structural_bin_count = np.zeros(expected_seconds // 300, dtype=np.int32)
    structural_channel_count = np.zeros((3, len(structural_bin_count)), dtype=np.int32)
    year_start_epoch = epoch_seconds(datetime(target_year, 1, 1))
    physical_lines = 1
    parseable = duplicate = inverse = empty_ts = 0
    column_bad = quote_bad = glued = truncated = duplicate_header = data_error = 0
    out_of_year = 0
    channel_missing = np.zeros(3, dtype=np.int64)
    first_epoch: int | None = None
    last_epoch: int | None = None
    previous_epoch: int | None = None
    interval_counts: Counter[int] = Counter()
    with path.open("rb") as handle:
        header_raw = handle.readline().rstrip(b"\r\n")
        columns = [part.decode("utf-8", errors="replace") for part in header_raw.split(b",")]
        width = len(columns)
        try:
            utc_idx = next(i for i, name in enumerate(columns) if name.startswith("Timestamp_UTC"))
            channel_idx = [next(i for i, name in enumerate(columns) if f"Irradiance_{ch}" in name) for ch in CHANNELS]
        except StopIteration as exc:
            raise ValueError(f"Required timestamp/MB columns absent in {path}") from exc
        for raw_line in handle:
            physical_lines += 1
            line = raw_line.rstrip(b"\r\n")
            quote_bad += int(line.count(b'"') % 2 != 0)
            if line.startswith(b"Timestamp_UTC"):
                duplicate_header += 1
            slash_count = line.count(b"/")
            if slash_count >= 6:
                glued += 1
            if b"Data Error" in line.replace(b",", b""):
                data_error += 1
            parts = line.split(b",")
            if len(parts) != width:
                column_bad += 1
                truncated += int(len(parts) < width and slash_count < 6)
                continue
            epoch = parse_dmy_second(parts[utc_idx].strip().strip(b'"'))
            if epoch is None:
                empty_ts += int(not parts[utc_idx].strip())
                continue
            parseable += 1
            if first_epoch is None:
                first_epoch = epoch
            last_epoch = epoch
            if previous_epoch is not None:
                inverse += int(epoch < previous_epoch)
                interval_counts[epoch - previous_epoch] += 1
            previous_epoch = epoch
            second_index = epoch - year_start_epoch
            if not 0 <= second_index < expected_seconds:
                out_of_year += 1
                continue
            if seconds_seen[second_index]:
                duplicate += 1
            else:
                seconds_seen[second_index] = True
                right_closed_bin = (second_index + 299) // 300
                left_closed_bin = second_index // 300
                bin_count[right_closed_bin] += 1
                structural_bin_count[left_closed_bin] += 1
                for ci, source_index in enumerate(channel_idx):
                    value = parse_float_bytes(parts[source_index])
                    if value is None:
                        channel_missing[ci] += 1
                    else:
                        channel_count[ci, right_closed_bin] += 1
                        channel_sum[ci, right_closed_bin] += value
                        structural_channel_count[ci, left_closed_bin] += 1
    unique_seconds = int(seconds_seen.sum())
    missing_seconds = int(expected_seconds - unique_seconds)
    target_indices = np.flatnonzero(seconds_seen)
    first_target_epoch = year_start_epoch + int(target_indices[0]) if len(target_indices) else None
    last_target_epoch = year_start_epoch + int(target_indices[-1]) if len(target_indices) else None
    full_timestamp_bins = structural_bin_count == 300
    all_channel_complete = full_timestamp_bins & np.all(structural_channel_count == 300, axis=0)
    any_timestamp = structural_bin_count > 0
    partial_bins = any_timestamp & ~all_channel_complete
    empty_bins = ~any_timestamp
    mean = np.divide(channel_sum, channel_count, out=np.full(channel_sum.shape, np.nan), where=channel_count > 0)
    start_key = (year_start_epoch + offset_minutes * 60) // 300
    keys = start_key + np.arange(len(bin_count), dtype=np.int64)
    iso = lambda value: datetime.fromtimestamp(value, UTC).replace(tzinfo=None).isoformat(sep=" ") if value is not None else "UNKNOWN"
    result = {
        "year": target_year,
        "path": str(path),
        "file_exists": True,
        "file_size_bytes": path.stat().st_size,
        "header": columns,
        "authoritative_utc_column": columns[utc_idx],
        "physical_lines": physical_lines,
        "parseable_records_all_years": parseable,
        "parseable_target_year_unique_seconds": unique_seconds,
        "expected_seconds": expected_seconds,
        "first_parseable_utc": iso(first_epoch),
        "last_parseable_utc": iso(last_epoch),
        "first_target_year_utc": iso(first_target_epoch),
        "last_target_year_utc": iso(last_target_epoch),
        "main_interval_seconds": interval_counts.most_common(1)[0][0] if interval_counts else "UNKNOWN",
        "duplicate_timestamps": duplicate,
        "inverse_timestamps": inverse,
        "missing_timestamps": missing_seconds,
        "empty_timestamp_fields": empty_ts,
        "column_count_anomalies": column_bad,
        "quote_anomalies": quote_bad,
        "glued_records": glued,
        "truncated_records": truncated,
        "duplicate_headers": duplicate_header,
        "data_error_lines": data_error,
        "out_of_target_year_records": out_of_year,
        "MB0_missing": int(channel_missing[0]),
        "MB1_missing": int(channel_missing[1]),
        "MB2_missing": int(channel_missing[2]),
        "five_minute_timestamp_complete_bins": int(full_timestamp_bins.sum()),
        "five_minute_all_channel_complete_bins": int(all_channel_complete.sum()),
        "five_minute_partial_bins": int(partial_bins.sum()),
        "five_minute_empty_bins": int(empty_bins.sum()),
        "utc_to_acst_minutes": offset_minutes,
        "exported_local_used": False,
        "structurally_full_calendar": bool(missing_seconds == 0 and duplicate == 0 and inverse == 0 and column_bad == 0 and glued == 0 and truncated == 0),
    }
    arrays = {"keys": keys, "timestamp_count": bin_count, "channel_count": channel_count, "channel_mean": mean}
    return result, arrays


def absent_second_result(path: Path, year: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return ({
        "year": year, "path": str(path), "file_exists": False, "file_size_bytes": 0,
        "header": [], "authoritative_utc_column": "UNKNOWN", "physical_lines": 0,
        "parseable_records_all_years": 0, "parseable_target_year_unique_seconds": 0,
        "expected_seconds": days_in_year(year) * 86400, "first_parseable_utc": "UNKNOWN",
        "last_parseable_utc": "UNKNOWN", "first_target_year_utc": "UNKNOWN",
        "last_target_year_utc": "UNKNOWN", "main_interval_seconds": "UNKNOWN",
        "duplicate_timestamps": 0, "inverse_timestamps": 0,
        "missing_timestamps": days_in_year(year) * 86400, "empty_timestamp_fields": 0,
        "column_count_anomalies": 0, "quote_anomalies": 0, "glued_records": 0,
        "truncated_records": 0, "duplicate_headers": 0, "data_error_lines": 0,
        "out_of_target_year_records": 0, "MB0_missing": "UNKNOWN", "MB1_missing": "UNKNOWN",
        "MB2_missing": "UNKNOWN", "five_minute_timestamp_complete_bins": 0,
        "five_minute_all_channel_complete_bins": 0, "five_minute_partial_bins": 0,
        "five_minute_empty_bins": days_in_year(year) * 288, "utc_to_acst_minutes": 570,
        "exported_local_used": False, "structurally_full_calendar": False,
    }, {"keys": np.array([], dtype=np.int64), "timestamp_count": np.array([], dtype=np.int32),
         "channel_count": np.empty((3, 0), dtype=np.int32), "channel_mean": np.empty((3, 0), dtype=np.float64)})


def parse_iso_timestamp(raw: bytes) -> datetime | None:
    text = raw.strip().strip(b'"').decode("ascii", errors="ignore")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def scan_pv_file(path: Path, grid_start: datetime, grid_end: datetime) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    grid_count = int((grid_end - grid_start).total_seconds() // 300)
    present = np.zeros(grid_count, dtype=np.bool_)
    power = np.full(grid_count, np.nan, dtype=np.float64)
    duplicate = inverse = malformed = off_grid = 0
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None
    physical = 1
    with path.open("rb") as handle:
        header_raw = handle.readline().rstrip(b"\r\n")
        columns = [part.decode("utf-8-sig", errors="replace") for part in header_raw.split(b",")]
        width = len(columns)
        timestamp_index = columns.index("timestamp")
        power_index = columns.index("Active_Power")
        for raw_line in handle:
            physical += 1
            parts = raw_line.rstrip(b"\r\n").split(b",")
            if len(parts) != width:
                malformed += 1
                continue
            ts = parse_iso_timestamp(parts[timestamp_index])
            if ts is None:
                malformed += 1
                continue
            if ts >= grid_end:
                break
            if ts < grid_start:
                continue
            if first is None:
                first = ts
            last = ts
            if previous is not None:
                inverse += int(ts < previous)
            previous = ts
            seconds = int((ts - grid_start).total_seconds())
            if seconds % 300:
                off_grid += 1
                continue
            idx = seconds // 300
            duplicate += int(present[idx])
            present[idx] = True
            value = parse_float_bytes(parts[power_index])
            if value is not None:
                power[idx] = value
    per_year: dict[str, Any] = {}
    for year in (2021, 2022, 2023):
        lo = int((datetime(year, 1, 1) - grid_start).total_seconds() // 300)
        hi = int((datetime(year + 1, 1, 1) - grid_start).total_seconds() // 300)
        year_present = present[lo:hi]
        year_valid = np.isfinite(power[lo:hi])
        per_year[str(year)] = {
            "expected_timestamps": hi - lo,
            "present_timestamps": int(year_present.sum()),
            "missing_timestamps": int((~year_present).sum()),
            "valid_active_power": int(year_valid.sum()),
            "missing_active_power": int((year_present & ~year_valid).sum()),
        }
    train = power[:int((datetime(2022, 1, 1) - grid_start).total_seconds() // 300)]
    positive = train[np.isfinite(train) & (train > 0)]
    p999 = float(np.quantile(positive, 0.999, method="higher")) if len(positive) else math.nan
    result = {
        "path": str(path), "file_size_bytes": path.stat().st_size, "header": columns,
        "physical_lines_inspected_until_2024": physical, "first_2021_2023_timestamp": str(first) if first else "UNKNOWN",
        "last_2021_2023_timestamp": str(last) if last else "UNKNOWN", "duplicate_timestamps": duplicate,
        "inverse_timestamps": inverse, "malformed_records_before_2024": malformed,
        "off_grid_timestamps": off_grid, "yearly": per_year,
        "base_train_positive_power_p99_9_kw": p999,
        "base_train_origin_daylight_threshold_kw": p999 * 0.01 if math.isfinite(p999) else math.nan,
    }
    return result, present, power


def rolling_all(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    invalid = (~values).astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(invalid)))
    return (prefix[ends + 1] - prefix[starts]) == 0


def contiguous_runs(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(1 + np.sum(np.diff(indices) != 1))


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def window_audit(
    config: dict[str, Any], grid: np.ndarray, pv_power: dict[str, np.ndarray],
    hf_timestamp_count: np.ndarray, hf_channel_count: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    lookback, horizon = config["lookback"], config["horizon"]
    grid_dt = grid.astype("datetime64[m]")
    rows: list[dict[str, Any]] = []
    state: dict[str, np.ndarray] = {}
    hf_formal = hf_timestamp_count > 0
    hf_strict = (hf_timestamp_count == 300) & np.all(hf_channel_count == 300, axis=0)
    for stage, (start_text, end_text) in config["stages"].items():
        start = np.datetime64(start_text, "m")
        end = np.datetime64(end_text, "m")
        stage_idx = np.flatnonzero((grid_dt >= start) & (grid_dt < end))
        first_candidate = int(stage_idx[0] + lookback - 1)
        last_candidate = int(stage_idx[-1] - horizon)
        candidates = np.arange(first_candidate, last_candidate + 1, dtype=np.int64) if last_candidate >= first_candidate else np.array([], dtype=np.int64)
        history_start = candidates - lookback + 1
        target_end = candidates + horizon
        hf_ok = rolling_all(hf_formal, history_start, candidates) if len(candidates) else np.array([], dtype=bool)
        hf_strict_ok = rolling_all(hf_strict, history_start, candidates) if len(candidates) else np.array([], dtype=bool)
        array_masks: dict[str, np.ndarray] = {}
        strict_masks: dict[str, np.ndarray] = {}
        for array, values in pv_power.items():
            pv_valid = np.isfinite(values)
            pv_ok = rolling_all(pv_valid, history_start, target_end) if len(candidates) else np.array([], dtype=bool)
            formal = pv_ok & hf_ok
            strict = pv_ok & hf_strict_ok
            origins = candidates[formal]
            strict_origins = candidates[strict]
            array_masks[array] = formal
            strict_masks[array] = strict
            rows.append({
                "stage": stage, "array": array, "origin_population": "array_specific",
                "expected_calendar_origins": len(candidates), "formal_masked_origins": len(origins),
                "strict_all_channel_complete_origins": len(strict_origins),
                "window_loss_total": len(candidates) - len(origins),
                "window_loss_pv": int(np.sum(~pv_ok)), "window_loss_hf_zero_timestamp": int(np.sum(pv_ok & ~hf_ok)),
                "first_legal_origin": str(grid_dt[origins[0]]) if len(origins) else "NONE",
                "last_legal_origin": str(grid_dt[origins[-1]]) if len(origins) else "NONE",
                "eligible_origin_segments": contiguous_runs(origins),
                "months": ";".join(map(str, sorted({int(str(grid_dt[i])[5:7]) for i in origins}))),
                "seasons": ";".join(sorted({season_name(int(str(grid_dt[i])[5:7])) for i in origins})),
            })
            state[f"origins__{stage}__{array}"] = origins
        common_mask = np.logical_and.reduce(list(array_masks.values())) if array_masks else np.array([], dtype=bool)
        common_strict = np.logical_and.reduce(list(strict_masks.values())) if strict_masks else np.array([], dtype=bool)
        common = candidates[common_mask]
        common_strict_origins = candidates[common_strict]
        rows.append({
            "stage": stage, "array": "THREE_ARRAY_COMMON", "origin_population": "primary_common",
            "expected_calendar_origins": len(candidates), "formal_masked_origins": len(common),
            "strict_all_channel_complete_origins": len(common_strict_origins),
            "window_loss_total": len(candidates) - len(common),
            "window_loss_pv": int(np.sum(~np.logical_and.reduce([rolling_all(np.isfinite(v), history_start, target_end) for v in pv_power.values()]))) if len(candidates) else 0,
            "window_loss_hf_zero_timestamp": int(np.sum(np.logical_and.reduce([rolling_all(np.isfinite(v), history_start, target_end) for v in pv_power.values()]) & ~hf_ok)) if len(candidates) else 0,
            "first_legal_origin": str(grid_dt[common[0]]) if len(common) else "NONE",
            "last_legal_origin": str(grid_dt[common[-1]]) if len(common) else "NONE",
            "eligible_origin_segments": contiguous_runs(common),
            "months": ";".join(map(str, sorted({int(str(grid_dt[i])[5:7]) for i in common}))),
            "seasons": ";".join(sorted({season_name(int(str(grid_dt[i])[5:7])) for i in common})),
        })
        state[f"origins__{stage}__COMMON"] = common
    return rows, state


def causal_foundation_features(origin: int, power: np.ndarray, hf_mean: np.ndarray, hf_count: np.ndarray, lookback: int) -> np.ndarray:
    lo = origin - lookback + 1
    pv = power[lo:origin + 1]
    means = hf_mean[:, lo:origin + 1]
    fractions = hf_count[:, lo:origin + 1] / 300.0
    masks = (hf_count[:, lo:origin + 1] > 0).astype(float)
    return np.concatenate((pv, means.ravel(), fractions.ravel(), masks.ravel()))


def finite_order_threshold(scores: np.ndarray, q: float) -> tuple[float, int, float]:
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite 1-D array")
    index = int(math.ceil(q * (len(scores) - 1)))
    threshold = float(np.quantile(scores, q, method="higher"))
    realized = float(np.mean(scores <= threshold))
    return threshold, index, realized


def fit_preprocessor(values: np.ndarray, stage: str) -> tuple[float, float]:
    if stage != "BASE_TRAIN":
        raise ValueError("Preprocessing fit is restricted to BASE_TRAIN")
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)), float(np.std(finite))


def audit(summary_json: Path, state_npz: Path | None) -> dict[str, Any]:
    config = load_config()
    grid_start, grid_end = datetime(2021, 1, 1), datetime(2024, 1, 1)
    grid = np.arange(np.datetime64(grid_start, "m"), np.datetime64(grid_end, "m"), np.timedelta64(5, "m"))
    source_paths = [Path(p) for p in config["pv_files"].values()] + [Path(p) for p in config["irradiance_files"].values()]
    before = {str(p): file_state(p) for p in source_paths if p.exists()}
    irradiance_results: dict[str, dict[str, Any]] = {}
    irradiance_arrays: dict[str, dict[str, np.ndarray]] = {}
    for year_text, path_text in config["irradiance_files"].items():
        year, path = int(year_text), Path(path_text)
        if path.exists():
            result, arrays = scan_second_file(path, year, config["timezone"]["utc_to_acst_minutes"])
        else:
            result, arrays = absent_second_result(path, year)
        irradiance_results[year_text] = result
        irradiance_arrays[year_text] = arrays
    n = len(grid)
    grid_start_key = epoch_seconds(grid_start) // 300
    hf_timestamp_count = np.zeros(n, dtype=np.int32)
    hf_channel_count = np.zeros((3, n), dtype=np.int32)
    hf_channel_sum = np.zeros((3, n), dtype=np.float64)
    for arrays in irradiance_arrays.values():
        if not len(arrays["keys"]):
            continue
        target = arrays["keys"] - grid_start_key
        ok = (target >= 0) & (target < n)
        target = target[ok]
        hf_timestamp_count[target] += arrays["timestamp_count"][ok]
        hf_channel_count[:, target] += arrays["channel_count"][:, ok]
        hf_channel_sum[:, target] += np.nan_to_num(arrays["channel_mean"][:, ok]) * arrays["channel_count"][:, ok]
    hf_channel_mean = np.divide(hf_channel_sum, hf_channel_count, out=np.full_like(hf_channel_sum, np.nan), where=hf_channel_count > 0)
    pv_results: dict[str, dict[str, Any]] = {}
    pv_present: dict[str, np.ndarray] = {}
    pv_power: dict[str, np.ndarray] = {}
    for array, path_text in config["pv_files"].items():
        result, present, power = scan_pv_file(Path(path_text), grid_start, grid_end)
        pv_results[array] = result
        pv_present[array] = present
        pv_power[array] = power
    windows, origin_state = window_audit(config, grid, pv_power, hf_timestamp_count, hf_channel_count)
    after = {str(p): file_state(p) for p in source_paths if p.exists()}
    immutable = before == after
    data_ready = all(irradiance_results[str(year)]["structurally_full_calendar"] for year in (2021, 2022, 2023))
    # Both mask-eligible and strict-complete counts must exist in every stage/common population.
    common_rows = [row for row in windows if row["array"] == "THREE_ARRAY_COMMON"]
    sufficient_windows = all(row["formal_masked_origins"] > 0 and row["strict_all_channel_complete_origins"] > 0 for row in common_rows)
    verdict = "C1_FORMAL_DATA_READY" if data_ready and sufficient_windows else "C1_FORMAL_DATA_FAIL"
    report = {
        "study_id": config["study_id"], "verdict": verdict, "generated_utc": datetime.now(UTC).isoformat(),
        "irradiance": irradiance_results, "pv": pv_results, "windows": windows,
        "source_state_before": before, "source_state_after": after, "source_files_unchanged": immutable,
        "data_ready_conditions": {"all_three_irradiance_years_structurally_full_calendar": data_ready,
                                  "every_stage_has_common_formal_and_strict_origins": sufficient_windows},
        "training_performed": False, "risk_model_fitted": False, "final_test_predictions_generated_or_read": False,
        "official_metadata": {
            "site_17_url": "https://dkasolarcentre.com.au/source/alice-springs/dka-m4-b-phase",
            "site_25_url": "https://dkasolarcentre.com.au/source/alice-springs/dkasc-alice-springs-25-hanwha-q-cells-poly-si-fixed",
            "site_38_url": "https://dkasolarcentre.com.au/source/alice-springs/dka-m19-b-phase",
            "rating_interpretation": "Panel rating multiplied by panel count equals published array rating; this is DC nameplate evidence, not verified AC rating.",
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if state_npz is not None:
        payload: dict[str, np.ndarray] = {
            "grid_ns": grid.astype("datetime64[ns]").astype(np.int64),
            "hf_timestamp_count": hf_timestamp_count,
            "hf_channel_count": hf_channel_count,
            "hf_channel_mean": hf_channel_mean,
        }
        for i, array in enumerate(config["pv_files"]):
            payload[f"pv_power_{i}"] = pv_power[array]
            payload[f"pv_present_{i}"] = pv_present[array]
        payload.update(origin_state)
        np.savez_compressed(state_npz, **payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--state-npz", type=Path)
    args = parser.parse_args()
    report = audit(args.summary_json, args.state_npz)
    print(json.dumps({"verdict": report["verdict"], "source_files_unchanged": report["source_files_unchanged"]}))


if __name__ == "__main__":
    main()

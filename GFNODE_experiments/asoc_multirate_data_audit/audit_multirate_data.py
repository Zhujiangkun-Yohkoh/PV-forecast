"""Read-only, source-backed audit of DKASC PV and multirate weather CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW_ROOT = ROOT.parent / "PV_improve_v1" / "原始Dataset"
PV_DIR = RAW_ROOT / "5min pv active power data"
WEATHER_DIR = RAW_ROOT / "高分辨率气象数据集"
INVENTORY = HERE / "DATA_FILE_INVENTORY.csv"
SUMMARY = HERE / "MULTIRATE_ALIGNMENT_SUMMARY.csv"
START = datetime(2021, 1, 1)
END_EXCLUSIVE = datetime(2024, 1, 1)
PV_NAMES = {"17": "Sanyo", "23": "Calyxo", "25": "Hanwha", "38": "Q CELLS"}

INVENTORY_FIELDS = [
    "relative_path", "size_bytes", "file_type", "column_names", "physical_lines",
    "parseable_records", "first_timestamp", "last_timestamp", "timestamp_format",
    "timezone_in_file", "utc_local_offset_distribution", "interval_distribution", "duplicate_timestamps", "time_inversions",
    "empty_timestamps", "missing_counts", "missing_ratios", "contains_pv_power",
    "contains_irradiance", "contains_weather", "contains_system_availability",
    "duplicate_headers", "column_count_anomalies", "quote_anomalies", "glued_lines",
    "truncated_records", "unrecoverable_records", "recovered_complete_records",
    "malformed_time_start", "malformed_time_end", "issue_time_ranges", "fully_parseable", "notes",
]
SUMMARY_FIELDS = ["section", "array", "year", "scope", "metric", "value", "unit", "notes"]


def j(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def parse_timestamp(text: str, style: str) -> datetime | None:
    text = text.strip().strip('"')
    try:
        if style == "pv" and len(text) >= 19:
            return datetime(int(text[0:4]), int(text[5:7]), int(text[8:10]), int(text[11:13]), int(text[14:16]), int(text[17:19]))
        if style == "dmy_sec" and len(text) >= 19:
            return datetime(int(text[6:10]), int(text[3:5]), int(text[0:2]), int(text[11:13]), int(text[14:16]), int(text[17:19]))
        if style == "dmy_min" and len(text) >= 16:
            return datetime(int(text[6:10]), int(text[3:5]), int(text[0:2]), int(text[11:13]), int(text[14:16]))
    except (ValueError, IndexError):
        return None
    return None


def timestamp_starts(line: str, style: str) -> list[int]:
    import re
    pattern = r'"?\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' if style == "pv" else r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}(?::\d{2})?'
    return [match.start() for match in re.finditer(pattern, line)]


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def five_minute_endpoint(ts: datetime) -> datetime:
    base = ts.replace(second=0, microsecond=0) - timedelta(minutes=ts.minute % 5)
    return base if ts == base else base + timedelta(minutes=5)


@dataclass
class IrrAggregate:
    rows: int = 0
    valid: int = 0
    total: float = 0.0
    total2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    first: float = math.nan
    last: float = math.nan
    max_abs_delta: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    sxy: float = 0.0
    max_source_time: datetime | None = None

    def add(self, ts: datetime, value: float, endpoint: datetime) -> None:
        self.rows += 1
        self.max_source_time = ts if self.max_source_time is None or ts > self.max_source_time else self.max_source_time
        if not math.isfinite(value):
            return
        x = (ts - (endpoint - timedelta(minutes=5))).total_seconds()
        if self.valid == 0:
            self.first = value
        else:
            self.max_abs_delta = max(self.max_abs_delta, abs(value - self.last))
        self.last = value
        self.valid += 1
        self.total += value
        self.total2 += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.sx += x
        self.sy += value
        self.sxx += x * x
        self.sxy += x * value

    def features(self) -> dict[str, float]:
        if not self.valid:
            return {key: math.nan for key in FEATURE_NAMES} | {"valid_count": 0.0}
        mean = self.total / self.valid
        variance = max(0.0, self.total2 / self.valid - mean * mean)
        denom = self.valid * self.sxx - self.sx * self.sx
        slope = (self.valid * self.sxy - self.sx * self.sy) / denom if self.valid > 1 and denom else math.nan
        return {
            "mean": mean, "std": math.sqrt(variance), "minimum": self.minimum,
            "maximum": self.maximum, "range": self.maximum - self.minimum,
            "first_last_change": self.last - self.first, "max_abs_first_difference": self.max_abs_delta,
            "slope": slope, "coefficient_of_variation": math.sqrt(variance) / abs(mean) if abs(mean) > 1e-12 else math.nan,
            "valid_count": float(self.valid),
        }


FEATURE_NAMES = ["mean", "std", "minimum", "maximum", "range", "first_last_change", "max_abs_first_difference", "slope", "coefficient_of_variation"]


def classify_headers(columns: list[str]) -> tuple[bool, bool, bool, bool]:
    lower = " ".join(columns).lower()
    return ("active_power" in lower or "active power" in lower,
            "irradiance" in lower or "radiation" in lower,
            any(x in lower for x in ("temperature", "humidity", "wind", "rain")),
            "system availability" in lower or "system_availability" in lower)


def scan_csv(path: Path, pv_store: dict[str, dict[datetime, tuple[float, bool]]], weather_bins: dict[datetime, IrrAggregate]) -> dict:
    is_pv = path.parent == PV_DIR
    is_1sec = "1sec" in path.name.lower()
    style = "pv" if is_pv else ("dmy_sec" if is_1sec else "dmy_min")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header_line = handle.readline().rstrip("\r\n")
        columns = next(csv.reader([header_line]))
        width = len(columns)
        missing = [0] * width
        physical = 1
        parseable = duplicate_headers = col_bad = quote_bad = glued = truncated = unrecoverable = recovered = empty_ts = 0
        timestamps: set[datetime] = set()
        high_seen = bytearray(int((END_EXCLUSIVE - START).total_seconds())) if is_1sec else None
        duplicates = inversions = 0
        first_ts = last_ts = previous_ts = None
        intervals: Counter[int] = Counter()
        malformed_times: list[datetime] = []
        issue_times: dict[str, list[datetime]] = {key: [] for key in ("glued", "truncated", "unrecoverable", "quote", "column_count")}
        offsets: Counter[int] = Counter()
        array = next((name for site, name in PV_NAMES.items() if path.name.startswith(site + " ")), "UNKNOWN")
        store = pv_store.setdefault(array, {}) if is_pv else None
        power_idx = columns.index("Active_Power") if "Active_Power" in columns else None
        local_idx = next((i for i, c in enumerate(columns) if c.startswith("Timestamp_Local")), 0)
        irradiance_idx = [i for i, c in enumerate(columns) if "Irradiance_" in c and "W/m" in c]

        def accept(row: list[str], recovered_row: bool = False) -> bool:
            nonlocal parseable, recovered, empty_ts, first_ts, last_ts, previous_ts, duplicates, inversions
            if len(row) != width:
                return False
            ts_text = row[0 if is_pv else local_idx]
            if not ts_text.strip():
                empty_ts += 1
                return False
            ts = parse_timestamp(ts_text, style)
            if ts is None:
                return False
            parseable += 1
            recovered += int(recovered_row)
            if high_seen is not None and START <= ts < END_EXCLUSIVE:
                second_index = int((ts - START).total_seconds())
                duplicates += int(bool(high_seen[second_index]))
                high_seen[second_index] = 1
            else:
                if ts in timestamps:
                    duplicates += 1
                timestamps.add(ts)
            if previous_ts is not None:
                delta = int((ts - previous_ts).total_seconds())
                intervals[delta] += 1
                if delta < 0:
                    inversions += 1
            previous_ts = ts
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts
            if not is_pv:
                utc_ts = parse_timestamp(row[0], style)
                local_ts = parse_timestamp(row[local_idx], style)
                if utc_ts and local_ts:
                    offsets[int((local_ts - utc_ts).total_seconds())] += 1
            for i, value in enumerate(row):
                if not value.strip():
                    missing[i] += 1
            if is_pv and START <= ts < END_EXCLUSIVE:
                power = safe_float(row[power_idx]) if power_idx is not None else math.nan
                store[ts] = (power, False)
            elif is_1sec and START <= ts < END_EXCLUSIVE:
                values = [safe_float(row[i]) for i in irradiance_idx]
                finite = [v for v in values if math.isfinite(v)]
                value = sum(finite) / len(finite) if finite else math.nan
                endpoint = five_minute_endpoint(ts)
                weather_bins.setdefault(endpoint, IrrAggregate()).add(ts, value, endpoint)
            return True

        for raw in handle:
            physical += 1
            line = raw.rstrip("\r\n")
            if line == header_line:
                duplicate_headers += 1
                continue
            starts = timestamp_starts(line, style)
            if not is_pv:
                import re
                date_starts = [m.start() for m in re.finditer(r"\d{2}/\d{2}/\d{4}", line)]
            else:
                date_starts = []
            if line.count('"') % 2:
                quote_bad += 1
                issue_times["quote"].extend(parse_timestamp(line[p:].lstrip('"')[:19], style) for p in starts)
            try:
                whole_row = next(csv.reader([line]))
            except csv.Error:
                whole_row = []
            if len(whole_row) == width and not (not is_pv and len(date_starts) > 2):
                if not accept(whole_row):
                    unrecoverable += 1
                continue
            expected_timestamp_fields = 1 if is_pv else 2
            if not is_pv:
                if len(date_starts) > expected_timestamp_fields:
                    glued += 1
                    recovered_suffix = False
                    # Date-only starts include a complete second UTC record even when the
                    # truncated prefix greedily resembles a full timestamp.
                    for candidate in sorted(set(date_starts[1:] + starts)):
                        if candidate == 0:
                            continue
                        try:
                            suffix_row = next(csv.reader([line[candidate:]]))
                        except csv.Error:
                            suffix_row = []
                        if len(suffix_row) == width and accept(suffix_row, recovered_row=True):
                            recovered_suffix = True
                            recovered_ts = parse_timestamp(suffix_row[local_idx], style)
                            if recovered_ts:
                                issue_times["glued"].append(recovered_ts)
                            break
                    truncated += 1
                    # The prefix timestamp itself may be cut through by the next date; do not guess its minute.
                    issue_times["truncated"].append(None)
                    if not recovered_suffix:
                        unrecoverable += 1
                    continue
            if len(starts) > expected_timestamp_fields:
                glued += 1
                record_starts = starts[::expected_timestamp_fields]
                boundaries = record_starts + [len(line)]
                for n in range(len(record_starts)):
                    segment = line[boundaries[n]:boundaries[n + 1]]
                    try:
                        row = next(csv.reader([segment]))
                    except csv.Error:
                        row = []
                    if len(row) == width and accept(row, recovered_row=n > 0):
                        recovered_ts = parse_timestamp(row[0 if is_pv else local_idx], style)
                        if recovered_ts:
                            issue_times["glued"].append(recovered_ts)
                        continue
                    truncated += 1
                    possible = parse_timestamp(segment.lstrip('"')[:19 if style != "dmy_min" else 16], style)
                    if possible:
                        malformed_times.append(possible)
                        issue_times["truncated"].append(possible)
                        if is_pv and START <= possible < END_EXCLUSIVE:
                            store[possible] = (math.nan, True)
                continue
            if len(whole_row) != width:
                col_bad += 1
                issue_times["column_count"].extend(parse_timestamp(line[p:].lstrip('"')[:19 if style != "dmy_min" else 16], style) for p in starts)
                truncated += int(bool(starts) or line.rstrip().endswith(",") or line.lstrip('"')[:4].isdigit())
                unrecoverable += 1
                issue_times["unrecoverable"].extend(parse_timestamp(line[p:].lstrip('"')[:19 if style != "dmy_min" else 16], style) for p in starts)
                if starts:
                    possible = parse_timestamp(line.lstrip('"')[:19 if style != "dmy_min" else 16], style)
                    if possible:
                        malformed_times.append(possible)
                        if is_pv and START <= possible < END_EXCLUSIVE:
                            store[possible] = (math.nan, True)
                continue
            # A width-correct row was already handled above; no silent skip path exists here.

    pv_power, irradiance, meteorology, availability = classify_headers(columns)
    normal_denominator = max(parseable, 1)
    fully = glued == 0 and truncated == 0 and unrecoverable == 0 and duplicate_headers == 0 and col_bad == 0 and quote_bad == 0
    return {
        "relative_path": str(path.relative_to(RAW_ROOT)), "size_bytes": path.stat().st_size,
        "file_type": path.suffix.lower().lstrip("."), "column_names": j(columns),
        "physical_lines": physical, "parseable_records": parseable,
        "first_timestamp": str(first_ts) if first_ts else "UNKNOWN", "last_timestamp": str(last_ts) if last_ts else "UNKNOWN",
        "timestamp_format": {"pv": "YYYY-MM-DD hh:mm:ss", "dmy_sec": "DD/MM/YYYY hh:mm:ss", "dmy_min": "DD/MM/YYYY hh:mm"}[style],
        "timezone_in_file": "UTC and Local columns; offset is measured from file values" if not is_pv else "NO",
        "utc_local_offset_distribution": j({str(timedelta(seconds=k)): v for k, v in offsets.most_common()}),
        "interval_distribution": j({str(timedelta(seconds=k)): v for k, v in intervals.most_common(20)}),
        "duplicate_timestamps": duplicates, "time_inversions": inversions, "empty_timestamps": empty_ts,
        "missing_counts": j(dict(zip(columns, missing))),
        "missing_ratios": j({c: round(n / normal_denominator, 8) for c, n in zip(columns, missing)}),
        "contains_pv_power": "YES" if pv_power else "NO", "contains_irradiance": "YES" if irradiance else "NO",
        "contains_weather": "YES" if meteorology else "NO", "contains_system_availability": "YES" if availability else "NO",
        "duplicate_headers": duplicate_headers, "column_count_anomalies": col_bad, "quote_anomalies": quote_bad,
        "glued_lines": glued, "truncated_records": truncated, "unrecoverable_records": unrecoverable,
        "recovered_complete_records": recovered,
        "malformed_time_start": str(min(malformed_times)) if malformed_times else "NONE",
        "malformed_time_end": str(max(malformed_times)) if malformed_times else "NONE",
        "issue_time_ranges": j({key: {"start": str(min(clean)), "end": str(max(clean)), "count_with_timestamp": len(clean)}
                                for key, values in issue_times.items() if (clean := [v for v in values if v is not None])}),
        "fully_parseable": "YES" if fully else "NO",
        "notes": "No bad line was silently skipped; complete suffix records on glued lines were recovered in memory only.",
    }


def add(rows: list[dict], section: str, array: str, year, scope: str, metric: str, value, unit: str = "", notes: str = "") -> None:
    rows.append(dict(section=section, array=array, year=year, scope=scope, metric=metric, value=value, unit=unit, notes=notes))


def longest_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def correlations(summary: list[dict], array: str, pv: dict, bins: dict, times: list[datetime]) -> None:
    records = []
    for ts in times:
        agg = bins.get(ts)
        if not agg or not agg.valid or ts not in pv or ts + timedelta(minutes=5) not in pv or ts + timedelta(hours=1) not in pv:
            continue
        p0, m0 = pv[ts]; p1, m1 = pv[ts + timedelta(minutes=5)]; p12, m12 = pv[ts + timedelta(hours=1)]
        if any((m0, m1, m12)) or not all(map(math.isfinite, (p0, p1, p12))):
            continue
        records.append(agg.features() | {"next_ramp": p1 - p0, "hour_ramp": p12 - p0, "abs_hour_ramp": abs(p12 - p0)})
    frame = pd.DataFrame(records)
    add(summary, "descriptive", array, 2023, "past_only", "usable_feature_rows", len(frame), "intervals")
    if len(frame) < 10:
        return
    for feature in FEATURE_NAMES + ["valid_count"]:
        for target in ["next_ramp", "hour_ramp", "abs_hour_ramp"]:
            pair = frame[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            pearson = pair[feature].corr(pair[target], method="pearson") if len(pair) > 2 else math.nan
            spearman = pair[feature].corr(pair[target], method="spearman") if len(pair) > 2 else math.nan
            add(summary, "correlation", array, 2023, "past_only", f"{feature}_vs_{target}_pearson", pearson, "r", "Feature interval is (t-5min,t]; target begins at t.")
            add(summary, "correlation", array, 2023, "past_only", f"{feature}_vs_{target}_spearman", spearman, "rho", "Feature interval is (t-5min,t]; target begins at t.")
    finite = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean", "abs_hour_ramp"])
    for feature in ["std", "range", "max_abs_first_difference", "slope", "coefficient_of_variation"]:
        part = finite.dropna(subset=[feature])
        if len(part) > 3:
            x = np.column_stack([np.ones(len(part)), part["mean"].to_numpy()])
            rf = part[feature].to_numpy() - x @ np.linalg.lstsq(x, part[feature].to_numpy(), rcond=None)[0]
            rt = part["abs_hour_ramp"].to_numpy() - x @ np.linalg.lstsq(x, part["abs_hour_ramp"].to_numpy(), rcond=None)[0]
            add(summary, "incremental", array, 2023, "past_only", f"{feature}_partial_corr_abs_hour_ramp_given_mean", np.corrcoef(rf, rt)[0, 1], "r")
    q1, q2 = finite["abs_hour_ramp"].quantile([1 / 3, 2 / 3])
    groups = pd.cut(finite["abs_hour_ramp"], [-np.inf, q1, q2, np.inf], labels=["stable", "moderate", "ramp"], include_lowest=True)
    for group in ["stable", "moderate", "ramp"]:
        part = finite[groups == group]
        for feature in ["mean", "std", "range", "max_abs_first_difference", "slope", "coefficient_of_variation"]:
            add(summary, "ramp_group", array, 2023, group, f"{feature}_mean", part[feature].replace([np.inf, -np.inf], np.nan).mean(), "")


def build_alignment(pv_store: dict[str, dict], bins: dict[datetime, IrrAggregate]) -> list[dict]:
    rows: list[dict] = []
    counts = Counter(agg.rows for ts, agg in bins.items() if START <= ts < END_EXCLUSIVE and agg.rows > 0)
    expected = counts.most_common(1)[0][0] if counts else 0
    add(rows, "weather", "ALL", "ALL", "past_only", "dominant_records_per_5min", expected, "records")
    add(rows, "weather", "ALL", "ALL", "past_only", "observed_5min_bins_2021_2023", len(bins), "intervals")
    add(rows, "weather", "ALL", "ALL", "past_only", "records_per_bin_distribution", j(dict(counts.most_common(30))), "records")
    weather_times = sorted(ts for ts in bins if START <= ts < END_EXCLUSIVE)
    weather_start = weather_times[0] if weather_times else None; weather_end = weather_times[-1] if weather_times else None
    for array, pv in sorted(pv_store.items()):
        pv_times = sorted(ts for ts in pv if START <= ts < END_EXCLUSIVE)
        for year in [2021, 2022, 2023]:
            year_start = datetime(year, 1, 1); year_end = datetime(year + 1, 1, 1)
            expected_year = int((year_end - year_start).total_seconds() // 300)
            pv_year = [ts for ts in pv_times if ts.year == year]
            weather_year = [ts for ts in weather_times if ts.year == year]
            add(rows, "source_coverage", array, year, "calendar_year", "expected_5min_intervals", expected_year, "intervals")
            add(rows, "source_coverage", array, year, "calendar_year", "pv_rows_present", len(pv_year), "intervals")
            add(rows, "source_coverage", array, year, "calendar_year", "pv_values_valid", sum(math.isfinite(pv[t][0]) and not pv[t][1] for t in pv_year), "intervals")
            add(rows, "source_coverage", array, year, "calendar_year", "weather_bins_present", len(weather_year), "intervals")
            add(rows, "source_coverage", array, year, "calendar_year", "weather_complete_bins", sum(bins[t].rows == expected and bins[t].valid == expected for t in weather_year), "intervals")
        if not pv_times or not weather_times:
            add(rows, "alignment", array, "ALL", "common", "status", "NO_COMMON_DATA")
            continue
        start = max(START, pv_times[0], weather_start); end = min(END_EXCLUSIVE - timedelta(minutes=5), pv_times[-1], weather_end)
        times = list(pd.date_range(start, end, freq="5min").to_pydatetime()) if start <= end else []
        add(rows, "alignment", array, "ALL", "common", "common_start", start)
        add(rows, "alignment", array, "ALL", "common", "common_end", end)
        for year in [2021, 2022, 2023]:
            yt = [ts for ts in times if ts.year == year]
            if not yt:
                for metric in ["expected_intervals", "pv_rows_present", "weather_rows_present", "complete_intervals", "partial_intervals", "no_weather_intervals"]:
                    add(rows, "coverage", array, year, "full_timeline", metric, 0, "intervals")
                continue
            present_pv = [ts in pv for ts in yt]
            pv_valid = [ts in pv and math.isfinite(pv[ts][0]) and not pv[ts][1] for ts in yt]
            malformed = [ts in pv and pv[ts][1] for ts in yt]
            wpresent = [ts in bins and bins[ts].rows > 0 for ts in yt]
            wvalid = [ts in bins and bins[ts].valid > 0 for ts in yt]
            complete = [ts in bins and bins[ts].rows == expected and bins[ts].valid == expected for ts in yt]
            partial = [p and not c for p, c in zip(wpresent, complete)]
            no_weather = [not p for p in wpresent]
            pv_valid_weather_incomplete = [p and not c for p, c in zip(pv_valid, complete)]
            weather_valid_pv_missing = [w and not p for w, p in zip(wvalid, pv_valid)]
            metrics = {
                "expected_intervals": len(yt), "pv_rows_present": sum(present_pv), "pv_values_valid": sum(pv_valid),
                "malformed_source_records": sum(malformed), "weather_rows_present": sum(wpresent),
                "weather_values_valid": sum(wvalid), "complete_intervals": sum(complete), "partial_intervals": sum(partial),
                "no_weather_intervals": sum(no_weather), "pv_valid_weather_incomplete": sum(pv_valid_weather_incomplete),
                "weather_valid_pv_missing": sum(weather_valid_pv_missing),
                "longest_complete_segment": longest_run(complete) * 5,
                "longest_weather_gap": longest_run(no_weather) * 5,
            }
            for metric, value in metrics.items():
                add(rows, "coverage", array, year, "full_timeline", metric, value, "minutes" if metric.startswith("longest") else "intervals")
            for label, selector in [("daylight", [ts in bins and bins[ts].valid and bins[ts].features()["mean"] > 20 for ts in yt]),
                                    ("night", [ts in bins and bins[ts].valid and bins[ts].features()["mean"] <= 20 for ts in yt])]:
                denom = sum(selector)
                add(rows, "coverage", array, year, label, "classified_intervals", denom, "intervals", "Daylight threshold: past-interval mean irradiance >20 W/m2; missing-weather intervals unclassified.")
                add(rows, "coverage", array, year, label, "pv_value_valid_ratio", sum(s and p for s, p in zip(selector, pv_valid)) / denom if denom else math.nan, "ratio")
                add(rows, "coverage", array, year, label, "weather_complete_ratio", sum(s and c for s, c in zip(selector, complete)) / denom if denom else math.nan, "ratio")
        correlations(rows, array, pv, bins, times)
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    if path.parent.resolve() != HERE.resolve():
        raise RuntimeError("Output escaped audit directory")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def audit() -> None:
    files = sorted([p for root in (PV_DIR, WEATHER_DIR) for p in root.rglob("*") if p.is_file()])
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    pv_store: dict[str, dict[datetime, tuple[float, bool]]] = {}
    weather_bins: dict[datetime, IrrAggregate] = {}
    inventory = [scan_csv(path, pv_store, weather_bins) for path in files]
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    if before != after:
        raise RuntimeError("Raw source size or modification time changed during audit")
    alignment = build_alignment(pv_store, weather_bins)
    write_csv(INVENTORY, INVENTORY_FIELDS, inventory)
    write_csv(SUMMARY, SUMMARY_FIELDS, alignment)
    print(f"Audited {len(files)} files; wrote {len(alignment)} compact summary rows")


def self_test() -> None:
    assert INVENTORY.parent.resolve() == HERE.resolve() == SUMMARY.parent.resolve()
    assert parse_timestamp("not a timestamp", "pv") is None
    sample = '"2025-08-23 05:30:00",1,2"2025-08-23 05:35:00",1,2,3'
    starts = timestamp_starts(sample, "pv")
    segments = [sample[starts[i]: starts[i + 1] if i + 1 < len(starts) else len(sample)] for i in range(len(starts))]
    assert len(next(csv.reader([segments[0]]))) != 4
    assert len(next(csv.reader([segments[1]]))) == 4
    endpoint = datetime(2023, 1, 1, 0, 5); agg = IrrAggregate()
    agg.add(datetime(2023, 1, 1, 0, 4, 59), 1.0, endpoint)
    assert agg.max_source_time <= endpoint
    inv = pd.read_csv(INVENTORY); summary = pd.read_csv(SUMMARY)
    assert len(inv) and len(summary)
    assert inv["relative_path"].map(lambda p: (RAW_ROOT / p).is_file()).all()
    assert (inv["size_bytes"].astype(int) == inv["relative_path"].map(lambda p: (RAW_ROOT / p).stat().st_size)).all()
    assert set(summary.columns) == set(SUMMARY_FIELDS)
    common = summary[(summary["section"] == "coverage") & (summary["scope"] == "full_timeline")]
    for (array, year), group in common.groupby(["array", "year"]):
        values = dict(zip(group["metric"], pd.to_numeric(group["value"], errors="coerce")))
        if values.get("expected_intervals", 0):
            assert values["complete_intervals"] + values["partial_intervals"] + values["no_weather_intervals"] == values["expected_intervals"]
    assert (pd.to_numeric(inv["parseable_records"], errors="raise") >= 0).all()
    assert not summary[(summary["section"].isin(["correlation", "incremental", "descriptive"])) & (summary["scope"] != "past_only")].shape[0]
    print(f"PASS: {len(inv)} real files; outputs restricted; malformed recovery and past-only feature assertions passed")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    self_test() if args.self_test else audit()


if __name__ == "__main__":
    main()

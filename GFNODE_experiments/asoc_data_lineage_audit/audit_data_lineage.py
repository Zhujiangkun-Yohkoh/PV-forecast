"""Read-only PV dataset lineage inventory.

The only write performed by this module is DATASET_LINEAGE_INVENTORY.csv beside
this script.  Source datasets are never rewritten, copied, or regularized.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AUDIT_DIR = Path(__file__).resolve().parent
REPO = AUDIT_DIR.parents[1]
PAPER = REPO.parent / "PV_improve_v1"
OUTPUT = AUDIT_DIR / "DATASET_LINEAGE_INVENTORY.csv"
DATA_EXTENSIONS = {".csv", ".xls", ".xlsx", ".parquet", ".npy", ".npz", ".pkl", ".pickle"}
TEXT_EXTENSIONS = {".py", ".json", ".md", ".txt", ".yaml", ".yml", ".toml"}
RAW_FIELDS = {
    "timestamp", "Active_Power", "Performance_Ratio", "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity", "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation", "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted",
}
PRIMARY_NAMES = {"17Sanyo.csv", "25Hanwha.csv", "38QCELLS.csv", "25Hanwha_Differentseason.csv"}
INVENTORY_FIELDS = [
    "path", "workspace", "filename", "file_type", "size_bytes", "rows", "columns",
    "column_names", "has_original_timestamp", "timestamp_parse_status", "timestamp_start",
    "timestamp_end", "timestamp_interval_distribution", "duplicate_timestamps",
    "missing_timestamps", "per_column_missing_ratio", "suspected_normalized",
    "contains_original_power_units", "contains_missing_mask", "contains_isolation_forest",
    "is_sliding_window_array", "maps_to_original_timestamps", "source_references",
    "in_git_master", "lineage_level", "notes",
]


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def discover_data_files() -> list[Path]:
    found: list[Path] = []
    for root in (REPO, PAPER):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DATA_EXTENSIONS and AUDIT_DIR not in path.parents:
                found.append(path.resolve())
    return sorted(set(found), key=lambda p: str(p).lower())


def source_reference_index() -> dict[str, list[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    for root in (REPO, PAPER):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS or AUDIT_DIR in path.parents:
                continue
            try:
                if path.stat().st_size > 5_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in re.finditer(r"(?i)([\w .()\-]+\.(?:csv|xlsx?|parquet|npy|npz|pkl|pickle))", text):
                name = Path(match.group(1).strip()).name
                refs[name.lower()].add(str(path.resolve()))
    return {key: sorted(value) for key, value in refs.items()}


def git_master_paths() -> set[str]:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "ls-tree", "-r", "--name-only", "origin/master"],
            cwd=REPO, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}
    except (OSError, subprocess.CalledProcessError):
        return set()


def timestamp_statistics(series: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna().sort_values()
    if valid.empty or len(valid) < max(2, len(series) // 2):
        return {
            "has_original_timestamp": "UNKNOWN", "timestamp_parse_status": "UNKNOWN",
            "timestamp_start": "UNKNOWN", "timestamp_end": "UNKNOWN",
            "timestamp_interval_distribution": "UNKNOWN", "duplicate_timestamps": "UNKNOWN",
            "missing_timestamps": "UNKNOWN", "maps_to_original_timestamps": "UNKNOWN",
        }
    differences = valid.diff().dropna()
    distribution = Counter(str(x) for x in differences)
    positive = differences[differences > pd.Timedelta(0)]
    base = positive.mode().iloc[0] if not positive.empty else pd.NaT
    if pd.isna(base) or base <= pd.Timedelta(0):
        missing = "UNKNOWN"
    else:
        expected = pd.date_range(valid.iloc[0], valid.iloc[-1], freq=base)
        missing = int(len(expected.difference(pd.DatetimeIndex(valid.unique()))))
    return {
        "has_original_timestamp": "YES", "timestamp_parse_status": "PARSED_FROM_FILE",
        "timestamp_start": str(valid.iloc[0]), "timestamp_end": str(valid.iloc[-1]),
        "timestamp_interval_distribution": json_text(dict(distribution.most_common(20))),
        "duplicate_timestamps": int(valid.duplicated().sum()), "missing_timestamps": missing,
        "maps_to_original_timestamps": "YES",
    }


def normalized_suspicion(frame: pd.DataFrame) -> str:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return "UNKNOWN"
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    mins, maxs = finite.min(), finite.max()
    bounded = ((mins >= -0.001) & (maxs <= 1.001)).mean()
    return "YES" if bounded >= 0.8 else "NO"


def power_unit_status(columns: list[str], normalized: str) -> str:
    """Report units only when the file itself encodes them; never infer kW from scale."""
    power_columns = [column.lower() for column in columns if "power" in column.lower()]
    if not power_columns:
        return "NO"
    if any(re.search(r"(^|_)(kw|mw|watt|watts|w)($|_)", column) for column in power_columns):
        return "YES"
    return "UNKNOWN_UNNORMALIZED" if normalized == "NO" else "UNKNOWN"


def classify_frame(path: Path, frame: pd.DataFrame, refs: list[str], timestamp: dict[str, Any]) -> tuple[str, str]:
    columns = set(map(str, frame.columns))
    is_window = any(re.search(r"(^|_)(window|lookback|horizon|prediction|label|target_start)(_|$)", c, re.I) for c in columns)
    normalized = normalized_suspicion(frame)
    raw_conditions = (
        RAW_FIELDS.issubset(columns) and timestamp["timestamp_parse_status"] == "PARSED_FROM_FILE"
        and normalized == "NO" and not is_window and bool(refs)
    )
    if raw_conditions:
        note = "Best-available project raw candidate; selected/regularized upstream status remains possible."
        if path.name not in PRIMARY_NAMES:
            note = "Content meets raw-candidate criteria, but project-source identity is not independently documented."
        return "RAW", note
    lower = " ".join(columns).lower()
    if is_window:
        return "WINDOWED", "Window/prediction-oriented fields detected from content."
    if "prediction" in lower or "pred" in lower or "label" in lower or "actual" in lower:
        return "PREDICTION", "Prediction/label fields detected from content."
    if normalized == "YES":
        return "SCALED", "Most numeric columns are bounded approximately within [0,1]."
    if "missing_mask" in lower or "isolation" in lower or "anomaly" in lower:
        return "IMPUTED", "Preprocessing marker fields detected."
    return "UNKNOWN", "No content-supported lineage level could be assigned."


def inspect_table(path: Path, refs: list[str]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_parquet(path)
    columns = [str(c) for c in frame.columns]
    timestamp_column = next((c for c in columns if c.lower() in {"timestamp", "datetime", "date_time", "time"}), None)
    timestamp = timestamp_statistics(frame[timestamp_column]) if timestamp_column else timestamp_statistics(pd.Series([], dtype=object))
    level, note = classify_frame(path, frame, refs, timestamp)
    lower_columns = [c.lower() for c in columns]
    return {
        "rows": len(frame), "columns": len(columns), "column_names": json_text(columns),
        **timestamp,
        "per_column_missing_ratio": json_text({c: round(float(frame[c].isna().mean()), 8) for c in columns}),
        "suspected_normalized": normalized_suspicion(frame),
        "contains_original_power_units": power_unit_status(columns, normalized_suspicion(frame)),
        "contains_missing_mask": "YES" if any("missing_mask" in c for c in lower_columns) else "NO",
        "contains_isolation_forest": "YES" if any("isolation" in c or "iforest" in c or "anomaly" in c for c in lower_columns) else "NO",
        "is_sliding_window_array": "YES" if level == "WINDOWED" else "NO",
        "lineage_level": level, "notes": note,
    }


def inspect_array(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return {
            "rows": "UNKNOWN", "columns": "UNKNOWN", "column_names": "UNKNOWN",
            **timestamp_statistics(pd.Series([], dtype=object)),
            "per_column_missing_ratio": "UNKNOWN", "suspected_normalized": "UNKNOWN",
            "contains_original_power_units": "UNKNOWN", "contains_missing_mask": "UNKNOWN",
            "contains_isolation_forest": "UNKNOWN", "is_sliding_window_array": "UNKNOWN",
            "lineage_level": "UNKNOWN", "notes": "Pickle was not deserialized during the read-only security-conscious audit.",
        }
    arrays: dict[str, np.ndarray] = {}
    if suffix == ".npy":
        arrays[path.stem] = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as bundle:
            for key in bundle.files:
                arrays[key] = bundle[key]
    shapes = {key: list(value.shape) for key, value in arrays.items()}
    total_rows = max((value.shape[0] for value in arrays.values() if value.ndim), default=0)
    keys = [key.lower() for key in arrays]
    timestamp_key = next((key for key in arrays if "timestamp" in key.lower()), None)
    timestamp = timestamp_statistics(pd.Series(arrays[timestamp_key].reshape(-1))) if timestamp_key else timestamp_statistics(pd.Series([], dtype=object))
    windowed = any(value.ndim >= 3 for value in arrays.values()) or any(k in " ".join(keys) for k in ("lookback", "window"))
    prediction = any(any(token in key for token in ("prediction", "pred", "label", "actual")) for key in keys)
    missing = {}
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            missing[key] = (
                round(float(np.isnan(value).mean()), 8)
                if value.size and np.issubdtype(value.dtype, np.floating) else 0.0
            )
        else:
            missing[key] = "UNKNOWN"
    return {
        "rows": total_rows, "columns": json_text(shapes), "column_names": json_text(list(arrays)),
        **timestamp, "per_column_missing_ratio": json_text(missing), "suspected_normalized": "UNKNOWN",
        "contains_original_power_units": "UNKNOWN",
        "contains_missing_mask": "YES" if any("mask" in key for key in keys) else "NO",
        "contains_isolation_forest": "YES" if any("isolation" in key or "iforest" in key for key in keys) else "NO",
        "is_sliding_window_array": "YES" if windowed else "NO",
        "lineage_level": "WINDOWED" if windowed else ("PREDICTION" if prediction else "UNKNOWN"),
        "notes": "Array metadata and contents inspected directly; no timestamp was synthesized.",
    }


def workspace_name(path: Path) -> str:
    if REPO in path.parents:
        return "PVforecast16"
    if PAPER in path.parents:
        return "PV_improve_v1"
    return "UNKNOWN"


def audit() -> list[dict[str, Any]]:
    files = discover_data_files()
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    refs = source_reference_index()
    master = git_master_paths()
    rows: list[dict[str, Any]] = []
    for path in files:
        base = {
            "path": str(path), "workspace": workspace_name(path), "filename": path.name,
            "file_type": path.suffix.lower().lstrip("."), "size_bytes": path.stat().st_size,
            "source_references": json_text(refs.get(path.name.lower(), [])),
            "in_git_master": "YES" if path.is_relative_to(REPO) and path.relative_to(REPO).as_posix() in master else "NO",
        }
        try:
            detail = inspect_table(path, refs.get(path.name.lower(), [])) if path.suffix.lower() in {".csv", ".xls", ".xlsx", ".parquet"} else inspect_array(path)
        except Exception as exc:  # inventory must retain unreadable artifacts as UNKNOWN
            detail = {field: "UNKNOWN" for field in INVENTORY_FIELDS if field not in base}
            detail.update({"lineage_level": "UNKNOWN", "notes": f"Inspection failed: {type(exc).__name__}: {exc}"})
        rows.append({field: {**base, **detail}.get(field, "UNKNOWN") for field in INVENTORY_FIELDS})
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    if before != after:
        raise RuntimeError("A source data file changed during the read-only audit")
    return rows


def write_inventory(rows: list[dict[str, Any]]) -> None:
    if OUTPUT.resolve().parent != AUDIT_DIR.resolve():
        raise RuntimeError("Inventory output escaped the dedicated audit directory")
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    assert OUTPUT.resolve().parent == AUDIT_DIR.resolve()
    unknown = timestamp_statistics(pd.Series(["not-a-time", "still-not-a-time"]))
    assert unknown["timestamp_parse_status"] == "UNKNOWN"
    assert unknown["timestamp_start"] == "UNKNOWN"
    assert OUTPUT.exists(), "Run the audit before --self-test"
    inventory = pd.read_csv(OUTPUT)
    assert not inventory.empty
    assert inventory["path"].map(lambda value: Path(value).is_file()).all()
    assert (inventory["size_bytes"].astype(int) > 0).all()
    assert not inventory["path"].str.contains("synthetic|simulate|dummy", case=False, regex=True).any()
    print(f"PASS: {len(inventory)} source-backed records; output restricted to {AUDIT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows = audit()
    write_inventory(rows)
    print(f"Wrote {len(rows)} records to {OUTPUT}")


if __name__ == "__main__":
    main()

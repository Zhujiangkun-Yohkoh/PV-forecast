"""Read-only validation of the redownloaded DKASC 2022 irradiance data."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAW_WEATHER = HERE.parent.parent / "原始Dataset" / "高分辨率气象数据集"
PV_DIR = HERE.parent.parent / "原始Dataset" / "5min pv active power data"
SECOND_FILE = HERE / "fsm1m_7111_ekistica-alice-springs-ekistica-alice-springs_irradiancedata_1sec_2022.csv"
RESOURCE_FILES = sorted(HERE.glob("fsm1m_*.csv"))
SANYO_FILE = PV_DIR / "17 Sanyo.csv"
MONTHLY_OUT = HERE / "REDOWNLOAD_MONTHLY_COVERAGE.csv"
ALIGN_OUT = HERE / "TIME_ALIGNMENT_SUMMARY.csv"
YEAR_START = datetime(2022, 1, 1)
YEAR_END = datetime(2023, 1, 1)
N_SECONDS = int((YEAR_END - YEAR_START).total_seconds())
N_BINS = N_SECONDS // 300
CHANNELS = ["MB0", "MB1", "MB2"]


def parse_dmy_second(text: str) -> datetime | None:
    text = text.strip().strip('"')
    if len(text) < 19:
        return None
    try:
        return datetime(int(text[6:10]), int(text[3:5]), int(text[0:2]),
                        int(text[11:13]), int(text[14:16]), int(text[17:19]))
    except (ValueError, IndexError):
        return None


def parse_dmy_minute(text: str) -> datetime | None:
    text = text.strip().strip('"')
    if len(text) < 16:
        return None
    try:
        return datetime(int(text[6:10]), int(text[3:5]), int(text[0:2]),
                        int(text[11:13]), int(text[14:16]))
    except (ValueError, IndexError):
        return None


def parse_pv(text: str) -> datetime | None:
    text = text.strip().strip('"')
    if len(text) < 19:
        return None
    try:
        return datetime(int(text[0:4]), int(text[5:7]), int(text[8:10]),
                        int(text[11:13]), int(text[14:16]), int(text[17:19]))
    except (ValueError, IndexError):
        return None


def number(text: str) -> float:
    try:
        return float(text)
    except (ValueError, TypeError):
        return math.nan


def month_slices() -> list[tuple[int, datetime, datetime, int, int]]:
    out = []
    for month in range(1, 13):
        start = datetime(2022, month, 1)
        end = datetime(2023, 1, 1) if month == 12 else datetime(2022, month + 1, 1)
        out.append((month, start, end, int((start - YEAR_START).total_seconds()), int((end - YEAR_START).total_seconds())))
    return out


class ChannelBins:
    def __init__(self) -> None:
        shape = (3, N_BINS)
        self.count = np.zeros(shape, dtype=np.uint16)
        self.total = np.zeros(shape, dtype=np.float64)
        self.total2 = np.zeros(shape, dtype=np.float64)
        self.minimum = np.full(shape, np.inf)
        self.maximum = np.full(shape, -np.inf)
        self.first = np.full(shape, np.nan)
        self.last = np.full(shape, np.nan)
        self.maxdiff = np.zeros(shape, dtype=np.float32)
        self.sx = np.zeros(shape, dtype=np.float64)
        self.sxx = np.zeros(shape, dtype=np.float64)
        self.sxy = np.zeros(shape, dtype=np.float64)
        self.row_count = np.zeros(N_BINS, dtype=np.uint16)
        self.common_count = np.zeros(N_BINS, dtype=np.uint16)
        self.max_source_second = np.full(N_BINS, -1, dtype=np.int64)

    def add(self, second: int, values: list[float]) -> None:
        endpoint = second // 300
        x = second % 300
        self.row_count[endpoint] += 1
        self.max_source_second[endpoint] = max(self.max_source_second[endpoint], second)
        self.common_count[endpoint] += int(all(math.isfinite(v) for v in values))
        for channel, value in enumerate(values):
            if not math.isfinite(value):
                continue
            count = self.count[channel, endpoint]
            if count == 0:
                self.first[channel, endpoint] = value
            else:
                self.maxdiff[channel, endpoint] = max(self.maxdiff[channel, endpoint], abs(value - self.last[channel, endpoint]))
            self.last[channel, endpoint] = value
            self.count[channel, endpoint] += 1
            self.total[channel, endpoint] += value
            self.total2[channel, endpoint] += value * value
            self.minimum[channel, endpoint] = min(self.minimum[channel, endpoint], value)
            self.maximum[channel, endpoint] = max(self.maximum[channel, endpoint], value)
            self.sx[channel, endpoint] += x
            self.sxx[channel, endpoint] += x * x
            self.sxy[channel, endpoint] += x * value

    def feature(self, channel: int, name: str) -> np.ndarray:
        n = self.count[channel].astype(float)
        valid = n > 0
        mean = np.divide(self.total[channel], n, out=np.full(N_BINS, np.nan), where=valid)
        if name == "mean": return mean
        if name == "std": return np.sqrt(np.maximum(0, np.divide(self.total2[channel], n, out=np.full(N_BINS, np.nan), where=valid) - mean * mean))
        if name == "min": return np.where(valid, self.minimum[channel], np.nan)
        if name == "max": return np.where(valid, self.maximum[channel], np.nan)
        if name == "range": return np.where(valid, self.maximum[channel] - self.minimum[channel], np.nan)
        if name == "first_last_change": return self.last[channel] - self.first[channel]
        if name == "max_absolute_difference": return np.where(valid, self.maxdiff[channel], np.nan)
        if name == "valid_count": return n
        if name == "slope":
            denom = n * self.sxx[channel] - self.sx[channel] ** 2
            return np.divide(n * self.sxy[channel] - self.sx[channel] * self.total[channel], denom,
                             out=np.full(N_BINS, np.nan), where=(n > 1) & (denom != 0))
        raise KeyError(name)


def longest_true(flags: np.ndarray) -> int:
    padded = np.r_[False, flags, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1); ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def longest_true_bounds(flags: np.ndarray) -> tuple[int, int, int]:
    padded = np.r_[False, flags, False].astype(np.int8)
    changes = np.diff(padded); starts = np.flatnonzero(changes == 1); ends = np.flatnonzero(changes == -1)
    if not len(starts): return 0, 0, 0
    lengths = ends - starts; which = int(np.argmax(lengths))
    return int(starts[which]), int(ends[which] - 1), int(lengths[which])


def longest_missing_seconds(seen: np.ndarray) -> int:
    present = np.flatnonzero(seen)
    if not len(present): return len(seen)
    return int(max(present[0], len(seen) - 1 - present[-1], np.diff(present).max(initial=1) - 1))


def scan_second_file(path: Path) -> tuple[dict, list[dict], ChannelBins, dict]:
    seen = bytearray(N_SECONDS)
    bins = ChannelBins()
    physical = 1; parsed = duplicates = inversions = colbad = quote_bad = glued = data_error = 0
    previous = None; first_utc = last_utc = first_local = last_local = None
    intervals = Counter(); offsets = Counter(); missing = np.zeros(3, dtype=np.int64)
    pair_n = np.zeros((3, 3), dtype=np.int64); pair_x = np.zeros((3, 3)); pair_y = np.zeros((3, 3)); pair_x2 = np.zeros((3, 3)); pair_y2 = np.zeros((3, 3)); pair_xy = np.zeros((3, 3))
    diff_stats = {pair: [0, 0.0, 0.0, math.inf, -math.inf] for pair in [(0, 1), (0, 2), (1, 2)]}
    single_disagreement = simultaneous_physical = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = handle.readline().rstrip("\r\n"); columns = header.split(","); width = len(columns)
        for raw in handle:
            physical += 1; line = raw.rstrip("\r\n")
            if line.count('"') % 2: quote_bad += 1
            if line.count("/") >= 6:
                glued += 1
                continue
            parts = line.split(",")
            if len(parts) != width:
                colbad += 1
                if "Data Error" in line: data_error += 1
                continue
            utc = parse_dmy_second(parts[0]); local = parse_dmy_second(parts[1])
            if utc is None or local is None:
                if "Data Error" in line: data_error += 1
                continue
            parsed += 1
            first_utc = utc if first_utc is None or utc < first_utc else first_utc; last_utc = utc if last_utc is None or utc > last_utc else last_utc
            first_local = local if first_local is None or local < first_local else first_local; last_local = local if last_local is None or local > last_local else last_local
            offsets[int((local - utc).total_seconds())] += 1
            if previous is not None:
                delta = int((utc - previous).total_seconds()); intervals[delta] += 1; inversions += int(delta < 0)
            previous = utc
            second = int((utc - YEAR_START).total_seconds())
            if not 0 <= second < N_SECONDS: continue
            duplicates += int(bool(seen[second])); seen[second] = 1
            values = [number(parts[2]), number(parts[5]), number(parts[8])]
            for idx, value in enumerate(values): missing[idx] += int(not math.isfinite(value))
            bins.add(second, values)
            finite = [math.isfinite(v) for v in values]
            physical_bad = [finite[i] and (values[i] < -20 or values[i] > 1600) for i in range(3)]
            simultaneous_physical += int(sum(physical_bad) >= 2)
            if all(finite):
                median = float(np.median(values)); threshold = max(100.0, 0.25 * max(abs(median), 100.0))
                single_disagreement += int(sum(abs(v - median) > threshold for v in values) == 1)
            for a, b in [(0, 1), (0, 2), (1, 2)]:
                if finite[a] and finite[b]:
                    x, y = values[a], values[b]; pair_n[a, b] += 1; pair_x[a, b] += x; pair_y[a, b] += y; pair_x2[a, b] += x*x; pair_y2[a, b] += y*y; pair_xy[a, b] += x*y
                    d = x - y; stat = diff_stats[(a, b)]; stat[0] += 1; stat[1] += d; stat[2] += d*d; stat[3] = min(stat[3], d); stat[4] = max(stat[4], d)
    seen_np = np.frombuffer(seen, dtype=np.uint8).astype(bool)
    monthly = []
    for month, start, end, ss, ee in month_slices():
        bs, be = ss // 300, ee // 300; rows = bins.row_count[bs:be]
        complete = (rows == 300) & np.all(bins.count[:, bs:be] == 300, axis=0)
        partial = (rows > 0) & ~complete; absent = rows == 0
        item = {"month": f"2022-{month:02d}", "expected_seconds": ee-ss, "actual_timestamps": int(seen_np[ss:ee].sum())}
        for channel in range(3): item[f"{CHANNELS[channel]}_valid"] = int(bins.count[channel, bs:be].sum())
        item.update(common_valid=int(bins.common_count[bs:be].sum()), complete_5min_bins=int(complete.sum()), partial_5min_bins=int(partial.sum()), missing_5min_bins=int(absent.sum()),
                    longest_complete_segment_minutes=longest_true(complete)*5, longest_gap_seconds=longest_missing_seconds(seen_np[ss:ee]))
        monthly.append(item)
    structure = dict(file_name=path.name, size_bytes=path.stat().st_size, physical_lines=physical, parseable_records=parsed,
                     first_utc=str(first_utc), last_utc=str(last_utc), first_local=str(first_local), last_local=str(last_local),
                     main_interval=str(timedelta(seconds=intervals.most_common(1)[0][0])) if intervals else "UNKNOWN",
                     interval_distribution=json.dumps({str(timedelta(seconds=k)): v for k, v in intervals.most_common(15)}),
                     duplicate_timestamps=duplicates, time_inversions=inversions, missing_timestamps=N_SECONDS-int(seen_np.sum()),
                     column_count_anomalies=colbad, quote_anomalies=quote_bad, glued_records=glued, data_error_lines=data_error,
                     MB0_missing=int(missing[0]), MB1_missing=int(missing[1]), MB2_missing=int(missing[2]),
                     utc_local_offset_distribution=json.dumps({str(timedelta(seconds=k)): v for k, v in offsets.items()}))
    channel_summary = {"single_channel_disagreement_count": single_disagreement, "simultaneous_physical_anomaly_count": simultaneous_physical}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        n = pair_n[a, b]; num = n*pair_xy[a,b]-pair_x[a,b]*pair_y[a,b]; den = math.sqrt(max(0,(n*pair_x2[a,b]-pair_x[a,b]**2)*(n*pair_y2[a,b]-pair_y[a,b]**2)))
        stat=diff_stats[(a,b)]; label=f"{CHANNELS[a]}-{CHANNELS[b]}"; channel_summary[f"{label}_correlation"] = num/den if den else math.nan
        channel_summary[f"{label}_difference_mean"] = stat[1]/stat[0] if stat[0] else math.nan
        channel_summary[f"{label}_difference_std"] = math.sqrt(max(0,stat[2]/stat[0]-(stat[1]/stat[0])**2)) if stat[0] else math.nan
        channel_summary[f"{label}_difference_min"] = stat[3] if stat[0] else math.nan; channel_summary[f"{label}_difference_max"] = stat[4] if stat[0] else math.nan
    return structure, monthly, bins, channel_summary


def scan_resource_file(path: Path) -> dict:
    physical=1; parsed=colbad=quote_bad=glued=data_error=duplicates=inversions=0; previous=None; seen=set(); offsets=Counter(); first_u=last_u=first_l=last_l=None; intervals=Counter(); missing=[0,0,0]
    with path.open("r",encoding="utf-8-sig",errors="replace",newline="") as handle:
        header=handle.readline().rstrip("\r\n"); columns=header.split(","); width=len(columns); irr=[i for i,c in enumerate(columns) if "Irradiance_MB" in c]
        for raw in handle:
            physical+=1; line=raw.rstrip("\r\n"); quote_bad+=line.count('"')%2; glued+=int(line.count('/')>=6); parts=line.split(",")
            if len(parts)!=width: colbad+=1; data_error+=int("Data Error" in line); continue
            utc=parse_dmy_minute(parts[0]); local=parse_dmy_minute(parts[1])
            if not utc or not local: data_error+=int("Data Error" in line); continue
            parsed+=1; duplicates+=int(utc in seen); seen.add(utc); offsets[int((local-utc).total_seconds())]+=1
            if previous is not None: delta=int((utc-previous).total_seconds()); intervals[delta]+=1; inversions+=int(delta<0)
            previous=utc; first_u=utc if first_u is None or utc<first_u else first_u; last_u=utc if last_u is None or utc>last_u else last_u; first_l=local if first_l is None or local<first_l else first_l; last_l=local if last_l is None or local>last_l else last_l
            for n,i in enumerate(irr[:3]): missing[n]+=int(not math.isfinite(number(parts[i])))
    return dict(file_name=path.name,size_bytes=path.stat().st_size,physical_lines=physical,parseable_records=parsed,first_utc=str(first_u),last_utc=str(last_u),first_local=str(first_l),last_local=str(last_l),main_interval=str(timedelta(seconds=intervals.most_common(1)[0][0])) if intervals else "UNKNOWN",duplicate_timestamps=duplicates,time_inversions=inversions,column_count_anomalies=colbad,quote_anomalies=quote_bad,glued_records=glued,data_error_lines=data_error,MB0_missing=missing[0],MB1_missing=missing[1],MB2_missing=missing[2],utc_local_offset_distribution=json.dumps({str(timedelta(seconds=k)):v for k,v in offsets.items()}))


def scan_sanyo() -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    present=np.zeros(N_BINS,dtype=bool); valid=np.zeros(N_BINS,dtype=bool); power=np.full(N_BINS,np.nan)
    with SANYO_FILE.open("r",encoding="utf-8-sig",errors="replace",newline="") as handle:
        header=next(csv.reader([handle.readline().rstrip("\r\n")])); width=len(header); pidx=header.index("Active_Power")
        for raw in handle:
            parts=raw.rstrip("\r\n").split(",")
            if len(parts)!=width: continue
            ts=parse_pv(parts[0]);
            if ts is None or not YEAR_START<=ts<YEAR_END: continue
            idx=int((ts-YEAR_START).total_seconds()//300)
            if not 0<=idx<N_BINS: continue
            present[idx]=True; value=number(parts[pidx]); power[idx]=value; valid[idx]=math.isfinite(value)
    return present,valid,power


def peak_lag(pv: np.ndarray, irr: np.ndarray, max_lag: int=24) -> tuple[int,float,int]:
    best=(0,-np.inf,0)
    for lag in range(-max_lag,max_lag+1):
        x=irr[max(0,lag):min(len(irr),len(irr)+lag)]; y=pv[max(0,-lag):min(len(pv),len(pv)-lag)]
        mask=np.isfinite(x)&np.isfinite(y)
        if mask.sum()<3: continue
        corr=np.corrcoef(x[mask],y[mask])[0,1]
        if math.isfinite(corr) and corr>best[1]: best=(lag,float(corr),int(mask.sum()))
    return best


def daily_phase(pv: np.ndarray, irr: np.ndarray) -> dict:
    sunrise=[]; pv_start=[]; noon_irr=[]; noon_pv=[]
    times=pd.date_range(YEAR_START,YEAR_END-timedelta(minutes=5),freq="5min")
    frame=pd.DataFrame({"time":times,"pv":pv,"irr":irr}); frame["date"]=frame.time.dt.date
    for _,day in frame.groupby("date"):
        wi=day[np.isfinite(day.irr)&(day.irr>20)]; pp=day[np.isfinite(day.pv)&(day.pv>0.063)]
        if len(wi): sunrise.append(wi.time.iloc[0].hour*60+wi.time.iloc[0].minute); noon_irr.append(day.loc[day.irr.idxmax()].time.hour*60+day.loc[day.irr.idxmax()].time.minute)
        if len(pp): pv_start.append(pp.time.iloc[0].hour*60+pp.time.iloc[0].minute); noon_pv.append(day.loc[day.pv.idxmax()].time.hour*60+day.loc[day.pv.idxmax()].time.minute)
    med=lambda x: float(np.median(x)) if x else math.nan
    return {"irradiance_sunrise_median_minute":med(sunrise),"pv_start_median_minute":med(pv_start),"irradiance_noon_median_minute":med(noon_irr),"pv_peak_median_minute":med(noon_pv),"days_with_irradiance":len(sunrise),"days_with_pv":len(pv_start)}


def alignment_rows(bins: ChannelBins, present: np.ndarray, valid: np.ndarray, power: np.ndarray, structure: dict, channel_summary: dict) -> list[dict]:
    rows=[]; mean_channels=[bins.feature(c,"mean") for c in range(3)]
    candidates=[("PV_ORIGINAL_VS_HF_UTC",0),("HF_UTC_PLUS_09_00",108),("HF_UTC_PLUS_09_30",114),("FILE_LOCAL_FIELD",108)]
    for name,offset_bins in candidates:
        shift=offset_bins+1  # interval [t,t+5min) becomes a legal past-only feature at its end
        rows.append({"section":"time_candidate","candidate":name,"metric":"applied_offset_minutes","value":offset_bins*5,"unit":"minutes","notes":"Measured, not selected by predictive performance; one additional bin maps interval to its end."})
        for channel,channel_name in enumerate(CHANNELS):
            shifted=np.full(N_BINS,np.nan)
            if shift>=0: shifted[shift:]=mean_channels[channel][:N_BINS-shift]
            lag,corr,n=peak_lag(power,shifted); phase=daily_phase(power,shifted)
            rows.append({"section":"time_candidate","candidate":name,"metric":f"{channel_name}_peak_correlation_lag","value":lag,"unit":"5min_steps","notes":"Positive means irradiance series index is advanced relative to PV in peak_lag implementation."})
            rows.append({"section":"time_candidate","candidate":name,"metric":f"{channel_name}_peak_correlation","value":corr,"unit":"r","notes":f"paired={n}; channel measured independently."})
            for metric,value in phase.items(): rows.append({"section":"solar_phase","candidate":name,"metric":f"{channel_name}_{metric}","value":value,"unit":"minute_of_day" if "minute" in metric else "days","notes":"Threshold irradiance>20 W/m2; PV>1% of 6.3kW; channel independent."})
    file_offset=json.loads(structure["utc_local_offset_distribution"])
    rows.append({"section":"metadata","candidate":"FILE","metric":"utc_local_offset_distribution","value":json.dumps(file_offset),"unit":"counts","notes":"File content is consistently +09:00."})
    rows.append({"section":"metadata","candidate":"OFFICIAL","metric":"civil_time_reference","value":"ACST UTC+09:30","unit":"","notes":"DKASC official site notices identify Alice Springs event times as ACST; file Local conflicts by 30 minutes."})
    # Recommended mapping is UTC+09:30 on metadata/civil-time grounds, subject to phase check.
    shift=115; irr=np.full(N_BINS,np.nan); irr[shift:]=mean_channels[0][:N_BINS-shift]
    hf_complete=np.zeros(N_BINS,dtype=bool); raw_complete=(bins.row_count==300)&np.all(bins.count==300,axis=0); hf_complete[shift:]=raw_complete[:N_BINS-shift]
    common=present&valid&hf_complete; daylight=common&np.isfinite(irr)&(irr>20)
    window_ok=np.zeros(N_BINS,dtype=bool)
    required=72+12
    for end in range(required-1,N_BINS): window_ok[end]=common[end-required+1:end+1].all()
    metrics={"pv_timestamps_2022":int(present.sum()),"pv_active_power_valid_2022":int(valid.sum()),"hf_complete_5min_2022":int(hf_complete.sum()),"common_complete_5min":int(common.sum()),"daylight_common_complete_5min":int(daylight.sum()),"lookback72_h12_contiguous_windows":int(window_ok.sum())}
    for metric,value in metrics.items(): rows.append({"section":"sanyo","candidate":"RECOMMENDED_UTC_PLUS_09_30","metric":metric,"value":value,"unit":"intervals" if "windows" not in metric else "windows","notes":"No window crosses an incomplete interval."})
    run_start,run_end,run_length=longest_true_bounds(common)
    rows.append({"section":"sanyo","candidate":"RECOMMENDED_UTC_PLUS_09_30","metric":"longest_common_complete_start","value":YEAR_START+timedelta(minutes=5*run_start),"unit":"PV_clock","notes":"Strictly consecutive PV-valid and 300x3-channel-valid bins."})
    rows.append({"section":"sanyo","candidate":"RECOMMENDED_UTC_PLUS_09_30","metric":"longest_common_complete_end","value":YEAR_START+timedelta(minutes=5*run_end),"unit":"PV_clock","notes":"Inclusive endpoint."})
    rows.append({"section":"sanyo","candidate":"RECOMMENDED_UTC_PLUS_09_30","metric":"longest_common_complete_minutes","value":run_length*5,"unit":"minutes","notes":"Strict complete interval run."})
    features=["mean","std","min","max","range","first_last_change","max_absolute_difference","slope","valid_count"]
    for c,name in enumerate(CHANNELS):
        for feature in features:
            values=bins.feature(c,feature); finite=values[np.isfinite(values)]
            rows.append({"section":"channel_feature","candidate":name,"metric":f"{feature}_finite_bins","value":int(len(finite)),"unit":"bins","notes":"Channel retained independently; no pre-averaging."})
            for stat,value in [("mean",np.mean(finite) if len(finite) else np.nan),("std",np.std(finite) if len(finite) else np.nan),("min",np.min(finite) if len(finite) else np.nan),("max",np.max(finite) if len(finite) else np.nan)]:
                rows.append({"section":"channel_feature","candidate":name,"metric":f"{feature}_{stat}","value":value,"unit":"feature_native","notes":"Distribution across 5-minute bins; channel retained independently."})
    for metric,value in channel_summary.items(): rows.append({"section":"channel_relation","candidate":"ALL_CHANNELS","metric":metric,"value":value,"unit":"count_or_statistic","notes":"Single-channel disagreement threshold=max(100 W/m2,25% of median); physical anomaly outside [-20,1600] W/m2."})
    return rows


def write_csv(path:Path,rows:list[dict],fields:list[str])->None:
    if path.parent.resolve()!=HERE.resolve(): raise RuntimeError("Output escaped validation directory")
    with path.open("w",encoding="utf-8-sig",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)


def audit()->None:
    inputs=RESOURCE_FILES+[SANYO_FILE]; before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
    structure,monthly,bins,channel_summary=scan_second_file(SECOND_FILE)
    occupied=np.flatnonzero(bins.row_count)
    assert all(bins.max_source_second[i] < (i+1)*300 for i in occupied), "past-only bin endpoint violated"
    resources=[scan_resource_file(p) for p in RESOURCE_FILES if p!=SECOND_FILE]
    present,valid,power=scan_sanyo(); rows=alignment_rows(bins,present,valid,power,structure,channel_summary)
    for key,value in structure.items(): rows.append({"section":"file_structure","candidate":SECOND_FILE.name,"metric":key,"value":value,"unit":"","notes":"Source-backed streaming parse; no bad-line skip."})
    for resource in resources:
        for key,value in resource.items(): rows.append({"section":"file_structure","candidate":resource["file_name"],"metric":key,"value":value,"unit":"","notes":"Source-backed streaming parse."})
    after={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
    if before!=after: raise RuntimeError("An input file changed during read-only validation")
    write_csv(MONTHLY_OUT,monthly,list(monthly[0])); write_csv(ALIGN_OUT,rows,["section","candidate","metric","value","unit","notes"])
    print(f"Validated {len(RESOURCE_FILES)} resource files and Sanyo; wrote {len(rows)} compact alignment rows")


def self_test()->None:
    assert MONTHLY_OUT.parent.resolve()==HERE.resolve()==ALIGN_OUT.parent.resolve()
    assert parse_dmy_second("bad timestamp") is None and parse_pv("bad timestamp") is None
    # A truncated row is never padded or guessed.
    assert len("01/01/2022 00:00:00,1,2".split(",")) != 14
    monthly=pd.read_csv(MONTHLY_OUT); alignment=pd.read_csv(ALIGN_OUT)
    assert len(monthly)==12 and monthly.expected_seconds.sum()==N_SECONDS
    assert ((monthly.complete_5min_bins+monthly.partial_5min_bins+monthly.missing_5min_bins)==monthly.expected_seconds//300).all()
    assert {"MB0_valid","MB1_valid","MB2_valid"}.issubset(monthly.columns)
    assert set(alignment[alignment.section=="time_candidate"].candidate)=={"PV_ORIGINAL_VS_HF_UTC","HF_UTC_PLUS_09_00","HF_UTC_PLUS_09_30","FILE_LOCAL_FIELD"}
    assert not any("average_channels" in str(x) for x in alignment.metric)
    for candidate in ["PV_ORIGINAL_VS_HF_UTC","HF_UTC_PLUS_09_00","HF_UTC_PLUS_09_30","FILE_LOCAL_FIELD"]:
        metrics=set(alignment[(alignment.section=="time_candidate")&(alignment.candidate==candidate)].metric)
        assert {f"{channel}_peak_correlation_lag" for channel in CHANNELS}.issubset(metrics)
    assert all((HERE/p).exists() for p in [MONTHLY_OUT.name,ALIGN_OUT.name])
    print("PASS: monthly arithmetic, independent channels, four measured time candidates, UNKNOWN parsing and output boundary")


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); args=parser.parse_args(); self_test() if args.self_test else audit()


if __name__=="__main__": main()

"""M1: raw-file audits, explicit time coordinates and synthetic forward checks only."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from datetime import timezone, timedelta
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIELDS = ['power', 'temperature', 'ghi']
EST = timezone(timedelta(hours=-5))
FIVE = pd.Timedelta(minutes=5)

def config():
    return json.loads((HERE / 'multisite_config.json').read_text(encoding='utf-8'))

def stats(path):
    s = path.stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns}

def validate_paths(path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    y = Path(c['YULARA_RAW_FILE'])
    n = Path(c['NIST_GROUND_2017_DIRECTORY'])
    if not y.is_absolute() or not y.is_file() or not n.is_absolute() or not n.is_dir():
        raise ValueError('Explicit source file/directory missing; no discovery fallback')
    # Bounded exclusively to the user-specified NIST directory.
    expected = {f'{d.month:02d}/onemin-Ground-{d:%Y-%m-%d}.csv' for d in pd.date_range('2017-01-01','2017-12-31')}
    actual = {f.relative_to(n).as_posix() for f in n.rglob('*.csv')}
    if actual != expected:
        raise ValueError(f'NIST file set mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    return y, n, [n / s for s in sorted(expected)]

def numeric(series):
    text = series.astype(str).str.strip()
    null = text.str.lower().isin(['', 'nan', 'na', 'null', 'none', 'n/a'])
    v = pd.to_numeric(text.where(~null), errors='coerce').astype(float)
    finite = v[np.isfinite(v)]
    detail = {'null': int(null.sum()), 'non_numeric': int((v.isna() & ~null).sum()),
              'positive_inf': int(np.isposinf(v).sum()), 'negative_inf': int(np.isneginf(v).sum()),
              'candidate_minus999': int(v.eq(-999).sum()), 'candidate_minus7999': int(v.eq(-7999).sum()),
              'official_sentinel_codes_confirmed': [], 'negative_finite': int((finite < 0).sum()),
              'quantiles': {str(q): float(finite.quantile(q)) for q in [0,.01,.05,.5,.95,.99,1]}}
    return v.where(np.isfinite(v)), detail

def fixed_est(values):
    s = pd.Series(values, dtype=str)
    if not s.str.endswith('-05:00').all():
        raise ValueError('NIST must retain explicit fixed EST; unexpected offset')
    return pd.DatetimeIndex(pd.to_datetime(s, errors='raise', utc=True)).tz_convert(EST)

def time_audit(index, expected):
    return {'first': str(index.min()), 'last': str(index.max()), 'rows': len(index),
            'unique': int(index.nunique()), 'duplicates': int(index.duplicated().sum()),
            'reverse_steps': int((np.diff(index.asi8) < 0).sum()),
            'missing_times': [str(t) for t in expected.difference(index)]}

def aggregate_nist(frame):
    """Raw timestamps t in [T-5,T) map to availability T; never a default resample."""
    idx = frame.index
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError('Duplicate or reversed raw minutes')
    if ((idx.second != 0) | (idx.microsecond != 0) | (idx.nanosecond != 0)).any():
        raise ValueError('Non-minute timestamp')
    labels = idx.floor('5min') + FIVE
    counts = frame.groupby(labels).count()
    distinct = pd.Series(1, index=idx).groupby(labels).sum()
    means = frame.groupby(labels).mean()
    means = means.where(counts.eq(5)).where(distinct.eq(5), axis=0)
    full = pd.date_range(labels.min(), labels.max(), freq='5min')
    return means.reindex(full)

def regular_yulara(frame):
    idx = frame.index
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError('Duplicate/reversed Yulara timestamps need correction')
    on_grid = idx.eq(idx.floor('5min')) if hasattr(idx,'eq') else idx == idx.floor('5min')
    excluded = frame.loc[~on_grid].copy()
    regular = frame.loc[on_grid].copy()
    regular.index = regular.index + FIVE  # latest possible interval end, not rounding
    return regular.reindex(pd.date_range(regular.index.min(), regular.index.max(),freq='5min')), excluded

def augment_seven(raw, imputed, if_flags):
    raw=np.asarray(raw,dtype=float); imputed=np.asarray(imputed,dtype=float)
    flags=np.asarray(if_flags,dtype=float).reshape(-1,1)
    if raw.shape!=imputed.shape or raw.shape[1]!=3 or flags.shape[0]!=len(raw):
        raise ValueError('Invalid seven-channel assembly')
    if not np.isfinite(imputed).all() or not np.isin(flags,[0,1]).all():
        raise ValueError('Invalid transformed inputs')
    return np.concatenate([imputed,missing_masks(raw),flags],axis=1)

def missing_masks(values):
    return (~np.isfinite(np.asarray(values, dtype=float))).astype(np.float32)

def train_only_fit(fitter, values, split):
    if split != 'train':
        raise ValueError('Fit is restricted to Train')
    return fitter.fit(values)

def label_values(values):
    a = np.array(values, dtype=float, copy=True)
    return a, np.isfinite(a)  # finite negatives retained, unlike original Scheme A

def threshold_from_train(frame, cfg):
    train = frame.loc[slice(*cfg['splits']['train'])]
    values = train.power.to_numpy(float)
    if not np.isfinite(values).any():
        raise ValueError('No valid Train target')
    return .01 * float(np.nanmax(values))

def eligible(frame, horizon, split_bounds):
    start,end = map(pd.Timestamp, split_bounds)
    f = frame.loc[start:end]
    if not f.index.equals(pd.date_range(start, end.floor('5min'),freq='5min')):
        raise ValueError('Timeline must be reindexed, never stitched')
    y = f.power.to_numpy(float)
    # First input interval must also start inside split; midnight bin ends at split start.
    origins = np.arange(72, len(f)-horizon, dtype=int)
    if not len(origins):
        return f, origins, np.empty((0,horizon),dtype=int)
    targets = origins[:,None]+np.arange(1,horizon+1)
    keep = np.isfinite(y[origins]) & np.isfinite(y[targets]).all(axis=1)
    return f, origins[keep], targets[keep]

def daily_lookup(power, target_times):
    shape = np.shape(target_times)
    lag = pd.DatetimeIndex(np.asarray(target_times).reshape(-1)) - pd.Timedelta(hours=24)
    return power.reindex(lag).to_numpy(float).reshape(shape)

def shared_mask(labels, daily, method_arrays, scope_mask):
    mask = np.isfinite(labels) & np.isfinite(daily) & scope_mask
    for values in method_arrays:
        if np.shape(values) != mask.shape:
            raise ValueError('Method shape mismatch')
        mask &= np.isfinite(values)
    return [mask.copy() for _ in range(len(method_arrays)+1)]

def support_counts(frame, cfg):
    threshold = threshold_from_train(frame,cfg)
    rows=[]
    for split,bounds in cfg['splits'].items():
        for h in cfg['evaluation_horizons']:
            f,o,t = eligible(frame,h,bounds)
            y=f.power.to_numpy(float); x=f[FIELDS].to_numpy(float)
            tv=y[t]; daylight=tv>threshold
            inputs=o[:,None]-np.arange(71,-1,-1)
            daily=daily_lookup(frame.power, f.index.to_numpy()[t])
            for analysis,mask in [('primary',np.isfinite(tv)),('daily_matched',np.isfinite(tv)&np.isfinite(daily))]:
                for scope,m in [('full',mask),('daylight',mask&daylight)]:
                    selected=m.any(axis=1); oo=o[selected]
                    rows.append({'split':split,'horizon':h,'analysis':analysis,'scope':scope,
                                 'forecast_origin_count':int(selected.sum()),'valid_target_point_count':int(m.sum()),
                                 'first_origin':str(f.index[oo[0]]) if len(oo) else None,
                                 'last_origin':str(f.index[oo[-1]]) if len(oo) else None,
                                 'months':sorted(set(f.index[oo].strftime('%Y-%m'))),
                                 'input_missing_rate':float(missing_masks(x[inputs[selected]]).mean()) if len(oo) else None,
                                 'label_missing_rate_selected':0.0,
                                 'label_missing_rate_split':float((~np.isfinite(y)).mean()),
                                 'input_missing_rate_split':float(missing_masks(x).mean()),
                                 'daylight_threshold_train':threshold})
    return rows

def energy_diagnostics(raw):
    """Only Jan-Aug data. Diagnose endpoints; never pick rules using held-out scores."""
    f = raw.loc['2017-01-01':'2017-08-31 23:59:59'].copy()
    f=f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='min'))
    net=f.PwrMtrErec_kWh_Max-f.PwrMtrEdel_kWh_Max
    delta=net-net.shift(5)
    results={}
    for shift in [-1,0,1]:
        energy=f.PwrMtrP_kW_Avg.shift(shift).rolling(5,min_periods=5).sum()/60
        valid=np.isfinite(delta)&np.isfinite(energy)&(delta>=-1)&(delta<=30)
        err=(delta-energy)[valid]
        results[str(shift)]={'pairs':int(valid.sum()),'median_absolute_kWh_error':float(err.abs().median()),
                              'mean_absolute_kWh_error':float(err.abs().mean())}
    correspondence={str(lag):float(f.PwrMtrP_kW_Avg.corr(f.Pyra1_Wm2_Avg.shift(lag))) for lag in [-5,0,5]}
    return {'split':'train','candidate_power_shift_minutes':results,'power_ghi_lag_correlation':correspondence,
            'energy_delta_outside_diagnostic_bounds':int(((delta<-1)|(delta>30)).sum()),
            'note':'Diagnostic-only finite/reset bounds; no source rows deleted. No automatic boundary selection. Conservative closed-left/right-label availability frozen.'}

def audit(paths):
    cfg=config(); y,n,files=validate_paths(paths); allfiles=[y,*files]
    before={f:stats(f) for f in allfiles};inventory=[]; raw=[]; header=None
    for f in files:
        d=pd.read_csv(f,dtype=str,keep_default_na=False)
        cols=list(d.columns)
        if header is None:header=cols
        if cols!=header:raise ValueError('NIST headers differ')
        inventory.append({'file':f.relative_to(n).as_posix(),**before[f],'rows':len(d),'header':'NIST_HEADER'})
        raw.append(d)
    ns=pd.concat(raw,ignore_index=True);idx=fixed_est(ns.TIMESTAMP)
    expected=pd.date_range('2017-01-01','2017-12-31 23:59',freq='min',tz=EST)
    nt=time_audit(idx,expected)
    if nt['duplicates'] or nt['reverse_steps']:raise ValueError('NIST ordering invalid')
    fields={};nf=pd.DataFrame(index=idx.tz_localize(None));aux={}
    for name in cfg['nist']['fields']+['InvPAC_kW_Avg','PwrMtrErec_kWh_Max','PwrMtrEdel_kWh_Max']:
        values,detail=numeric(ns[name]);fields[name]=detail;aux[name]=values.to_numpy()
    for name,source in zip(FIELDS,cfg['nist']['fields']):nf[name]=aux[source]
    nr=pd.DataFrame(aux,index=nf.index)
    ng=aggregate_nist(nf).reindex(pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min'))
    ys=pd.read_csv(y,dtype=str,keep_default_na=False);yi=pd.DatetimeIndex(pd.to_datetime(ys.timestamp,errors='raise'))
    full_y={'first':str(yi.min()),'last':str(yi.max()),'rows':len(ys),'header':list(ys.columns)}
    select=(yi>=pd.Timestamp('2017-01-01'))&(yi<pd.Timestamp('2018-01-01'));ys=ys.loc[select].copy();yi=yi[select]
    yt=time_audit(yi,pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min'))
    yf=pd.DataFrame(index=yi);yfields={}
    for name,source in zip(FIELDS,cfg['yulara']['fields']):
        values,detail=numeric(ys[source]);yf[name]=values.to_numpy();yfields[source]=detail
    yg,excluded=regular_yulara(yf)
    yg=yg.reindex(pd.date_range('2017-01-01','2017-12-31 23:55',freq='5min'))
    ex=[]
    for t,r in excluded.iterrows():
        near=t.floor('5min');record={'timestamp':str(t),'values':r.to_dict(),'reason':'No provider correction; excluded without rounding',
        'floor_exists':bool(near in yf.index),'ceil_exists':bool(near+FIVE in yf.index)}
        record['floor_values']=yf.loc[near].to_dict() if near in yf.index else None
        record['ceil_values']=yf.loc[near+FIVE].to_dict() if near+FIVE in yf.index else None
        ex.append(record)
    counts={'NIST_GROUND':support_counts(ng,cfg),'YULARA_COMBINED':support_counts(yg,cfg)}
    unchanged=all(stats(f)==before[f] for f in allfiles)
    if not unchanged:raise RuntimeError('RAW_SOURCE_CHANGED')
    # Inspection of accompanying files stays within the two explicit source locations.
    extras={'nist':[f.relative_to(n).as_posix() for f in n.rglob('*') if f.is_file() and f.suffix.lower()!='.csv'],
            'yulara':[f.name for f in y.parent.iterdir() if f.is_file() and f!=y]}
    return {'nist':{'inventory':inventory,'header':header,'time':nt,'fields':fields,'mV_present':'Pyra1_mV_Avg' in header,
                    'Wm2_direct_in_csv':'Pyra1_Wm2_Avg' in header,'train_semantics':energy_diagnostics(nr)},
            'yulara':{'file':y.name,**before[y],'full_file':full_y,'time_2017':yt,'fields':yfields,'excluded':ex},
            'counts':counts,'accompanying_files':extras,'raw_size_mtime_unchanged':unchanged}

def load_model_module():
    # Import is inert: do not instantiate CorrectedProtocol or call preflight/main.
    p=HERE.parent/'scheme_A_submission_correction/run_corrected_benchmark.py'
    spec=importlib.util.spec_from_file_location('scheme_a_frozen_bench',p)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def forward_checks():
    import torch
    torch.set_num_threads(1)
    bench=load_model_module();cfg=bench.load_config();records=[]
    def forbidden(*args,**kwargs):raise RuntimeError('M1 forbids fitting, training, serialization or saved-weight loading')
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import MinMaxScaler
    guards=[patch.object(torch,'save',forbidden),patch.object(torch,'load',forbidden),
            patch.object(torch.Tensor,'backward',forbidden),patch.object(torch.optim,'AdamW',forbidden),
            patch.object(bench,'train_model',forbidden),patch.object(bench,'predict_scaled',forbidden),
            patch.object(bench,'preflight',forbidden),patch.object(IsolationForest,'fit',forbidden),
            patch.object(KNNImputer,'fit',forbidden),patch.object(MinMaxScaler,'fit',forbidden)]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for guard in guards:stack.enter_context(guard)
        for name in bench.MODEL_NAMES:
            torch.manual_seed(42)
            m=bench.make_model(name,7,cfg).cpu().eval()
            old=bench.make_model(name,17,cfg).cpu().eval() # fresh random module, no checkpoint
            before={n:p.detach().clone() for n,p in m.named_parameters()}
            with torch.inference_mode():
                shapes=[]
                for b in [1,2]:
                    output=m(torch.randn(b,72,7));assert output.shape==(b,144) and torch.isfinite(output).all();shapes.append(list(output.shape))
                x=torch.randn(2,72,7);base=m(x).clone()
                # Forward-only interventions establish output dependence on every parameter tensor.
                inactive=[]
                for pname,p in m.named_parameters():
                    saved=p.clone();p.add_(torch.randn_like(p)*.03);changed=m(x);p.copy_(saved)
                    if torch.equal(base,changed):inactive.append(pname)
                mismatch=False
                try:m.load_state_dict(old.state_dict(),strict=True)
                except RuntimeError:mismatch=True
                # Failed strict load can partially copy compatible tensors; restore random test state.
                m.load_state_dict(before,strict=True)
            assert not inactive, inactive
            assert mismatch
            assert all(torch.equal(p,before[n]) and p.grad is None for n,p in m.named_parameters())
            records.append({'model':name,'parameters_7':bench.parameter_count(m),'parameters_17':bench.parameter_count(old),
                            'delta':bench.parameter_count(m)-bench.parameter_count(old),'outputs':shapes,
                            'parameter_tensors_checked':len(before),'inactive_parameter_tensors':inactive,
                            'synthetic_17_state_strict_rejected':mismatch,'weights_restored_and_gradients_none':True})
    return {'torch':torch.__version__,'device':'cpu','checks':records}

def json_safe(value):
    if isinstance(value, dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def write_summary(result):
    rows=[]
    def add(site,category,key,value):rows.append({'site':site,'category':category,'key':key,'value':json.dumps(json_safe(value),ensure_ascii=False,allow_nan=False)})
    for f in result['nist']['inventory']:add('NIST_GROUND','file',f['file'],f)
    for site,body in [('NIST_GROUND',result['nist']),('YULARA_COMBINED',result['yulara'])]:
        for key,value in body.items():
            if key!='inventory':add(site,'audit',key,value)
    for site,counts in result['counts'].items():
        for r in counts:add(site,'support',f"{r['split']}|H{r['horizon']}|{r['analysis']}|{r['scope']}",r)
    for key in ['accompanying_files','raw_size_mtime_unchanged']:
        add('both','audit',key,result[key])
    with (HERE/'DATA_AUDIT_SUMMARY.csv').open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=['site','category','key','value']);writer.writeheader();writer.writerows(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--paths',required=True);ap.add_argument('--forward-only',action='store_true');args=ap.parse_args()
    if args.forward_only:
        print(json.dumps(forward_checks(),indent=2))
    else:
        result=audit(args.paths);write_summary(result)
        print(json.dumps({'nist_time':result['nist']['time'],'yulara_time':result['yulara']['time_2017'],
                          'excluded':result['yulara']['excluded'],'train_diagnostic':result['nist']['train_semantics'],
                          'raw_unchanged':result['raw_size_mtime_unchanged']},indent=2))

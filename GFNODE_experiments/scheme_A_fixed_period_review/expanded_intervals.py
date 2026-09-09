"""Saved expanded-grid predictions only; no fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from paired import reduce_blocks,intervals
HERE=Path(__file__).resolve().parent
def read(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def aligned(a,b):
    for k in ['forecast_origin','target_start','target_valid']:assert np.array_equal(a[k],b[k]),k
    for k in ['labels','daily','last_power']:assert np.allclose(a[k],b[k],equal_nan=True),k
def run(paths):
    c=json.loads(Path(paths).read_text(encoding='utf8'));out=HERE/'results';out.mkdir(exist_ok=True);rows=[];blocks=[];supports=[]
    for site in ['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']:
        old=Path(c['new_results'])/site;new=Path(c['expanded_results'][site]);a=read(old/'Ridge_predictions.npz')
        preds={}
        for key,root,stem in [('Original A',old,'Ridge'),('Original B',old,'Ridge_day'),('Expanded A',new,'Ridge'),('Expanded B',new,'Ridge_day')]:
            z=read(root/(stem+'_predictions.npz'));aligned(a,z);preds[key]=z['predictions']
        preds['Daily']=a['daily'];external=site in ['YULARA_COMBINED','NIST_GROUND']
        for f in Path(c['external_results' if external else 'alice_results']).glob('*/completed.json'):
            info=json.loads(f.read_text(encoding='utf8'))
            if info.get('site',info.get('dataset'))!=site or 'inverted' not in info['model'].lower():continue
            z=read(f.parent/('test_predictions.npz' if external else 'test_H144.npz'))
            for k in ['forecast_origin','target_start','target_valid']:assert np.array_equal(a[k],z[k]),k
            assert np.allclose(a['labels'],z['labels'],equal_nan=True)
            preds['Inverted'+str(info['seed'])]=z['predictions']
        assert all('Inverted'+str(s) in preds for s in [42,43,44])
        y=a['labels'];o=pd.DatetimeIndex(pd.to_datetime(a['forecast_origin'])).as_unit('ns');targets=o.asi8[:,None]+np.arange(1,145)*300_000_000_000
        assert pd.DatetimeIndex(pd.to_datetime(a['target_start'])).as_unit('ns').equals(o+pd.Timedelta(minutes=5))
        base=a['target_valid'].all(1)[:,None]&a['target_valid']&np.isfinite(a['last_power'])[:,None]&np.isfinite(a['daily'])
        comp=[('Expanded B',x) for x in ['Expanded A','Daily','Inverted42','Inverted43','Inverted44','Inverted mean']]+[('Expanded A','Original A'),('Expanded B','Original B')]
        for scope,mask in [('full',base),('power-active',base&(y>a['daylight_threshold'])),('low-power',base&(y<=a['daylight_threshold']))]:
            meta=dict(site=site,scope=scope,horizon=144,support='complete_H144_Daily_intersection')
            supports.append(dict(**meta,origins=int(mask.any(1).sum()),points=int(mask.sum()),unique_targets=len(np.unique(targets[mask]))))
            for h in [24,48,72]:
                b=reduce_blocks(y,preds,mask,o,'2017-10-01' if external else '2018-08-08',h);rows += [dict(**meta,block_hours=h,**r) for r in intervals(b,comp)];blocks.append(b.assign(**meta,block_hours=h))
        print(site,'exact saved origin/target/mask alignment and intervals completed',flush=True)
    pd.DataFrame(rows).to_csv(out/'expanded_paired_intervals.csv',index=False);pd.concat(blocks).to_csv(out/'expanded_block_sse.csv',index=False);pd.DataFrame(supports).to_csv(out/'expanded_support.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);v=p.parse_args();run(v.paths)

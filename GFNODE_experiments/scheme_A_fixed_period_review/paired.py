"""Paired origin-time block reduction, with an independently reusable CSV entry."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def reduce_blocks(y,predictions,mask,origins,anchor,hours):
    o=pd.DatetimeIndex(pd.to_datetime(origins)).as_unit('ns')
    anchor=pd.Timestamp(anchor,tz=o.tz) if pd.Timestamp(anchor).tz is None else pd.Timestamp(anchor)
    ids=np.asarray((o-anchor).total_seconds()//(hours*3600),int)
    assert ids.min()>=0
    nb=ids.max()+1
    data={'block':np.arange(nb),'count':np.bincount(ids,weights=mask.sum(1),minlength=nb)}
    for name,p in predictions.items():
        assert p.shape==y.shape and np.isfinite(p[mask]).all(),name
        e=np.where(mask,p.astype(float)-y.astype(float),0)
        data[name+'_SSE']=np.bincount(ids,weights=(e*e).sum(1),minlength=nb)
    return pd.DataFrame(data)

def intervals(blocks,comparisons):
    n=blocks['count'].to_numpy();nb=len(n);ix=np.random.default_rng(20260908).integers(0,nb,(2000,nb));nn=n[ix].sum(1);ok=nn>0
    rm={};point={}
    for k in blocks:
        if k.endswith('_SSE'):
            v=blocks[k].to_numpy();rm[k[:-4]]=np.sqrt(v[ix][ok].sum(1)/nn[ok]);point[k[:-4]]=np.sqrt(v.sum()/n.sum()) if n.sum() else np.nan
    rows=[]
    for method,reference in comparisons:
        methods=['Inverted42','Inverted43','Inverted44'] if method=='Inverted mean' else [method]
        refs=['Inverted42','Inverted43','Inverted44'] if reference=='Inverted mean' else [reference]
        pairs=[(a,r) for a in methods for r in refs]
        ds=np.array([rm[a]-rm[r] for a,r in pairs]);ps=np.array([point[a]-point[r] for a,r in pairs])
        with np.errstate(divide='ignore',invalid='ignore'):
            sk=np.mean([np.where(rm[r]>0,1-rm[a]/rm[r],np.nan) for a,r in pairs],axis=0)
            psk=np.mean([1-point[a]/point[r] if point[r]>0 else np.nan for a,r in pairs])
        d=ds.mean(0);finite=np.isfinite(sk)
        q=np.quantile(d,[.025,.975]) if len(d) else [np.nan]*2
        sq=np.quantile(sk[finite],[.025,.975]) if finite.any() else [np.nan]*2
        rows.append(dict(method=method,reference=reference,rmse=np.mean([point[a] for a in methods]),reference_rmse=np.mean([point[r] for r in refs]),effect_kW=ps.mean(),ci_low_kW=q[0],ci_high_kW=q[1],skill=psk,skill_ci_low=sq[0],skill_ci_high=sq[1],blocks=nb,nonempty_blocks=int((n>0).sum()),points=int(n.sum()),replicates=2000,valid_replicates=int(ok.sum()),zero_support_replicates=int((~ok).sum()),valid_skill_replicates=int(finite.sum())))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('blocks');p.add_argument('intervals');v=p.parse_args()
    b=pd.read_csv(v.blocks);expected=pd.read_csv(v.intervals);keys=[k for k in ['site','period','support','horizon','scope','block_hours'] if k in b]
    checked=0
    for key,g in b.groupby(keys,dropna=False):
        row=expected
        for k,x in zip(keys,key if isinstance(key,tuple) else [key]):row=row[row[k]==x]
        actual=intervals(g,list(zip(row.method,row.reference)))
        for a,(_,r) in zip(actual,row.iterrows()):
            for k,value in a.items():
                if isinstance(value,(int,float,np.number)):assert np.isclose(value,r[k],rtol=1e-9,atol=1e-10,equal_nan=True),(key,k,value,r[k])
        checked+=len(actual)
    print('CSV block replay:',checked,'paired rows; no model inference or training')

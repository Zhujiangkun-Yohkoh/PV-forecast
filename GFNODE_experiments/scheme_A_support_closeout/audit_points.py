"""Independent saved-point metric/support audit, no production scoring functions."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent;R=H/'results'
def read(p):
 with np.load(p,allow_pickle=True) as z:return {k:z[k] for k in z.files}
def audit(c):
 checked=0;support_checked=0
 for site in ['Sanyo','Hanwha','Qcells','YULARA_COMBINED','NIST_GROUND']:
  for period in (['Alice2018Test'] if site in ['Sanyo','Hanwha','Qcells'] else ['2017','2018']):
   alice=period=='Alice2018Test';stem=site if alice else site+'_'+period
   z=read(Path(c['destination'])/site/'support_and_ridge.npz') if alice else read(Path(c['closeout_destination'] if period=='2017' else c['destination'])/site/('unified_2017_predictions.npz' if period=='2017' else 'fixed_2018_predictions.npz'))
   y=z['labels'];o=pd.DatetimeIndex(pd.to_datetime(z['forecast_origin'])).as_unit('ns');t=o.asi8[:,None]+np.arange(1,145)*300_000_000_000;assert np.array_equal(z['target_timestamp_ns'],t);assert ((t-86400000000000)<=o.asi8[:,None]).all()
   p={k.replace('_',' '):z[k] for k in ['Original_A','Original_B','Expanded_A','Expanded_B']}
   if alice:
    p['Daily']=z['daily'];p['Last-value']=np.broadcast_to(z['last_power'][:,None],y.shape);threshold=float(read(Path(c['new_results'])/site/'Ridge_predictions.npz')['daylight_threshold']);valid=np.isfinite(y)&(y>=0)
    states=pd.read_csv(R/'Alice_method_status.csv');states=states[states.site==site]
    for row in states.itertuples():
     a=read(Path(c['destination'])/site/(row.method.replace(' ','_')+'.npz'));assert np.array_equal(a['forecast_origin'],z['forecast_origin']);p[row.method]=a['predictions']
   else:
    threshold=float(z['threshold']);valid=np.isfinite(y)
    for k in ['Daily','Last-value','Inverted42','Inverted43','Inverted44']:p[k]=z[k]
   met=pd.read_csv(R/(stem+'_metrics.csv'));su=pd.read_csv(R/(stem+'_support.csv'))
   for (h,support,scope),g in met.groupby(['horizon','support','scope']):
    yy=y[:,:h];v=valid[:,:h];complete=v.all(1) if alice else valid.all(1)
    mask=np.isfinite(yy) if support in ['S2','S2_new_origins','pointwise_finite'] else v.copy()
    if support in ['Original','complete_H144']:mask=mask&complete[:,None]
    if support.endswith('new_origins'):mask=mask&~complete[:,None]
    mask=mask&np.isfinite(p['Daily'][:,:h])&np.isfinite(p['Last-value'][:,0])[:,None]
    if scope=='power-active':mask &= yy>threshold
    if scope=='low-power':mask &= yy<=threshold
    ss=su[(su.horizon==h)&(su.support==support)&(su.scope==scope)].iloc[0]
    assert ss.points==mask.sum() and ss.origins==mask.any(1).sum() and ss.unique_targets==len(np.unique(t[:,:h][mask]));support_checked+=1
    for row in g.itertuples():
     e=p[row.method][:,:h][mask].astype(float)-yy[mask].astype(float);N=len(e);S=np.sum(e**2);rm=np.sqrt(S/N) if N else np.nan;ma=np.mean(abs(e)) if N else np.nan;bias=np.mean(e) if N else np.nan
     assert np.allclose([S,rm,ma,bias],[row.SSE,row.RMSE,row.MAE,row.bias],rtol=1e-10,atol=1e-10,equal_nan=True),(site,period,support,scope,row.method);checked+=1
   print(site,period,'point audit completed',flush=True)
 result=dict(actual_saved_point_metric_rows=checked,support_groups=support_checked,failed=0,skipped=0,neural_training=False,independent_of_production_mask_and_metrics=True);(R/'POINT_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);audit(json.loads(Path(p.parse_args().paths).read_text(encoding='utf8')))

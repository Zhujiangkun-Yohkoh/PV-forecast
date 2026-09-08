"""Quantify the supplemental small-alpha solver sensitivity without replacing predictions."""
from pathlib import Path
import argparse,pickle,json
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
from designs import variant_design,load_npz,HERE
def run(cache,folder):
 with Path(cache).open('rb') as f:d=pickle.load(f)
 assert d['folder'].name=='NIST_GROUND';_,X,c,_=variant_design(d,'Ridge');folder=Path(folder);saved=load_npz(folder/'Ridge_predictions.npz');alpha=json.loads((folder/'Ridge_completed.json').read_text(encoding='utf8'))['selected_alpha'];assert alpha==.0001
 Y=d['labels']['train']*c['target_scale']+c['target_min']-c['intercept'];cols=X['train'].shape[1];coef,_,rank,_=lstsq(np.vstack([X['train'],np.sqrt(alpha)*np.eye(cols)]),np.vstack([Y,np.zeros((cols,144))]),lapack_driver='gelsy');pred=(X['test']@coef+c['intercept']-c['target_min'])/c['target_scale'];assert np.array_equal(saved['forecast_origin'],d['archive']['forecast_origin']);a=d['archive'];y=a['labels'];base=a['target_valid'].all(1)[:,None]&a['target_valid']&np.isfinite(a['daily'])&np.isfinite(a['last_power'])[:,None];rows=[]
 for scope,mask in [('full',base),('power-active',base&(y>a['daylight_threshold'])),('low-power',base&(y<=a['daylight_threshold']))]:
  sr=np.sqrt(np.mean((saved['predictions'][mask]-y[mask])**2));qr=np.sqrt(np.mean((pred[mask]-y[mask])**2));rows.append(dict(site='NIST_GROUND',model='Ridge',alpha=alpha,scope=scope,points=int(mask.sum()),spectral_RMSE=sr,QR_RMSE=qr,RMSE_difference=qr-sr,max_prediction_difference=np.abs(pred[mask]-saved['predictions'][mask]).max(),augmented_rank=rank))
 pd.DataFrame(rows).to_csv(HERE/'results/nist_QR_metric_sensitivity.csv',index=False);np.savez_compressed(folder/'Ridge_QR_diagnostic_predictions.npz',predictions=pred,forecast_origin=saved['forecast_origin'],target_start=saved['target_start']);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--folder',required=True);a=p.parse_args()
 with threadpool_limits(limits=4):run(a.cache,a.folder)

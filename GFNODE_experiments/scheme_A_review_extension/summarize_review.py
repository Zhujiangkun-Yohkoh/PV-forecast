"""Small review tables from per-seed reductions; never aggregates cross-site kW."""
from pathlib import Path
import json,argparse
import pandas as pd
H=Path(__file__).resolve().parent;R=H/'results'
def run(paths):
 d=pd.read_csv(R/'metrics_decomposition.csv');q=d[(d.analysis=='daily_matched')&(d.support=='horizon_specific')];g=q.groupby(['site','h','model','scope']).agg(RMSE_mean=('RMSE','mean'),RMSE_seed_SD=('RMSE','std'),MAE_mean=('MAE','mean'),bias_mean=('bias','mean'),points=('points','first'),weighted_MSE=('weighted_MSE_contribution','mean')).reset_index()
 daily=g[g.model=='Daily'][['site','h','scope','weighted_MSE']].rename(columns={'weighted_MSE':'Daily_weighted_MSE'});g=g.merge(daily,on=['site','h','scope']);g['MSE_contribution_difference_vs_Daily']=g.weighted_MSE-g.Daily_weighted_MSE
 full=g[g.scope=='full'][['site','h','model','points']].rename(columns={'points':'full_points'});g=g.merge(full,on=['site','h','model']);g['target_fraction']=g.points/g.full_points;g.to_csv(R/'power_regime_summary.csv',index=False)
 q=d[(d.model=='Inverted')&(d.seed==42)];keys=['site','h','analysis','scope'];hs=q[q.support=='horizon_specific'][keys+['origins','points']];co=q[q.support=='common_H144'][keys+['origins','points']];z=hs.merge(co,on=keys,suffixes=('_horizon_specific','_common_H144'));z['origin_change']=z.origins_common_H144-z.origins_horizon_specific;z['point_change']=z.points_common_H144-z.points_horizon_specific;z.to_csv(R/'support_sensitivity.csv',index=False)
 c=json.loads(Path(paths).read_text());rows=[json.loads(p.read_text()) for p in Path(c['new_results']).rglob('*_completed.json')];assert len(rows)==10;pd.DataFrame(rows).to_csv(R/'ridge_run_summary.csv',index=False)
 print('Summaries:',len(g),'regime rows;',len(z),'support rows;',len(rows),'linear fits')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);run(p.parse_args().paths)

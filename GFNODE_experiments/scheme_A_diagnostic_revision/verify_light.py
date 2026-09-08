"""CSV-only independent arithmetic checks; not a replacement for raw/matrix replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def run():
 checks={};s=pd.read_csv(OUT/'ridge_solver_comparison.csv');assert len(s)==6;assert (s.max_prediction_difference_kW<1e-6).all();assert (s.saved_relative_normal_residual<1e-8).all();train=s[s.split=='train'];assert np.allclose(train.saved_objective,train.stable_objective,rtol=1e-10);checks['actual_matrix_record_consistency']=len(s)
 flags=pd.read_csv(OUT/'qcells_origin_flags.csv');support=pd.read_csv(OUT/'qcells_support.csv');hours=pd.read_csv(OUT/'qcells_hours.csv');counter=0
 for (split,h),g in flags.groupby(['split','h']):
  cause=np.select([g.boundary,g.future_missing_nonfinite,g.future_negative,g.origin_invalid],['boundary','future_missing_nonfinite','future_negative','origin_invalid'],default='retained');assert np.array_equal(cause,g.exclusive_cause);assert np.array_equal(cause=='retained',g.primary_keep)
  label='H12_support' if h==12 else 'common_H144';row=support[(support.split==split)&(support.support==label)&(support.h==h)].iloc[0];assert g.primary_keep.sum()==row.origins;assert hours[(hours.split==split)&(hours.support==label)].origins.sum()==row.origins;counter+=1
 checks['split_horizon_exclusion_reconciliation']=counter
 blocks=pd.read_csv(OUT/'ridge_block_sse.csv');results=pd.read_csv(OUT/'ridge_paired_intervals.csv');counter=0
 for key,g in blocks.groupby(['site','scope','block_hours']):
  g=g.sort_values('block');n=g['count'].to_numpy();nb=len(g);draw=np.random.default_rng(20260908).integers(nb,size=(2000,nb));counts=n[draw].sum(1);ok=counts>0;rm={};pts={}
  for name in ['Ridge A','Ridge B','Daily','Inverted42','Inverted43','Inverted44']:
   se=g[name+'_SSE'].to_numpy();rm[name]=np.sqrt(se[draw][ok].sum(1)/counts[ok]);pts[name]=np.sqrt(se.sum()/n.sum())
  q=results[(results.site==key[0])&(results.scope==key[1])&(results.block_hours==key[2])]
  for row in q.itertuples():
   refs=['Inverted42','Inverted43','Inverted44'] if row.reference=='Inverted mean' else [row.reference]
   contrast=np.mean([rm['Ridge B']-rm[r] for r in refs],axis=0);effect=np.mean([pts['Ridge B']-pts[r] for r in refs]);skill=np.mean([1-rm['Ridge B']/rm[r] for r in refs],axis=0);pointskill=np.mean([1-pts['Ridge B']/pts[r] for r in refs]);ci=np.quantile(contrast,[.025,.975]);sc=np.quantile(skill,[.025,.975]);assert np.allclose([effect,*ci,pointskill,*sc],[row.effect_kW,row.ci_low_kW,row.ci_high_kW,row.skill,row.skill_ci_low,row.skill_ci_high],rtol=1e-9,atol=1e-9);assert row.points==n.sum();counter+=1
 checks['paired_interval_rows_from_block_SSE']=counter
 grid=pd.read_csv(OUT/'expanded_alpha_validation.csv');chosen=pd.read_csv(OUT/'expanded_ridge_metrics.csv').drop_duplicates(['site','model']);assert len(chosen)==10 and len(grid)==130
 for r in chosen.itertuples():
  g=grid[(grid.site==r.site)&(grid.model==r.model)];assert np.allclose(sorted(g.alpha),[10.**k for k in range(-4,9)]);expected=g.sort_values(['validation_MSE_kW2','alpha'],ascending=[True,False]).iloc[0];assert r.selected_alpha==expected.alpha;assert np.isclose(r.validation_MSE_kW2,expected.validation_MSE_kW2)
  if r.site=='NIST_GROUND' and r.model=='Ridge':
   qr=pd.read_csv(OUT/'nist_QR_metric_sensitivity.csv');assert len(qr)==3;assert (qr.augmented_rank==504).all();assert (qr.RMSE_difference.abs()<1e-6).all();assert np.allclose(qr.QR_RMSE-qr.spectral_RMSE,qr.RMSE_difference,atol=1e-12)
  else:assert r.real_matrix_solver_maxdiff_kW<1e-5
 checks['expanded_grid_Validation_selections']=10
 original=pd.read_csv(HERE.parent/'scheme_A_review_extension/results/ridge_run_summary.csv');assert len(original)==10;assert original.selected_alpha.isin([.1,1000]).sum()==7;checks['original_endpoint_choices']=7
 decomposition=pd.read_csv(OUT/'NIST_DECOMPOSITION_RECHECK.csv')
 for seed,g in decomposition.groupby('seed'):
  full=g[g.scope=='full'].iloc[0];parts=g[g.scope!='full'];assert parts.points.sum()==full.points;assert np.isclose(parts.neural_SSE.sum(),full.neural_SSE);assert np.isclose(parts.Daily_SSE.sum(),full.Daily_SSE);assert np.isclose(parts.weighted_MSE_difference.sum(),full.weighted_MSE_difference)
 checks['NIST_per_seed_additive_MSE']=3
 a=json.loads((OUT/'FROZEN_HASH_BEFORE.json').read_text(encoding='utf8'));b=json.loads((OUT/'FROZEN_HASH_AFTER.json').read_text(encoding='utf8'));assert a==b;checks['immutable_source_inventory_record']=len(a)
 result=dict(checks=checks,failed=0,skipped=0,scope='Light package: recomputes CSV/block arithmetic and record consistency. Does not re-read heavy raw/checkpoint files or reconstruct real matrices. Full-path scripts and original evidence are required for those checks.');(OUT/'LIGHT_VERIFICATION.json').write_text(json.dumps(result,indent=2), encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':run()

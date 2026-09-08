"""Compare new point-array reductions with immutable historical metric CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_predictions import MODELS
H=Path(__file__).resolve().parent;d=pd.read_csv(H/'results/metrics_decomposition.csv');checks=0
for external,folder in [(False,'scheme_A_submission_correction'),(True,'scheme_A_multisite_extension')]:
 f=pd.read_csv(H.parent/folder/('metrics_per_seed.csv' if external else 'corrected_metrics.csv'))
 if not external:f=f[(f.statistic=='per_seed')&f.metric.isin(['RMSE','MAE','Bias'])]
 for x in f.itertuples():
  if x.model not in MODELS:continue
  scope='power-active' if x.scope=='daylight' else 'full';analysis='daily_matched' if 'daily' in x.analysis else 'primary';support='common_H144' if x.analysis=='secondary_h144_common' else 'horizon_specific'
  site=x.site if external else x.dataset;h=x.horizon if external else x.horizon_steps
  z=d[(d.site==site)&(d.model==MODELS[x.model])&(d.seed==int(x.seed))&(d.h==h)&(d.scope==scope)&(d.support==support)&(d.analysis==analysis)];assert len(z)==1;z=z.iloc[0]
  for metric in ['RMSE','MAE','bias'] if external else [x.metric.lower() if x.metric=='Bias' else x.metric]:
   expected=getattr(x,metric) if external else x.value;assert np.isclose(z[metric],expected,rtol=1e-6,atol=1e-7),(site,x.model,h,metric,z[metric],expected);checks+=1
  assert z.points==x.valid_target_count;checks+=1
result=dict(passed=checks,failed=0,skipped=0,source='Point reductions versus frozen CSV; no frozen CSV writes');(H/'results/FROZEN_COMPARISON.json').write_text(json.dumps(result,indent=2));print(result)

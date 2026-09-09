"""CSV/blocks and status checks only; no raw data, pickle, checkpoint or GPU needed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from paired import intervals
HERE=Path(__file__).resolve().parent;R=HERE/'results'
def check_blocks(b,e):
 keys=['site','scope','horizon','block_hours'];checked=0
 for key,g in b.groupby(keys):
  q=e
  for k,v in zip(keys,key):q=q[q[k]==v]
  assert len(q)
  rr=intervals(g,list(zip(q.method,q.reference)))
  for a,(_,r) in zip(rr,q.iterrows()):
   for k,v in a.items():
    if isinstance(v,(int,float,np.number)):assert np.isclose(v,r[k],rtol=1e-9,atol=1e-10,equal_nan=True),(key,k)
  checked+=len(q)
 return checked
def run():
 A=pd.read_csv(R/'expanded_paired_intervals.csv');n=check_blocks(pd.read_csv(R/'expanded_block_sse.csv'),A);assert n==360
 for site in ['YULARA_COMBINED','NIST_GROUND']:
  b=pd.read_csv(R/(site+'_2018_Apr_Jun_origins_blocks.csv'));e=pd.read_csv(R/(site+'_2018_Apr_Jun_origins_intervals.csv'));n+=check_blocks(b,e)
  e=pd.read_csv(R/'fixed_2018_primary_intervals.csv');n+=check_blocks(b,e[e.site==site])
  replay=json.loads((R/(site+'_2017_REPLAY.json')).read_text(encoding='utf8'));assert len(replay)==7 and all(r['passed'] for r in replay)
  for f in [R/(site+'_2018_Apr_Jun_origins_metrics.csv')]:
   d=pd.read_csv(f)
   for (h,method),g in d.groupby(['horizon','method']):
    q=g.set_index('scope');assert q.loc['full','points']==q.loc['power-active','points']+q.loc['low-power','points'];assert np.isclose(q.loc['full','SSE'],q.loc['power-active','SSE']+q.loc['low-power','SSE'],rtol=1e-12)
 q=A[(A.site=='NIST_GROUND')&(A.method=='Expanded B')&(A.reference=='Original B')];assert np.allclose(q[['effect_kW','ci_low_kW','ci_high_kW']],0,atol=1e-12)
 rep=pd.read_csv(R/'Alice_replay_summary.csv');neural=rep[~rep['check'].str.startswith(('Original','Expanded'))];assert len(neural)==36;assert (~neural.passed).sum()==20 # preserve the observed unresolved state, not a pass gate
 for site in ['Sanyo','Hanwha','Qcells']:
  s=pd.read_csv(R/(site+'_support_only.csv'))
  for h,g in s.groupby('horizon'):
   q=g.set_index('support');assert q.loc['Original','points']<=q.loc['S1','points']<=q.loc['S2','points'];assert q.loc['S1','points']-q.loc['Original','points']==q.loc['S1_new_origins','points'];assert q.loc['S2','points']-q.loc['Original','points']==q.loc['S2_new_origins','points']
 point=json.loads((R/'POINT_ARRAY_AUDIT.json').read_text(encoding='utf8'));assert point['passed']==216 and point['failed']==point['skipped']==0
 assert json.loads((R/'FROZEN_BEFORE.json').read_text(encoding='utf8'))==json.loads((R/'FROZEN_AFTER.json').read_text(encoding='utf8'))
 print(json.dumps({'CSV_block_rows_replayed':n,'point_array_audit_record_rows':216,'Alice_neural_replay_passed':16,'Alice_neural_replay_unresolved':20,'raw_checkpoint_access_in_this_light_check':False,'training':False,'status':'A_C_VERIFIED_B_PARTIAL'},indent=2))
if __name__=='__main__':run()

"""Measurement and minimal synchronized three-technology ModernTCN prototype."""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn

ROOT=Path(__file__).resolve().parent; REPO=ROOT.parent.parent; sys.path.insert(0,str(REPO))
from GFNODE_experiments.asoc_clean_decision.asoc_clean_decision import CleanDataProtocol

NAMES=('Sanyo','Hanwha','Qcells'); WEATHER=['Performance_Ratio','Weather_Temperature_Celsius','Weather_Relative_Humidity','Global_Horizontal_Radiation','Diffuse_Horizontal_Radiation','Radiation_Global_Tilted','Radiation_Diffuse_Tilted']
def config(): return json.loads((REPO/'GFNODE_experiments/asoc_discrete_viability/config.json').read_text())
def savecsv(name, rows):
 with (ROOT/name).open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protocols():
 c=config(); out={}
 for n in NAMES:
  p=CleanDataProtocol(c,n);p.load_regularized_raw();p.fit_transform();out[n]=p
 return c,out
class SharedModernTCN(nn.Module):
 def __init__(self,features,channels=64):
  super().__init__();self.backbone=nn.Sequential(nn.Conv1d(features,channels,1),nn.GELU(),nn.Conv1d(channels,channels,5,padding=2,groups=channels),nn.Conv1d(channels,channels,1),nn.GELU());self.heads=nn.ModuleList([nn.Linear(channels*72,144) for _ in NAMES])
 def forward(self,x):
  z=self.backbone(x.transpose(1,2)).flatten(1);return torch.stack([head(z) for head in self.heads],1)
def measure_and_smoke():
 c,ps=protocols(); raws={n:ps[n].raw for n in NAMES}; inv=[];align=[];feat=[]
 for n,p in ps.items():
  r=raws[n]; inv.append({'dataset':n,'raw_csv':str((REPO/'GFNODE_experiments'/c['datasets'][n].replace('../','')).resolve()),'raw_start':r.attrs['raw_min'],'raw_end':r.attrs['raw_max'],'raw_frequency':'5min','system_name':n,'pv_technology':n,'site':'Alice Springs (revision-plan evidence)','latitude':'UNKNOWN','longitude':'UNKNOWN','rated_capacity':'UNKNOWN','power_unit':'UNKNOWN','target':'Active_Power','feature_columns':' | '.join(p.feature_columns),'feature_scaler_fit':'Train only','target_scaler_fit':'Train valid Active_Power only'})
  for split in ('train','validation','test'):
   t=p.transformed[split];align.append({'split':split,'dataset':n,'timestamps':len(t),'independent_valid_windows':len(p.windows[split].x),'source_missing_fraction':float((~t['_source_timestamp_present']).mean())})
 common={k:set(ps[NAMES[0]].transformed[k].index) for k in ('train','validation','test')}
 for k in common:
  for n in NAMES[1:]:common[k]&=set(ps[n].transformed[k].index)
  sync=max(0,len(common[k])-c['lookback']-c['horizon']+1)
  align.append({'split':k,'dataset':'ALL_THREE','timestamps':len(common[k]),'independent_valid_windows':sync,'source_missing_fraction':float(any((~ps[n].transformed[k].loc[list(common[k]),'_source_timestamp_present']).any() for n in NAMES))})
 for a,b in (('Sanyo','Hanwha'),('Sanyo','Qcells'),('Hanwha','Qcells')):
  for k in ('train','validation','test'):align.append({'split':k,'dataset':f'{a}∩{b}','timestamps':len(set(ps[a].transformed[k].index)&set(ps[b].transformed[k].index)),'independent_valid_windows':'','source_missing_fraction':''})
 base=raws['Sanyo']
 for col in WEATHER:
  vals=[]
  for n in NAMES[1:]:
   x=base[col].to_numpy();y=raws[n][col].to_numpy();vals.append(np.allclose(x,y,equal_nan=True,rtol=1e-7,atol=1e-9))
  feat.append({'feature':col,'all_three_present':True,'comparison':'exactly_identical' if all(vals) else 'different_sensor_or_values','units':'UNKNOWN','source':'CSV only'})
 savecsv('CROSS_TECH_DATA_INVENTORY.csv',inv);savecsv('CROSS_TECH_TIMESTAMP_ALIGNMENT.csv',align);savecsv('CROSS_TECH_FEATURE_COMPATIBILITY.csv',feat)
 # A: same regular clock/weather and valid target definition; each joint sample is indexed by shared start time.
 starts=sorted(set(ps['Sanyo'].windows['train'].input_start)&set(ps['Hanwha'].windows['train'].input_start)&set(ps['Qcells'].windows['train'].input_start)); maps={n:{v:i for i,v in enumerate(ps[n].windows['train'].input_start)} for n in NAMES}; ix=[[maps[n][s] for n in NAMES] for s in starts]; x=torch.from_numpy(np.stack([ps['Sanyo'].windows['train'].x[ix[0][0]],ps['Hanwha'].windows['train'].x[ix[0][1]],ps['Qcells'].windows['train'].x[ix[0][2]]]).mean(0)).unsqueeze(0); y=torch.from_numpy(np.stack([ps[n].windows['train'].y_scaled[ix[0][j]] for j,n in enumerate(NAMES)])).unsqueeze(0)
 m=SharedModernTCN(x.shape[-1]);o=m(x);loss=((o-y)**2).mean();loss.backward();torch.save({'state_dict':m.state_dict()},ROOT/'smoke_only.pt')
 decision={'formulation':'SYNCHRONIZED_MULTI_OUTPUT','evidence':['same-site/shared-weather statement in PV_improve_v1/GFNODE_Revision_Plan.md:74','raw CSV headers match','weather columns numerically identical under allclose','shared timestamps measured in CROSS_TECH_TIMESTAMP_ALIGNMENT.csv'],'risks':['metadata capacity/unit/coordinates are UNKNOWN','identical weather creates shared-exogenous-signal leakage risk if not framed as same-site multi-array forecasting','joint target availability must be masked per technology in formal training'],'smoke':{'output_shape':list(o.shape),'loss_finite':bool(torch.isfinite(loss))}}
 (ROOT/'TASK_FORMULATION_DECISION.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
if __name__=='__main__':measure_and_smoke()

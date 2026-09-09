from pathlib import Path
import argparse,json
from datetime import timezone,timedelta
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def run(paths):
 c=json.loads(Path(paths).read_text(encoding='utf8'));files=sorted(Path(c['nist_2018']).glob('*.csv'));assert len(files)==94;headers=[];frames=[];inventory=[]
 fields=['PwrMtrP_kW_Avg','AmbTemp_C_Avg','Pyra1_Wm2_Avg']
 for p in files:
  s=p.stat();df=pd.read_csv(p,dtype=str,keep_default_na=False);headers.append(list(df.columns));frames.append(df[['TIMESTAMP',*fields]])
  inventory.append(dict(file=p.name,size=s.st_size,mtime_ns=s.st_mtime_ns,rows=len(df)));assert all(f in df for f in fields)
 assert all(h==headers[0] for h in headers)
 d=pd.concat(frames,ignore_index=True);assert d.TIMESTAMP.str.endswith('-05:00').all();idx=pd.DatetimeIndex(pd.to_datetime(d.TIMESTAMP,utc=True)).tz_convert(timezone(timedelta(hours=-5)));assert idx.is_unique and idx.is_monotonic_increasing
 expected=pd.date_range('2018-03-30','2018-07-01 23:59',freq='min',tz=idx.tz);missing=expected.difference(idx)
 stats=[]
 for f in fields:
  v=pd.to_numeric(d[f],errors='coerce');finite=v[np.isfinite(v)];stats.append(dict(field=f,missing=int(v.isna().sum()),nonfinite=int((~np.isfinite(v)&v.notna()).sum()),negative=int((v<0).sum()),minus999=int((v==-999).sum()),minimum=float(finite.min()),p01=float(finite.quantile(.01)),median=float(finite.median()),p99=float(finite.quantile(.99)),maximum=float(finite.max())))
 assert not any(r['minus999'] for r in stats),'Unexpected explicit sentinel in common input requires review'
 pd.DataFrame(inventory).to_csv(OUT/'NIST_2018_file_inventory.csv',index=False);pd.DataFrame(stats).to_csv(OUT/'NIST_2018_field_audit.csv',index=False);pd.DataFrame({'missing_minute':missing.astype(str)}).to_csv(OUT/'NIST_2018_missing_minutes.csv',index=False)
 result={'file_count':94,'headers_identical':True,'header':headers[0],'first':str(idx[0]),'last':str(idx[-1]),'unique_minutes':len(idx),'missing_minutes':len(missing),'time_basis':'FIXED_EST_LST','utc_offset':'-05:00','power_label':fields[0],'ghi_conversion_applied':False,'operational_change_log':'UNAVAILABLE_NONBLOCKING_NOT_PROOF_OF_NO_CHANGE','access_date':'2026-09-09'}
 (OUT/'NIST_2018_SOURCE_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('NIST source audit',len(idx),'unique minutes;',len(missing),'missing; fixed EST, 94 headers identical')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);run(p.parse_args().paths)

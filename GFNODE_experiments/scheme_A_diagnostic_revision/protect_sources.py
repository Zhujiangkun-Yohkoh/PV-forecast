"""Explicit-scope immutable-evidence SHA-256 inventory; no disk discovery."""
from pathlib import Path
import json,hashlib,argparse,csv
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
 return h.hexdigest()
def inventory(path):
 c=json.loads(Path(path).read_text()); ext=json.loads(Path(c['external_paths']).read_text()); items=[]
 for key in ['alice_results','external_results','new_results']:
  root=Path(c[key])
  items += [(key+'/'+p.relative_to(root).as_posix(),p) for p in root.rglob('*') if p.is_file() and p.suffix not in ['.pyc','.zip']]
 items += [('raw/Alice/'+Path(v).name,Path(v)) for v in c['alice_raw'].values()]
 y=Path(ext['YULARA_RAW_FILE']);items.append(('raw/Yulara/'+y.name,y));n=Path(ext['NIST_GROUND_2017_DIRECTORY'])
 items += [('raw/NIST/'+p.relative_to(n).as_posix(),p) for p in n.rglob('*') if p.is_file()]
 for name in ['scheme_A_submission_correction','scheme_A_multisite_extension','scheme_A_review_extension']:
  root=HERE.parent/name
  items += [('frozen/'+name+'/'+p.relative_to(root).as_posix(),p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 return {k:dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=digest(p)) for k,p in items}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--paths',required=True);a.add_argument('--phase',choices=['before','after'],required=True);v=a.parse_args();o=HERE/'results';o.mkdir(exist_ok=True);x=inventory(v.paths)
 if v.phase=='after':
  old=json.loads((o/'FROZEN_HASH_BEFORE.json').read_text());assert x==old,'Frozen evidence hash/stat mismatch'
 (o/('FROZEN_HASH_'+v.phase.upper()+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8');print('Protected',len(x),'files:',v.phase,flush=True)

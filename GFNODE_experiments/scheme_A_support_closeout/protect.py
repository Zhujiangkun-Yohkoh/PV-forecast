from pathlib import Path
import sys,json,argparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scheme_A_diagnostic_revision'))
from protect_sources import inventory,digest
p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--phase',choices=['before','after'],required=True);a=p.parse_args();c=json.loads(Path(a.paths).read_text(encoding='utf8'));v=inventory(a.paths)
for label,root in [('previous_forward',c['destination']),('nist_2018',c['nist_2018']),*[(f'expanded/{k}',x) for k,x in c['expanded_results'].items()]]:
 for f in Path(root).rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts:v[label+'/'+f.relative_to(root).as_posix()]={'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'sha256':digest(f)}
if a.phase=='after':assert v==json.loads((HERE/'results/FROZEN_BEFORE.json').read_text(encoding='utf8'))
(HERE/'results'/('FROZEN_'+a.phase.upper()+'.json')).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8');print(len(v),'protected files',a.phase,flush=True)

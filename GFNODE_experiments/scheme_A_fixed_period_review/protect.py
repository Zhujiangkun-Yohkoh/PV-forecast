from pathlib import Path
import sys,json,argparse
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scheme_A_diagnostic_revision'))
from protect_sources import inventory,digest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--phase',choices=['before','after'],required=True);v=p.parse_args();c=json.loads(Path(v.paths).read_text(encoding='utf8'));x=inventory(v.paths)
 for site,folder in c['expanded_results'].items():
  for f in Path(folder).glob('*'):
   if f.is_file():x['expanded/'+site+'/'+f.name]={'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'sha256':digest(f)}
 out=HERE/'results';out.mkdir(exist_ok=True)
 if v.phase=='after':assert x==json.loads((out/'FROZEN_BEFORE.json').read_text(encoding='utf8'))
 (out/('FROZEN_'+v.phase.upper()+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8');print(len(x),'protected files',v.phase)

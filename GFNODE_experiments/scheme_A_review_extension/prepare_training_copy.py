"""Prepare a standalone, explicit-path training copy; never starts training."""
from pathlib import Path
import argparse,json,shutil
HERE=Path(__file__).resolve().parent
def prepare(paths,output):
 c=json.loads(Path(paths).read_text());out=Path(output).resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit('Output must be a new empty directory; existing evidence is never overwritten.')
 out.mkdir(parents=True,exist_ok=True)
 for folder in ['scheme_A_submission_correction','scheme_A_multisite_extension']:
  dest=out/'GFNODE_experiments'/folder;dest.mkdir(parents=True)
  for p in (HERE.parent/folder).iterdir():
   if p.suffix in ['.py','.json','.md','.csv'] and p.is_file():shutil.copy2(p,dest/p.name)
 raw={k:str(Path(v).resolve()) for k,v in c['alice_raw'].items()};ext=json.loads(Path(c['external_paths']).read_text());ext={k:str(Path(v).resolve()) for k,v in ext.items()};(out/'external_paths.json').write_text(json.dumps(ext,indent=2));(out/'alice_paths.json').write_text(json.dumps(raw,indent=2))
 wrapper='''from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'GFNODE_experiments/scheme_A_submission_correction'))
sys.path.insert(0,str(ROOT/'GFNODE_experiments/scheme_A_multisite_extension'))
import run_corrected_benchmark as alice
import run_multisite_benchmark as external
raw=json.loads((ROOT/'alice_paths.json').read_text())
alice.resolve_data_path=lambda cfg,dataset:Path(raw[dataset])
alice.RESULTS=ROOT/'new_alice_results'
external.RESULTS=ROOT/'new_external_results'
# Standalone copy uses its included, provenance-documented M1-R config.
# The historical source retains its git-show check unchanged.
external.frozen_config=external.a.config
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','train-alice','train-external']);args=p.parse_args()
 if args.action=='inspect':
  assert all(Path(v).is_file() for v in raw.values())
  external.a.validate_paths(ROOT/'external_paths.json')
  assert len(external.frozen_config()['run_matrix'])==24
  print('Imports, 3 explicit Alice CSV paths, external paths and 24-run config inspected; no training.')
 elif args.action=='train-alice':alice.run_all()
 else:external.run(str(ROOT/'external_paths.json'))
'''
 (out/'reproduce.py').write_text(wrapper)
 (out/'PROVENANCE.md').write_text('Standalone preparation from Scheme A review branch. Original M1-R config and compact model code are copied without numerical changes. Wrapper replaces private path discovery and historical git-show lookup with explicitly supplied files and the included configuration. Historical source check remains present in copied source. Full neural training has NOT been retested this review. Use reproduce.py inspect first; train-alice and train-external intentionally start 36/24 new fits in separate output directories.\n')
 print('Prepared; no training:',out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.paths,a.output)

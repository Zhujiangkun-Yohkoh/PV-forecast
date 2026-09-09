"""Self-contained review entry points. No historical git object or private path fallback."""
from pathlib import Path
import argparse,json,sys,runpy
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load_paths(file):
 p=Path(file).resolve()
 if not p.is_file():raise SystemExit('Missing explicit path JSON. See ENVIRONMENT_AND_REPRODUCTION.md; no disk search is performed.')
 c=json.loads(p.read_text(encoding='utf8'))
 required=[Path(c[k]) for k in ['alice_results','external_results','external_paths']]+[Path(v) for v in c['alice_raw'].values()]
 missing=[str(x) for x in required if not x.exists()]
 if missing:raise SystemExit('Missing required data/artifacts: '+', '.join(missing))
 for key,pat,n in [('alice_results','*/test_H144.npz',36),('external_results','*/test_predictions.npz',24)]:
  if len(list(Path(c[key]).glob(pat)))!=n:raise SystemExit(f'Missing or extra prediction artifacts: {key} requires {n}')
 return c
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['figures','metrics','verify','inspect']);p.add_argument('--paths');args=p.parse_args()
 if args.action=='figures':
  try:import matplotlib
  except ImportError:raise SystemExit('Install matplotlib, numpy and pandas in your review environment.')
  runpy.run_path(str(HERE/'build_review_figures.py'),run_name='__main__');return
 if not args.paths:raise SystemExit('--paths is required; raw files/checkpoints are not inferred from a private worktree.')
 c=load_paths(args.paths)
 if args.action=='inspect':print('Explicit sources present; 36 Alice and 24 external prediction groups. No model loading or training.');return
 sys.path.insert(0,str(HERE))
 if args.action=='metrics':
  from analyze_predictions import run
  run(args.paths)
 else:
  from verify_review import run
  run(args.paths)
if __name__=='__main__':main()

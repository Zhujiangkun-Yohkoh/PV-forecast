"""Private full review ZIP with evidence, explicit paths, CRC and extraction checks."""
from pathlib import Path
import argparse,csv,json,shutil,subprocess,zipfile,datetime,filecmp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def run(paths,destination):
 c=json.loads(Path(paths).read_text());external=json.loads(Path(c['external_paths']).read_text());stamp=datetime.datetime.now().strftime('%Y-%m-%d_%H%M');name=f'Scheme_A_Full_Project_Review_{stamp}_v2';dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
 final=dest/(name+'.zip')
 if final.exists():raise RuntimeError('Never overwrite an existing review ZIP')
 stage=ROOT/'.local'/name;extract=ROOT/'.local'/(name+'_extraction_check');stage.mkdir();extract.mkdir();sources={};roles={}
 def copy(src,relative,role,critical=True):
  src=Path(src);s=src.stat();sources[str(src)]=(s.st_size,s.st_mtime_ns);p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);roles[relative]=(role,critical)
 prefixes=['GFNODE_experiments/scheme_A_submission_correction/','GFNODE_experiments/scheme_A_multisite_extension/','GFNODE_experiments/scheme_A_review_extension/','manuscript/clean_pv_benchmark/']
 names=['START_HERE_REVIEW.md','PROJECT_REPORT_CN.md','REVIEW_RESPONSE_MATRIX.md','ENVIRONMENT_AND_REPRODUCTION.md','JOURNAL_STRATEGY_CN.md','REVIEW_VALIDATION.md','FIGURE_QA.md','.gitignore']
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
 for rel in tracked:
  if not rel or not (rel in names or any(rel.startswith(x) for x in prefixes)):continue
  if any(x in Path(rel).parts for x in ['__pycache__','.local','.git']):continue
  if Path(rel).suffix.lower() in ['.pt','.npz','.pkl','.zip','.pyc'] or Path(rel).name=='.env':raise RuntimeError('Unexpected tracked binary/private evidence: '+rel)
  copy(ROOT/rel,'project/'+rel,'project source / manuscript / frozen or review result')
 for key,label in [('alice_results','alice'),('external_results','external'),('new_results','review_ridge')]:
  source=Path(c[key]);files=[p for p in source.rglob('*') if p.is_file()]
  for p in files:
   if '__pycache__' in p.parts or p.suffix in ['.pyc','.zip'] or p.name=='.env':continue
   copy(p,'evidence/'+label+'/'+p.relative_to(source).as_posix(),'local evidence: '+label)
 for site,p in c['alice_raw'].items():copy(p,'data/Alice/'+Path(p).name,'authorized raw Alice source')
 y=Path(external['YULARA_RAW_FILE']);copy(y,'data/Yulara/'+y.name,'provider raw Yulara source')
 n=Path(external['NIST_GROUND_2017_DIRECTORY'])
 for p in n.rglob('*'):
  if p.is_file():copy(p,'data/NIST/'+p.relative_to(n).as_posix(),'provider NIST source / metadata')
 # Complete figure-only handoff, also runnable without the original worktree.
 for p in (ROOT/'manuscript/clean_pv_benchmark/review_figures').iterdir():
  if p.is_file():copy(p,'figure_handoff/manuscript/clean_pv_benchmark/review_figures/'+p.name,'figure handoff',False)
 for p in (HERE/'results').glob('*.csv'):copy(p,'figure_handoff/GFNODE_experiments/scheme_A_review_extension/results/'+p.name,'figure source',False)
 copy(HERE/'build_review_figures.py','figure_handoff/GFNODE_experiments/scheme_A_review_extension/build_review_figures.py','plot code',False)
 copy(HERE.parent/'scheme_A_submission_correction/corrected_metrics.csv','figure_handoff/GFNODE_experiments/scheme_A_submission_correction/corrected_metrics.csv','historical plot source',False)
 (stage/'figure_handoff/README.md').write_text('Run from this folder: python GFNODE_experiments/scheme_A_review_extension/build_review_figures.py. Requires numpy,pandas,matplotlib. Full current figures, CSVs and portable code included. S2/S7 panel label aesthetics may be adjusted after journal sizing without changing data or selections.\n')
 config=dict(alice_results='../evidence/alice',external_results='../evidence/external',new_results='../evidence/review_ridge',alice_raw={s:'../data/Alice/'+Path(p).name for s,p in c['alice_raw'].items()},external_paths='external_paths.json')
 (stage/'project/review_paths.json').write_text(json.dumps(config,indent=2));(stage/'project/external_paths.json').write_text(json.dumps(dict(YULARA_RAW_FILE='../data/Yulara/'+y.name,NIST_GROUND_2017_DIRECTORY='../data/NIST'),indent=2))
 (stage/'START_HERE_REVIEW.md').write_text('# Complete private Scheme A review package\n\nStart with [project report](project/PROJECT_REPORT_CN.md), [response matrix](project/REVIEW_RESPONSE_MATRIX.md), [main PDF](project/manuscript/clean_pv_benchmark/main.pdf), [Supplement](project/manuscript/clean_pv_benchmark/supplementary.pdf), and [journal strategy](project/JOURNAL_STRATEGY_CN.md).\n\nproject contains the complete relevant Scheme A development/manuscript files; evidence contains 60 neural checkpoints and prediction groups, 10 Ridge prediction/coefficient groups and available processors; data contains 369 authorized raw CSVs. figure_handoff is self-contained for plotting.\n\nUse a short extraction path on Windows. For relative-path commands change directory to project, then follow ENVIRONMENT_AND_REPRODUCTION.md using review_paths.json. This is private author review, not a public release or data license. Historical Alice processors were not saved as separate files; raw inputs/config/code and new Ridge processors are supplied, without pretending the latter are original neural artifacts.\n')
 git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT).decode().strip()
 (stage/'GIT_DELIVERY.md').write_text('# Git delivery\n\nBranch: '+git('branch','--show-current')+'\n\nStart commit: 8070d6b8fc9f11f596ef9990ab1b8005321160ae\n\nLocal commit: '+git('rev-parse','HEAD')+'\n\nRemote tracking commit: '+git('rev-parse','origin/manuscript/clean-pv-benchmark-multisite-revision')+'\n\nTree: '+git('rev-parse','HEAD^{tree}')+'\n\nPR: https://github.com/Zhujiangkun-Yohkoh/PV-forecast/pull/19\n\nWorktree status at packaging:\n```\n'+git('status','--short')+'\n```\nRemote identity and push status are reported in the delivery message; no rebase, force push or merge.\n')
 def manifest():
  with (stage/'PACKAGE_MANIFEST.csv').open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.writer(f);w.writerow(['relative_path','role','size','critical'])
   for p in sorted(stage.rglob('*')):
    if p.is_file() and p.name!='PACKAGE_MANIFEST.csv':
     rel=p.relative_to(stage).as_posix();role,critical=roles.get(rel,('package navigation/verification',True));w.writerow([rel,role,p.stat().st_size,critical])
 def archive(path):
  with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
   for p in stage.rglob('*'):
    if p.is_file():z.write(p,p.relative_to(stage).as_posix())
 def verify(path):
  with zipfile.ZipFile(path) as z:
   assert z.testzip() is None;assert all(not Path(s).is_absolute() and '..' not in Path(s).parts for s in z.namelist());z.extractall(extract)
  files=[p for p in stage.rglob('*') if p.is_file()]
  for p in files:assert filecmp.cmp(p,extract/p.relative_to(stage),shallow=False),p
  for src,s in sources.items():
   p=Path(src);assert (p.stat().st_size,p.stat().st_mtime_ns)==s,src
  with (extract/'PACKAGE_MANIFEST.csv').open(encoding='utf-8-sig') as f:
   rows=list(csv.DictReader(f))
  for row in rows:assert (extract/row['relative_path']).stat().st_size==int(row['size'])
  return len(files)
 manifest();probe=ROOT/'.local'/(name+'_precheck.zip');archive(probe);count=verify(probe)
 checkpoints=list((stage/'evidence').rglob('best_validation.pt'));pred=list((stage/'evidence').rglob('*.npz'));raw=list((stage/'data').rglob('*.csv'));assert len(checkpoints)==60 and len(raw)==369
 (stage/'PACKAGE_VERIFICATION.md').write_text(f'''# Package verification

The full preliminary archive was CRC-checked, extracted to a fresh directory and all {count} files compared byte-for-byte with staged sources. Manifest sizes matched. Every source file size/mtime_ns remained unchanged. The final archive adds this record and refreshes the manifest; the same complete CRC/extraction/content/source-stat checks are executed again before successful delivery.

Included: 60 neural best checkpoints; 36 Alice test_H144 NPZs; 24 external test_predictions NPZs; 10 Ridge prediction NPZs; 10 Ridge coefficient NPZs; 369 raw CSVs; all locally available completion records, learning histories and processors. Total NPZ files: {len(pred)}. Original Alice neural preprocessors were not separately saved; raw inputs and preprocessing code are included, and the new Ridge objects are identified separately. This is not an omitted known processor file.

Excluded: virtual environments, rebuildable caches, credentials/.env, git database, unrelated project routes, old ZIPs. No known locally available Scheme A prediction/checkpoint evidence was excluded because of size. Provider operation-log / metadata gaps are described in the original protocol; unavailable documents were not invented.

Manifest excludes its own self-referential row. ZIP uses standard ZIP64-compatible deflate and needs no proprietary split utility. Test extraction is packaging verification, not a full neural training reproduction. Scientific rechecks and unrerun historical suites are listed in project/REVIEW_VALIDATION.md.\n''')
 manifest();archive(final);count=verify(final)
 result=dict(zip=str(final),bytes=final.stat().st_size,files=count,neural_checkpoints=60,raw_csv=369,CRC_passed=True,full_trial_extraction=True,byte_comparison_passed=True,source_size_mtime_unchanged=True,extract_path=str(extract),stage_path=str(stage),complete_available_local_evidence=True)
 final.with_suffix('.verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--destination',required=True);a=p.parse_args();run(a.paths,a.destination)

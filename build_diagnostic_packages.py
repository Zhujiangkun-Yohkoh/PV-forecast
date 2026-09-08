"""Create three complete-version archives; explicit evidence only, no disk discovery."""
from pathlib import Path
import argparse,csv,io,json,hashlib,subprocess,zipfile,datetime,sys
ROOT=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(paths,expanded,destination):
 dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);stamp=datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
 git=lambda *x:subprocess.check_output(['git',*x],cwd=ROOT).decode().strip()
 commit=git('rev-parse','HEAD');state=git('status','--short');assert not state,'Commit source changes before packaging'
 cfg=json.loads(Path(paths).read_text());ext=json.loads(Path(cfg['external_paths']).read_text());project={};full={};stats={}
 def add(mapping,p,rel,role):
  p=Path(p);assert p.is_file(),p;assert rel not in mapping,rel;assert not any(x in p.parts for x in ['__pycache__','.git']);assert p.suffix.lower() not in ['.zip','.pyc'] and p.name!='.env';s=p.stat();stats[str(p)]=(s.st_size,s.st_mtime_ns);mapping[rel]=(p,role)
 prefixes=['GFNODE_experiments/scheme_A_submission_correction/','GFNODE_experiments/scheme_A_multisite_extension/','GFNODE_experiments/scheme_A_review_extension/','GFNODE_experiments/scheme_A_diagnostic_revision/','manuscript/clean_pv_benchmark/']
 for rel in git('ls-files','-z').split('\0'):
  if not rel:continue
  if any(rel.startswith(x) for x in prefixes) or (len(Path(rel).parts)==1 and Path(rel).suffix in ['.md','.py','.json','.txt']):add(project,ROOT/rel,'project/'+rel,'project source / manuscript / versioned evidence')
 full.update(project)
 for key,label in [('alice_results','alice'),('external_results','external'),('new_results','original_ridge')]:
  src=Path(cfg[key])
  for p in src.rglob('*'):
   if p.is_file() and p.suffix not in ['.zip','.pyc'] and '__pycache__' not in p.parts:add(full,p,'evidence/'+label+'/'+p.relative_to(src).as_posix(),'frozen '+label+' evidence')
 for src in map(Path,expanded):
  for p in src.rglob('*'):
   if p.is_file() and p.suffix in ['.npz','.json']:
    rel=p.relative_to(src).as_posix();add(full,p,'evidence/expanded_ridge/'+rel,'post hoc expanded Ridge / QR diagnostic')
 for s,p in cfg['alice_raw'].items():add(full,p,'data/Alice/'+Path(p).name,'authorized raw Alice CSV')
 y=Path(ext['YULARA_RAW_FILE']);add(full,y,'data/Yulara/'+y.name,'provider raw Yulara CSV');n=Path(ext['NIST_GROUND_2017_DIRECTORY'])
 for p in n.rglob('*'):
  if p.is_file():add(full,p,'data/NIST/'+p.relative_to(n).as_posix(),'provider NIST raw / metadata')
 assert sum(k.endswith('best_validation.pt') for k in full)==60
 assert sum(k.startswith('data/') and k.endswith('.csv') for k in full)==369
 assert sum(k.startswith('evidence/expanded_ridge/') and k.endswith('_completed.json') for k in full)==10
 assert sum(k.startswith('evidence/original_ridge/') and k.endswith('_predictions.npz') for k in full)==10
 # Full source manifest is also carried by lite so omitted heavy files are explicit.
 print('Hashing',len(full),'full payload files',flush=True);meta={}
 for rel,(p,role) in full.items():meta[rel]=(role,p.stat().st_size,sha(p),True)
 def manifest(items):
  out=io.StringIO();w=csv.writer(out);w.writerow(['relative_path','role','size','sha256','critical'])
  for k,v in sorted(items.items()):w.writerow([k,*v])
  return out.getvalue().encode('utf-8-sig')
 gittext=f'# Version\n\nBranch: {git("branch","--show-current")}\nStart: 14942f4b18bc85d43b168cafc5e7bad81eb9d311\nCommit: {commit}\nTree: {git("rev-parse","HEAD^{tree}")}\nRemote tracking: {git("rev-parse","origin/manuscript/clean-pv-benchmark-multisite-revision")}\nPR: https://github.com/Zhujiangkun-Yohkoh/PV-forecast/pull/19\nWorktree at archive creation: clean.\n'
 configs={'alice_results':'../evidence/alice','external_results':'../evidence/external','new_results':'../evidence/original_ridge','alice_raw':{s:'../data/Alice/'+Path(p).name for s,p in cfg['alice_raw'].items()},'external_paths':'external_paths.json'}
 lite={k:v for k,v in project.items() if '/figures/' not in k and '/review_figures/' not in k and not k.endswith('build_figures_legacy.py')}
 figure={k:v for k,v in project.items() if '/diagnostic_figures/' in k or k.endswith('/scheme_A_diagnostic_revision/build_figures.py') or k.endswith('/scheme_A_diagnostic_revision/additional_figures.py') or ('/scheme_A_review_extension/results/' in k and k.endswith('.csv')) or ('/scheme_A_diagnostic_revision/results/' in k and k.endswith('.csv')) or k.endswith('/scheme_A_submission_correction/corrected_metrics.csv') or k.endswith('/FIGURE_QA.md')}
 outcomes=[]
 for kind,source in [('Full_Project_Review',full),('Review_Lite',lite),('Figure_Handoff',figure)]:
  name=f'Scheme_A_{kind}_{stamp}';target=dest/(name+'.zip');extract=dest/(name+'_trial');assert not target.exists() and not extract.exists();extra={}
  extra['GIT_DELIVERY.md']=gittext.encode();extra['FULL_EVIDENCE_MANIFEST.csv']=manifest(meta)
  extra['START_HERE_REVIEW.md']=('''# Scheme A diagnostic review package

Read project/PROJECT_REPORT_CN.md, the P01/P02/P03 reports under project/GFNODE_experiments/scheme_A_diagnostic_revision, and the latest paper under project/manuscript/clean_pv_benchmark. The figure-only package contains its complete plotting inputs and QA instead of the manuscript and heavy evidence. Run plotting from project with python GFNODE_experiments/scheme_A_diagnostic_revision/build_figures.py (numpy/pandas/matplotlib).

Full: all locally available raw data, 60 neural checkpoint/prediction groups, original 10 Ridge groups and processors, expanded 10 Ridge groups plus the QR diagnostic. Original Alice neural processors were not saved separately; their absence is historical, not silently replaced by newer processors. Missing provider logs/metadata remain documented limitations.

Lite: no full raw/checkpoint/NPZ objects. FULL_EVIDENCE_MANIFEST.csv gives every heavy relative location, size and SHA in the matching full archive. It supports block-SSE and table arithmetic, not actual-matrix replay or neural forward reproduction. Figure-only validates figure formats/data mapping; no scientific fit verification claim.

Private reviewer package, not a public data redistribution license. Excluded: old ZIPs, virtual environments, credentials, caches, temporary design-matrix pickle, unrelated project routes and git database. Full training was not rerun. Use short extraction paths on Windows. Never execute a training entry point as a packaging test.
''').encode()
  if kind=='Full_Project_Review':
   extra['project/review_paths.json']=json.dumps(configs,indent=2).encode();extra['project/external_paths.json']=json.dumps({'YULARA_RAW_FILE':'../data/Yulara/'+y.name,'NIST_GROUND_2017_DIRECTORY':'../data/NIST'},indent=2).encode();extra['project/expanded_paths.json']=json.dumps({'expanded_results':'../evidence/expanded_ridge'},indent=2).encode()
  extra['PACKAGE_VERIFICATION.md']=('''# Verification scope

The builder checks ZIP CRC, extracts every file into a new directory, compares every manifest SHA-256 and size, and runs the applicable lightweight verification in that extracted copy. Any failure aborts delivery. The adjacent .verification.json contains the actual final outcomes and .sha256 verifies the entire ZIP. Manifest excludes itself to avoid self-reference; the ZIP SHA covers it too.

Full and Lite run verify_light.py, which independently reconstructs 270 paired interval rows from block SSE/counts and checks support attribution and Validation alpha selections. This is not checkpoint replay or full neural training. Figure package checks all 20 PDF/SVG/PNG/source/caption/alt groups; the complete figures were generated and visually inspected in the original revision before packaging. No training runs occur during extraction checks.
''').encode()
  entries={k:meta[k] for k in source}
  for k,b in extra.items():entries[k]=('package navigation / relative configuration',len(b),hashlib.sha256(b).hexdigest(),True)
  extra['PACKAGE_MANIFEST.csv']=manifest(entries)
  print('Writing',target.name,flush=True)
  with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
   for k,(p,_) in source.items():z.write(p,k)
   for k,b in extra.items():z.writestr(k,b)
  print('CRC and full trial extraction',target.name,flush=True)
  with zipfile.ZipFile(target) as z:
   assert z.testzip() is None;assert all(not Path(k).is_absolute() and '..' not in Path(k).parts for k in z.namelist());z.extractall(extract)
  for k,v in entries.items():p=extract/k;assert p.stat().st_size==v[1] and sha(p)==v[2],k
  if kind!='Figure_Handoff':
   process=subprocess.run([sys.executable,'-B','GFNODE_experiments/scheme_A_diagnostic_revision/verify_light.py'],cwd=extract/'project',capture_output=True,text=True);assert process.returncode==0,process.stdout+process.stderr;check=process.stdout
  else:
   figs=extract/'project/manuscript/clean_pv_benchmark/diagnostic_figures';names=[p.stem for p in figs.glob('fig*.pdf')];assert len(names)==20
   for stem in names:
    for ending in ['.pdf','.svg','.png','_data.csv','_caption.txt','_alt.txt']:assert (figs/(stem+ending)).stat().st_size>0
   check='20 figure groups / six required assets per group; no heavy prediction replay.'
  for p,s in stats.items():assert (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==s,p
  digest=sha(target);target.with_suffix('.sha256').write_text(digest+'  '+target.name+'\n');result=dict(kind=kind,path=str(target),commit=commit,sha256=digest,bytes=target.stat().st_size,manifest_files=len(entries),CRC=True,fresh_full_extraction=True,all_manifest_hashes=True,light_validation=check,source_size_mtime_unchanged=True,neural_training=False)
  target.with_suffix('.verification.json').write_text(json.dumps(result,indent=2));outcomes.append(result);print('Verified',target.name,round(target.stat().st_size/1024**2,2),'MiB',flush=True)
 (dest/('DELIVERY_'+stamp+'.json')).write_text(json.dumps(outcomes,indent=2));print(json.dumps(outcomes,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--expanded',nargs='+',required=True);p.add_argument('--destination',required=True);a=p.parse_args();run(a.paths,a.expanded,a.destination)

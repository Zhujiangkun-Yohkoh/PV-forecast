"""Synchronize current captions, metadata, counts and review navigation."""
from pathlib import Path
import json,re
import pandas as pd
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=ROOT/'manuscript/clean_pv_benchmark';F=P/'diagnostic_figures'
def words(t):
 t=re.sub(r'(?<!\\)%[^\n]*','',t);t=re.sub(r'\\[a-zA-Z]+\*?(?:\[[^]]*\])?', ' ',t);return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",t))
text=(P/'main.tex').read_text();abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',text,re.S).group(1)
def expand(t):
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand((P/(m[1]+('.tex' if not m[1].endswith('.tex') else ''))).read_text()),t)
expanded=expand(text);figblocks=re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',expanded,re.S);tabs=re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',expanded,re.S)
body=expanded[expanded.index('\\section{Introduction}'):expanded.index('\\section*{Supplementary Material}')]
for b in figblocks+tabs:body=body.replace(b,'')
body=re.sub(r'\\(cite|ref|label)\{[^}]*\}','',body)
counts={'abstract':words(abstract),'body_excluding_float_environments':words(body),'figure_captions':sum(words(b) for b in figblocks),'table_environments_including_captions_headers_values':sum(words(b) for b in tabs),'references_bbl':words((ROOT/'.local/tex-diag-main/main.bbl').read_text()),'main_figures':len(figblocks),'main_tables':len(tabs),'method':'Token regex over TeX after command stripping and input expansion. Body excludes floats; figure captions include their source labels/paths tokens (approximate), table count includes headers and numbers; BBL separately. Not PDF full-text word count and not publisher TeXcount.'}
(HERE/'results/WORD_COUNTS.json').write_text(json.dumps(counts,indent=2));assert counts['abstract']<=250
records=json.loads((F/'FIGURE_CAPTIONS_AND_ALT.json').read_text());monthly=next(x for x in records if x['name']=='figS6_monthly');monthly['caption']=monthly['caption'].split(' O denotes')[0]+' O denotes origins and N scored origin--lead target pairs, printed for each month.';(F/'FIGURE_CAPTIONS_AND_ALT.json').write_text(json.dumps(records,indent=2));(F/'figS6_monthly_caption.txt').write_text(monthly['caption'])
def tex(s):
 return s.replace('%','\\%').replace('&','\\&').replace('²','$^2$').replace('10^-4','$10^{-4}$').replace('10^8','$10^8$')
sp=(P/'supplementary_diagnostic_figures.tex').read_text();sp=re.sub(r'\\caption\{Monthly errors.*?\}\n',lambda _:r'\caption{'+tex(monthly['caption'])+'}\n',sp);(P/'supplementary_diagnostic_figures.tex').write_text(sp,encoding='utf8')
pkg=P/'submission_package'
(pkg/'FIGURE_ALT_TEXT.txt').write_text('\n\n'.join(r['name']+'\nCaption: '+r['caption']+'\nAlt: '+r['alt'] for r in records),encoding='utf8')
(pkg/'FILE_UPLOAD_MANIFEST.md').write_text(f'''# Current author-review files (2026-09-09)

Main: 12 pages, {counts['abstract']} words by documented token rule, 6 figures and 5 tables. Supplement: 43 pages, 14 figures and 18 historical tables. Latest plot source is diagnostic_figures (20 groups: PDF, SVG, 320dpi PNG, CSV, caption, alt). Older review_figures/figures remain historical. PDF copies in this directory are identical to paper-root copies.

Sources: main.tex, supplementary.tex, supplementary_diagnostic_front.tex, supplementary_diagnostic_figures.tex, supplementary_historical_body.tex, references.bib, all included table TeX files. New diagnostics: scheme_A_diagnostic_revision; old frozen/reference evidence remains separately named. Final journal formatting, author signoff and license/public URL are not supplied by this review. No upload performed. Full private-review evidence is not automatically a journal attachment or authorized public redistribution.
''',encoding='utf8')
(pkg/'SUBMISSION_METADATA_CHECKLIST.md').write_text(f'''# Current metadata checklist (2026-09-09)

- Title: Reference Information and Target-Power Regimes in Multi-Window Photovoltaic Forecasting.
- Empirical original research; reference information, support selection and target-power regimes, not a new architecture.
- Candidate route Renewable Energy → Solar Energy → JRSE. CAS edition/major/subcategory require institutional verification; not a verified CAS ordering.
- Main 12 pages, {counts['abstract']} abstract tokens, 6 figures/5 tables; Supplement 43 pages, 14 figures/18 tables. See WORD_COUNTS.json for exact counting scope.
- 60 frozen neural runs; 10 original deterministic Ridge fits plus 10 selected uniform-grid sensitivity fits. No new neural training.
- Current author fields retained. Authors must approve final manuscript, declarations, journal, license and public code details; nothing filled or submitted on their behalf.
- Final target-specific format and publication option remain open. No OA selection, payment or visibility change.
''',encoding='utf8')
cover=(pkg/'COVER_LETTER_DRAFT.md').read_text();needle='The work contributes';addition='The revision diagnoses rare-missingness extrapolation in the original Yulara Ridge design using actual matrices and a matched independent solver. A uniformly expanded Validation-selected grid is reported separately, including unfavorable changes. It also attributes the Qcells complete-window selection to the frozen negative-label rule and gives paired temporal intervals for the original Ridge contrasts. These findings qualify simple-reference and neural comparisons rather than hiding unstable or adverse results.\n\n'
if addition not in cover:cover=cover.replace(needle,addition+needle)
(pkg/'COVER_LETTER_DRAFT.md').write_text(cover,encoding='utf8')
env=ROOT/'ENVIRONMENT_AND_REPRODUCTION.md';old=env.read_text(encoding='utf8');intro='''# 本轮入口（2026-09-09，优先于下方历史指南）

①CSV→最新20组图：`python -B GFNODE_experiments/scheme_A_diagnostic_revision/build_figures.py`。依赖numpy/pandas/matplotlib，输出diagnostic_figures，使用DejaVu，无私人字体路径。本轮实际运行并在最终PDF检查；旧portable_entry figures输出上轮图，不是当前入口。

②保存预测→新诊断：`python -B GFNODE_experiments/scheme_A_diagnostic_revision/ridge_uncertainty.py --paths review_paths.json`。真实矩阵：`diagnose_ridge.py --paths review_paths.json`；Qcells：`diagnose_qcells.py --paths review_paths.json`。本轮均执行；完整包在project目录提供相对配置。轻量包仅运行`python -B GFNODE_experiments/scheme_A_diagnostic_revision/verify_light.py`，不需要raw或PyTorch，只复算CSV/块SSE。缺失重型路径会明确失败，不搜索磁盘。已有可信pickle需匹配环境：Python3.12、numpy2.0、pandas2.2.2、sklearn1.5、scipy1.13.1、torch2.7.1+cu118。

③官方数据→训练：保留原代码、明确路径及上轮prepare_training_copy；本轮未运行60次神经训练，不能将包内验证称为全训练复现。新增Ridge网格的expanded_ridge.py是明确的补充拟合入口，不是轻量检查。QR脚本需实际矩阵cache；可由expanded_ridge --matrix-cache在显式目录产生，cache不打包，因为原始数据/系数/处理器足以重建。

本轮命令和运行中修复见新诊断VALIDATION_REPORT.md。完整包包含全部已知本地证据；原Alice神经处理器并未另存，不能把新的Ridge处理器称为该历史对象。下文所有“本轮”字样指9月8日上一轮历史记录。

## 旧指南（历史记录）

'''
if not old.startswith('# 本轮入口（2026-09-09'):env.write_text(intro+old,encoding='utf8')
# Derive additive NIST MSE directly from this round's per-seed block SSE reductions.
b=pd.read_csv(HERE/'results/ridge_block_sse.csv');b=b[(b.site=='NIST_GROUND')&(b.block_hours==48)];tot=b.groupby('scope').sum(numeric_only=True);N=tot.loc['full','count'];rows=[]
for seed in [42,43,44]:
 for scope in ['full','power-active','low-power']:
  t=tot.loc[scope];rows.append(dict(seed=seed,scope=scope,points=int(t['count']),neural_SSE=t['Inverted'+str(seed)+'_SSE'],Daily_SSE=t.Daily_SSE,weighted_MSE_difference=(t['Inverted'+str(seed)+'_SSE']-t.Daily_SSE)/N))
z=pd.DataFrame(rows)
for seed,g in z.groupby('seed'):assert abs(g[g.scope=='full'].weighted_MSE_difference.iloc[0]-g[g.scope!='full'].weighted_MSE_difference.sum())<1e-9
z.to_csv(HERE/'results/NIST_DECOMPOSITION_RECHECK.csv',index=False)
# Editable-asset overview and old/new thumbnail comparison, not a substitute for PDF viewing.
canvas=Image.new('RGB',(1600,5*330),'white');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(records):
 im=Image.open(F/(r['name']+'.png'));im.thumbnail((390,285));x=(i%4)*400;y=(i//4)*330;canvas.paste(im,(x+(390-im.width)//2,y+30));draw.text((x+5,y+6),r['name'],fill='black')
canvas.save(F/'FIGURE_OVERVIEW.png')
for block in range((len(records)+3)//4):
 cvs=Image.new('RGB',(1400,4*340),'white');dr=ImageDraw.Draw(cvs)
 for j,r in enumerate(records[block*4:block*4+4]):
  for col,folder in enumerate([P/'review_figures',F]):
   p=folder/(r['name']+'.png');dr.text((col*700+8,j*340+5),('Previous: ' if col==0 else 'Current: ')+r['name'],fill='black')
   if p.exists():im=Image.open(p);im.thumbnail((680,300));cvs.paste(im,(col*700+(680-im.width)//2,j*340+32))
   else:dr.text((15,j*340+55),'New diagnostic figure; no predecessor',fill='black')
 cvs.save(F/f'BEFORE_AFTER_{block+1}.png')
print(json.dumps(counts,indent=2));print(z.groupby('scope').weighted_MSE_difference.mean())

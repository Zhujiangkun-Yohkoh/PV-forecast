"""Render actual final PDFs and inventory embedded figure fonts (not visual approval)."""
from pathlib import Path
import argparse,subprocess,json,re,shutil
import pdfplumber
from PIL import Image,ImageOps,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PAPER=ROOT/'manuscript/clean_pv_benchmark'
def run(dest):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);results={}
 for name,build in [('main','tex-diag-main'),('supplementary','tex-diag-supp')]:
  pdf=ROOT/'.local'/build/(name+'.pdf');shutil.copy2(pdf,PAPER/(name+'.pdf'));shutil.copy2(pdf,PAPER/'submission_package'/(name+'.pdf'))
  with pdfplumber.open(pdf) as doc:
   pages=[]
   for i,p in enumerate(doc.pages):
    figchars=[c for c in p.chars if 'DejaVu' in c['fontname'] and c['text'].strip() and c.get('upright')];pages.append(dict(page=i+1,figure_min_font_pt=min([c['size'] for c in figchars],default=None),small_character_text=''.join(c['text'] for c in figchars if c['size']<8),raster_images=len(p.images)))
  fonts=subprocess.run(['pdffonts',str(pdf)],capture_output=True,text=True).stdout;log=pdf.with_suffix('.log').read_text(errors='replace', encoding='utf8');results[name]=dict(pages=pages,font_table=fonts,overfull_lines=[x for x in log.splitlines() if 'Overfull' in x],undefined_references=bool(re.search(r'(Reference .*undefined|Citation .*undefined|multiply defined)',log)))
  folder=dest/name;folder.mkdir(exist_ok=True);subprocess.run(['pdftoppm','-r','85','-png',str(pdf),str(folder/'page')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  imgs=sorted(folder.glob('page-*.png'))
  for k in range(0,len(imgs),8):
   canvas=Image.new('RGB',(1240,1780),'#ddd');draw=ImageDraw.Draw(canvas)
   for j,p in enumerate(imgs[k:k+8]):
    im=Image.open(p).convert('RGB');im.thumbnail((610,405));x=(j%2)*620;y=(j//2)*445;canvas.paste(im,(x+(610-im.width)//2,y+25));draw.text((x+10,y+5),f'{name} page {k+j+1}',fill='black')
   canvas.save(dest/f'{name}_overview_{k//8+1}.png')
 (HERE/'results/PDF_QA.json').write_text(json.dumps(results,indent=2), encoding='utf8');print(json.dumps({k:{'pages':len(v['pages']),'minimum_figure_font':min([p['figure_min_font_pt'] for p in v['pages'] if p['figure_min_font_pt'] is not None]),'overfull':v['overfull_lines']} for k,v in results.items()},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--destination',required=True);a=p.parse_args();run(a.destination)

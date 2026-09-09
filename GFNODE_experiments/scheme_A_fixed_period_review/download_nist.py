"""Official portal Socket.IO public download, without persisting its access token."""
import argparse,json,time,zipfile,re
from pathlib import Path
import requests
def run(destination):
 out=Path(destination);out.mkdir(parents=True,exist_ok=True);s=requests.Session();base='https://pvdata.nist.gov/socket.io/'
 q={'EIO':4,'transport':'polling'};r=s.get(base,params=q,timeout=35);r.raise_for_status();q['sid']=json.loads(r.text[1:])['sid']
 def post(payload):
  r=s.post(base,params=q,data=payload,timeout=35);r.raise_for_status()
 post('40');token=None;fileid=None;events=[]
 requests_sent=False
 for _ in range(12):
  r=s.get(base,params=q,timeout=35);r.raise_for_status()
  for packet in r.text.split('\x1e'):
   if packet=='2':post('3')
   elif packet.startswith('42'):
    name,value=json.loads(packet[2:]);events.append(name)
    if name=='box token':token=value.get('token',value.get('access_token'));print('Public download token received (not logged)',flush=True)
    elif name=='bulk boxfile':fileid=value.get('boxfile');print('Official bulk file identifier:',fileid,flush=True)
    else:print(name,str(value)[:400],flush=True)
  if not requests_sent:
   for name,value in [('get box token',{}),('get downloads',{'loc':'Ground'}),('get bulk boxfile',{'date':'2018-04-01T05:00:00.000Z','location':'Ground','dataset':'1-min-avg','view':'EntireArray'})]:post('42'+json.dumps([name,value]))
   requests_sent=True
  if token and fileid:break
 if not(token and fileid):raise RuntimeError('No official download identifier/token: '+str(events))
 meta=s.get('https://api.box.com/2.0/files/'+str(fileid),params={'fields':'download_url,name,size'},headers={'Authorization':'Bearer '+token},timeout=35);meta.raise_for_status();m=meta.json();url=m['download_url']
 archive=out/'official_2018_ground.zip'
 with s.get(url,stream=True,timeout=60) as r:
  r.raise_for_status()
  with archive.open('wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
 record={'source':'https://pvdata.nist.gov/','access_date':time.strftime('%Y-%m-%d'),'box_file_id':fileid,'provider_filename':m['name'],'download_bytes':archive.stat().st_size,'extracted':[]}
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  names=z.namelist();print('Archive members:',names[:4],flush=True)
  for name in names:
   if name.lower().endswith('.csv'):
    # Keep the fixed period plus March 30/31 and July 1 buffers only.
    dates=re.findall(r'2018[-_]?([01]\d)[-_]?([0-3]\d)',name)
    if not dates:raise ValueError('Unrecognized official filename '+name)
    mm,dd=dates[-1];day=f'2018-{mm}-{dd}'
    if '2018-03-30'<=day<='2018-07-01':
     dest=out/Path(name).name
     with z.open(name) as src,dest.open('wb') as dst:
      import shutil;shutil.copyfileobj(src,dst)
     record['extracted'].append({'file':dest.name,'size':dest.stat().st_size,'date':day})
 (out/'DOWNLOAD_RECORD.json').write_text(json.dumps(record,indent=2),encoding='utf8');print('Extracted',len(record['extracted']),'fixed-period/buffer CSVs',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('destination');run(p.parse_args().destination)

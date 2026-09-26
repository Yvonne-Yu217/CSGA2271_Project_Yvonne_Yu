from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests,zipfile
URL='https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/annotations.zip'
SIZE=29284070
path=Path(__file__).resolve().parent/'data'/'annotations.zip'
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists() and zipfile.is_zipfile(path):print('Already complete');raise SystemExit
block=262144
parts=list(range(0,SIZE,block))
def fetch(start):
 end=min(SIZE,start+block)-1
 for attempt in range(3):
  try:
   r=requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=45)
   r.raise_for_status()
   assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{SIZE}' and len(r.content)==end-start+1
   return start,r.content
  except Exception:
   if attempt==2:raise
with path.with_suffix('.partial').open('wb') as f:
 f.truncate(SIZE)
 with ThreadPoolExecutor(max_workers=6) as pool:
  for i,(start,content) in enumerate(pool.map(fetch,parts)):
   f.seek(start);f.write(content)
   if i%10==0:print('annotation chunks',i+1,'/',len(parts),flush=True)
tmp=path.with_suffix('.partial')
with zipfile.ZipFile(tmp) as z:
 assert z.testzip() is None
print('ZIP CRC checks passed',flush=True);tmp.replace(path)

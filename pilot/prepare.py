"""Fetch a reproducible small Flickr30K Entities sample; never extract whole archives."""
import argparse, collections, hashlib, json, random, re, zipfile, struct, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fsspec, requests
from PIL import Image
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
BASE='https://raw.githubusercontent.com/BryanPlummer/flickr30k_entities/master/'
IMAGE_URL='https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip'
ANN_URL=BASE+'annotations.zip'

def remote_zip(url):
    return zipfile.ZipFile(fsspec.open(url,mode='rb',block_size=65536,cache_type='blockcache').open())

def parse_caption(line):
    phrases=[]; words=[]; cursor=0
    for m in re.finditer(r'\[(/EN#[^ ]+) ([^\]]+)\]',line):
        words.extend(line[cursor:m.start()].split()); meta=m[1].split('/'); phrase=m[2]
        phrases.append(dict(id=meta[1].removeprefix('EN#'),types=meta[2:],phrase=phrase,start=len(words),end=len(words)+len(phrase.split())))
        words.extend(phrase.split()); cursor=m.end()
    words.extend(line[cursor:].split())
    return ' '.join(words),phrases

def overlap(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return x*y>0

def contains_phrase(text,phrase):
    clean=lambda x:re.sub(r'[^a-z0-9 ]+',' ',x.lower()).split()
    words,target=clean(text),clean(phrase)
    return any(words[i:i+len(target)]==target for i in range(len(words)-len(target)+1))

def matched_generalization(phrase,types):
    base='thing'
    mapping={'people':'person','animals':'animal','vehicles':'vehicle','instruments':'object','clothing':'clothing','bodyparts':'body part','scene':'scene','other':'thing'}
    for kind in types:
        if kind in mapping:base=mapping[kind];break
    words=phrase.split(); generic=base.split()
    if len(words)==1:return 'entity' if generic[-1].lower()==words[0].lower() else generic[-1]
    lead=words[0] if words[0].lower() in {'a','an','the','one','two','three','four','five','six','several','many'} else 'unspecified'
    replacement=[lead]+['unspecified']*max(0,len(words)-len(generic)-1)+generic
    candidate=' '.join(replacement[:len(words)])
    if contains_phrase(candidate,phrase):
        replacement[-1]='entity';candidate=' '.join(replacement[:len(words)])
    return candidate

def build(image_id,xml,lines,split):
    tree=ET.fromstring(xml); size=tree.find('size'); w=int(size.findtext('width'));h=int(size.findtext('height'))
    boxes=collections.defaultdict(list)
    for obj in tree.findall('object'):
        bb=obj.find('bndbox')
        if bb is None: continue
        b=[int(bb.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
        b=[max(0,b[0]-1),max(0,b[1]-1),min(w,b[2]),min(h,b[3])]
        if b[2]-b[0]<16 or b[3]-b[1]<16: continue
        for name in obj.findall('name'): boxes[name.text].append(b)
    result=[]
    for ci,line in enumerate(lines):
        text,ps=parse_caption(line); ids=collections.Counter(p['id'] for p in ps)
        valid=[p for p in ps if len(boxes[p['id']])==1 and ids[p['id']]==1 and 'notvisual' not in p['types'] and len(p['phrase'].split())>=2]
        if len(valid)<2:continue
        for p in valid:
            controls=[q for q in valid if q['id']!=p['id'] and not overlap(boxes[q['id']][0],boxes[p['id']][0])]
            if not controls:continue
            # Deliberately basic deletion pilot: no claim of grammatical or semantic audit.
            minus=' '.join(text.split()[:p['start']]+text.split()[p['end']:])
            if len(minus.split())<3 or contains_phrase(minus,p['phrase']):continue
            q=controls[0]; ctrl=' '.join(text.split()[:q['start']]+text.split()[q['end']:])
            replacement=matched_generalization(p['phrase'],p['types'])
            matched=' '.join(text.split()[:p['start']]+replacement.split()+text.split()[p['end']:])
            natural=None
            for oi,other_line in enumerate(lines):
                if oi==ci:continue
                other_text,other_phrases=parse_caption(other_line)
                if p['id'] not in {x['id'] for x in other_phrases} and not contains_phrase(other_text,p['phrase']):
                    natural=other_text;break
            # Both edits start from the same full caption; the unrelated edit should preserve target coverage.
            regions=[p]+controls
            result.append(dict(image_id=image_id,split=split,caption_index=ci,tminus=minus,tmatched=matched,tnatural=natural,tplus=text,tcontrol=ctrl,target_phrase=p['phrase'],regions=[dict(id=x['id'],box=boxes[x['id']][0],phrase=x['phrase']) for x in regions]))
        if len(result)>=4:break
    return result[:4]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--counts',default='200,50,100');args=ap.parse_args()
    data=ROOT/'data'; (data/'images').mkdir(parents=True,exist_ok=True);(data/'annotations').mkdir(exist_ok=True)
    ann=zipfile.ZipFile(data/'annotations.zip'); print('Annotation index ready',flush=True)
    names=ann.namelist(); print(names[:5],flush=True)
    lookup={Path(n).name:n for n in names if not n.startswith('__MACOSX')}
    imgs=remote_zip(IMAGE_URL); print('Image index ready',flush=True)
    ilook={Path(n).stem:n for n in imgs.namelist() if n.startswith('flickr30k-images/') and n.endswith('.jpg')}
    ordered=sorted(imgs.infolist(),key=lambda x:x.header_offset)
    ranges={x.filename:(x.header_offset,ordered[i+1].header_offset-1) for i,x in enumerate(ordered[:-1])}
    manifest=[];hashes={}; exclusions=collections.Counter()
    def fetch_image(ident):
        ip=data/'images'/(ident+'.jpg')
        if not ip.exists():
            name=ilook[ident];info=imgs.getinfo(name);start,end=ranges[name]
            for attempt in range(3):
                try:
                    resp=requests.get(IMAGE_URL,headers={'Range':f'bytes={start}-{end}'},timeout=60)
                    resp.raise_for_status()
                    assert resp.status_code==206 and len(resp.content)==end-start+1
                    raw=resp.content;head=struct.unpack('<4s5H3I2H',raw[:30]);assert head[0]==b'PK\x03\x04'
                    offset=30+head[-2]+head[-1];compressed=raw[offset:offset+info.compress_size]
                    if info.compress_type==8:payload=zlib.decompress(compressed,-15)
                    elif info.compress_type==0:payload=compressed
                    else:raise ValueError('Unsupported compression')
                    assert len(payload)==info.file_size and zlib.crc32(payload)==info.CRC
                    ip.write_bytes(payload);break
                except Exception:
                    if attempt==2:raise
        with Image.open(ip) as im:im.verify()
        return ident,hashlib.sha256(ip.read_bytes()).hexdigest()
    for split,n in zip(['train','val','test'],map(int,args.counts.split(','))):
        path=data/(split+'.txt')
        if not path.exists():
            r=requests.get(BASE+split+'.txt',timeout=60);r.raise_for_status();path.write_text(r.text)
        ids=path.read_text().split();random.Random(2271).shuffle(ids); selected=0
        for ident in ids:
            if ident not in ilook:continue
            xp=data/'annotations'/(ident+'.xml');tp=data/'annotations'/(ident+'.txt')
            for p in [xp,tp]:
                if not p.exists():p.write_bytes(ann.read(lookup[p.name]))
            rows=build(ident,xp.read_bytes(),tp.read_text().splitlines(),split)
            if not rows:exclusions['no_eligible_pair']+=1;continue
            manifest.extend(rows);selected+=1
            if selected%10==0:print(split,selected,'images',len(manifest),'pairs',flush=True)
            if selected>=n:break
        (data/'manifest.json').write_text(json.dumps(manifest,indent=2))
    image_ids=sorted({r['image_id'] for r in manifest})
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,(ident,sha) in enumerate(pool.map(fetch_image,image_ids)):
            hashes[ident]=sha
            if i%10==0:print('downloaded',i+1,'/',len(image_ids),flush=True)
    (data/'provenance.json').write_text(json.dumps(dict(annotation_url=ANN_URL,image_url=IMAGE_URL,seed=2271,requested_counts=args.counts,image_sha256=hashes,exclusions=dict(exclusions)),indent=2))
    print('DONE',collections.Counter(x['split'] for x in manifest),flush=True)
if __name__=='__main__':main()

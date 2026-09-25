"""Frozen-CLIP entity omission pilot. Metrics do not establish factual complementarity."""
import argparse, collections, hashlib, json, platform, time, re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';OUT=ROOT/'results'; MODEL='openai/clip-vit-base-patch16'
torch.set_num_threads(4)
DEVICE='auto'; BATCH_SIZE=16; LOCAL_ONLY=True

def features(rows):
    digest=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
    path=OUT/'features.npz'
    if path.exists():
        c=np.load(path)
        if str(c['fingerprint'])==digest and 'ngram_scores' in c:return c
    device=DEVICE if DEVICE!='auto' else ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model=CLIPModel.from_pretrained(MODEL,local_files_only=LOCAL_ONLY).to(device).eval()
    proc=CLIPProcessor.from_pretrained(MODEL,local_files_only=LOCAL_ONLY,use_fast=False)
    texts=sorted({r[k] for r in rows for k in ['tminus','tplus','tcontrol']}); ti={t:i for i,t in enumerate(texts)}
    crops=sorted({(r['image_id'],*reg['box']) for r in rows for reg in r['regions']});vi={c:i for i,c in enumerate(crops)}
    def encode(seq,visual):
        chunks=[]
        for i in range(0,len(seq),BATCH_SIZE):
            items=seq[i:i+BATCH_SIZE]
            if visual:
                images=[]
                for key in items:
                    with Image.open(DATA/'images'/(key[0]+'.jpg')) as im: images.append(im.convert('RGB').crop(key[1:]))
                batch=proc(images=images,return_tensors='pt').to(device)
            else:batch=proc(text=items,return_tensors='pt',padding=True,truncation=True,max_length=77).to(device)
            with torch.inference_mode():
                z=model.get_image_features(**batch) if visual else model.get_text_features(**batch)
                z=nn.functional.normalize(z,dim=-1)
            chunks.append(z.cpu().numpy())
            if i%160==0: print('encode','crops' if visual else 'texts',i,'/',len(seq),flush=True)
        return np.concatenate(chunks)
    start=time.time();v=encode(crops,True);t=encode(texts,False)
    flat=[];ptr=[0]
    for r in rows:
        for reg in r['regions']:flat.append([vi[(r['image_id'],*reg['box'])],*[ti[r[k]] for k in ['tminus','tplus','tcontrol']]])
        ptr.append(len(flat))
    stop={'a','an','the','of','in','on','at','with','and','or','is','are','to','by','for'}
    candidates={}
    for text in texts:
        words=re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",text.lower())
        spans={' '.join(words[i:i+n]) for n in range(1,5) for i in range(len(words)-n+1) if any(w not in stop for w in words[i:i+n])}
        candidates[text]=sorted(spans) or [text]
    phrases=sorted({p for ps in candidates.values() for p in ps}); pi={p:i for i,p in enumerate(phrases)}
    pt=encode(['a photo of '+p for p in phrases],False)
    ngram_scores=[]
    for r in rows:
        cv=v[[vi[(r['image_id'],*reg['box'])] for reg in r['regions']]]
        ss=[]
        for key in ['tminus','tplus','tcontrol']:
            candidates_ix=[pi[p] for p in candidates[r[key]]]
            ss.append(.5*(1-(cv@pt[candidates_ix].T).max(axis=1)))
        ngram_scores.extend(np.stack(ss,axis=1))
    np.savez(path,v=v,t=t,index=np.array(flat),ptr=np.array(ptr),fingerprint=digest,ngram_scores=np.array(ngram_scores))
    (OUT/'feature_runtime.json').write_text(json.dumps(dict(device=device,seconds=time.time()-start,crops=len(crops),texts=len(texts),model=MODEL,torch=torch.__version__,transformers=__import__('transformers').__version__,python=platform.python_version(),fingerprint=digest),indent=2))
    return np.load(path)

def per_image(rows,ptr,scores,subset):
    results=collections.defaultdict(list)
    for i in subset:
        a,b=ptr[i:i+2];q=scores[a:b];delta=q[:,0]-q[:,1];d=delta[0]
        higher=(delta>d+1e-8).sum();equal=(np.abs(delta-d)<=1e-8).sum()
        acc=1/equal if higher==0 else 0
        mrr=np.mean([1/(higher+j+1) for j in range(equal)])
        results[rows[i]['image_id']].append([float(d),float(np.abs(delta[1:]).mean()),float(d-delta[1:].mean()),float(d>0),float(acc),float(mrr),float(abs(q[0,2]-q[0,1]))])
    return {k:np.mean(v,axis=0) for k,v in results.items()}
NAMES=['TIG','off_target_drift','local_contrast','positive_gap_rate','delta_acc1','delta_mrr','unrelated_edit_target_drift']

def summarize(p):
    arr=np.stack(list(p.values()));rng=np.random.default_rng(2271)
    boot=np.stack([arr[rng.integers(len(arr),size=len(arr))].mean(0) for _ in range(2000)])
    lo,hi=np.quantile(boot,[.025,.975],axis=0)
    return {k:dict(mean=float(v),ci95=[float(l),float(h)]) for k,v,l,h in zip(NAMES,arr.mean(0),lo,hi)}

class Scorer(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.net=nn.Sequential(nn.Linear(2048,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())
    def forward(self,v,t):
        if self.mode=='text_only':v=torch.zeros_like(v)
        if self.mode=='image_only':t=torch.zeros_like(t)
        return self.net(torch.cat([v,t,v*t,abs(v-t)],-1)).squeeze(-1)

def main():
    global DATA,OUT,DEVICE,BATCH_SIZE,LOCAL_ONLY
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=DATA)
    ap.add_argument('--output-dir',type=Path,default=OUT)
    ap.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--allow-model-download',action='store_true')
    args=ap.parse_args(); DATA=args.data_dir;OUT=args.output_dir;DEVICE=args.device;BATCH_SIZE=args.batch_size;LOCAL_ONLY=not args.allow_model_download
    if BATCH_SIZE<1:ap.error('--batch-size must be positive')
    if DEVICE=='cuda' and not torch.cuda.is_available():ap.error('CUDA requested but unavailable')
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time();rows=json.loads((DATA/'manifest.json').read_text());c=features(rows)
    ptr=c['ptr'];ix=c['index'];v=torch.tensor(c['v'][ix[:,0]]);ts=[torch.tensor(c['t'][ix[:,j]]) for j in [1,2,3]]
    splits={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['train','val','test']}
    ids={s:{rows[i]['image_id'] for i in a} for s,a in splits.items()}
    assert not(ids['train']&ids['test'] or ids['train']&ids['val'] or ids['val']&ids['test'])
    scores={};cos=np.stack([.5*(1-(v*t).sum(-1).numpy()) for t in ts],axis=1);scores['inverse_cosine']=cos
    scores['max_ngram_cosine']=c['ngram_scores']
    scores['constant']=np.full_like(cos,.5)
    scores['negative_text_length']=np.array([[-len(rows[i][k].split())/77 for k in ['tminus','tplus','tcontrol']] for i in range(len(rows)) for _ in rows[i]['regions']])
    scores['region_area']=np.array([[float((reg['box'][2]-reg['box'][0])*(reg['box'][3]-reg['box'][1]))]*3 for r in rows for reg in r['regions']])
    trainflat=torch.tensor(np.concatenate([np.arange(ptr[i],ptr[i+1]) for i in splits['train']]))
    targets=torch.tensor([ptr[i] for i in splits['train']]); off=torch.tensor(np.concatenate([np.arange(ptr[i]+1,ptr[i+1]) for i in splits['train']]))
    labels=torch.zeros(len(v));labels[targets]=1
    history={};checkpoints=OUT/'checkpoints';checkpoints.mkdir(exist_ok=True)
    for mode in ['rank','rank_local','rank_local_control','supervised_omission','text_only','image_only']:
        for seed in [0,1,2]:
            torch.manual_seed(seed);model=Scorer(mode);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);best=-1;beststate=None;bestepoch=0
            for epoch in range(1,61):
                model.train();qm,qp,qc=[model(v,t) for t in ts]
                if mode=='supervised_omission':
                    loss=nn.functional.binary_cross_entropy(qm[trainflat],labels[trainflat])+nn.functional.binary_cross_entropy(qp[trainflat],torch.zeros(len(trainflat)))
                else:
                    loss=torch.relu(.1-qm[targets]+qp[targets]).mean()
                    if mode!='rank':loss=loss+abs(qm[off]-qp[off]).mean()
                    if mode in ['rank_local_control','text_only','image_only']:loss=loss+abs(qc[targets]-qp[targets]).mean()
                opt.zero_grad();loss.backward();opt.step()
                if epoch%5==0:
                    model.eval()
                    with torch.no_grad():ss=np.stack([model(v,t).numpy() for t in ts],axis=1)
                    metric=np.mean([x[4] for x in per_image(rows,ptr,ss,splits['val']).values()])
                    if metric>best:best=metric;beststate={k:x.detach().clone() for k,x in model.state_dict().items()};bestepoch=epoch
            model.load_state_dict(beststate);model.eval();name=f'{mode}_seed{seed}'
            with torch.no_grad():scores[name]=np.stack([model(v,t).numpy() for t in ts],axis=1)
            torch.save(beststate,checkpoints/(name+'.pt'));history[name]=dict(best_epoch=bestepoch,validation_delta_acc1=float(best))
            if mode=='rank_local_control':
                # Swap only test visual features across images; text and targets remain unchanged.
                shuffled=v.clone();test_flat=np.concatenate([np.arange(ptr[i],ptr[i+1]) for i in splits['test']])
                owners=np.array([r['image_id'] for r in rows for _ in r['regions']]);rng=np.random.default_rng(2271)
                for j in test_flat:
                    candidates=test_flat[owners[test_flat]!=owners[j]];shuffled[j]=v[int(rng.choice(candidates))]
                with torch.no_grad():scores[f'shuffled_image_seed{seed}']=np.stack([model(shuffled,t).numpy() for t in ts],axis=1)
            print(name,'validation',best,'epoch',bestepoch,flush=True)
    per={k:per_image(rows,ptr,s,splits['test']) for k,s in scores.items()}
    result={k:summarize(p) for k,p in per.items()}
    paired={}
    for k in per:
        keys=sorted(per[k]);delta={i:per[k][i]-per['inverse_cosine'][i] for i in keys};paired[k]=summarize(delta)['delta_acc1']
    report=dict(scope='Entity phrase deletion feasibility pilot; no independent semantic labels or manual audit',image_counts={k:len(x) for k,x in ids.items()},pair_counts={k:len(x) for k,x in splits.items()},metrics=result,paired_delta_acc1_vs_inverse=paired,selection=history,seconds=time.time()-start)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2));np.savez(OUT/'predictions.npz',**scores)
    (OUT/'per_image_metrics.json').write_text(json.dumps({k:{i:x.tolist() for i,x in p.items()} for k,p in per.items()},indent=2))
    print(json.dumps({k:round(x['delta_acc1']['mean'],4) for k,x in result.items()},indent=2),flush=True)
if __name__=='__main__':main()

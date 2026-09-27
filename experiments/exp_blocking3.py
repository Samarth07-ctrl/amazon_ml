"""Experiment 3: fast token + pair IDF-weighted retrieval (NO char n-grams,
so it scales). Tests recall/candidate tradeoffs for the real blocker design."""
import sys, os, time, random, math
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import normalization as N
from collections import defaultdict
import numpy as np

DATA = "utils/datasets/train"; random.seed(42)

def load_gt(path):
    gt={}
    with open(path,encoding="utf-8") as f:
        next(f)
        for line in f:
            s1,tab,rest=line.partition("\t"); rest=rest.rstrip("\n")
            gt[s1]=[x for x in rest.split(",") if x] if rest.strip() else []
    return gt
t=time.time(); gt=load_gt(os.path.join(DATA,"train_ground_truth.tsv"))
all_s1=list(gt.keys()); random.shuffle(all_s1)
val_s1=set(all_s1[:15000]); needed=set()
for s in val_s1: needed.update(gt[s])
print(f"GT {len(gt)} val {len(val_s1)} needed {len(needed)} {time.time()-t:.1f}s")

def load_src(path, keep=None, need=None, rate=0.0):
    rows=[]
    with open(path,encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)<4: continue
            i=p[0]
            if (keep and i in keep) or (need and i in need) or (rate>0 and random.random()<rate):
                rows.append((i,p[1],p[2],p[3]))
    return rows
t=time.time()
s1_rows=load_src(os.path.join(DATA,"train_source1.tsv"),keep=val_s1)
docs=load_src(os.path.join(DATA,"train_source2.tsv"),need=needed,rate=0.20)
docs+=load_src(os.path.join(DATA,"train_source3.tsv"),need=needed,rate=0.20)
print(f"s1={len(s1_rows)} docs={len(docs)} {time.time()-t:.1f}s")

def feats(name, addr, country, use_pair, use_addr):
    nr=N.name_reprs(name); c=country; fs=[]
    toks=[t for t in nr["tokset"] if len(t)>=3]
    sqs=set()
    for tok in toks:
        fs.append("t|"+c+"|"+tok)
        sq=N.squeeze(tok)
        if len(sq)>=3: fs.append("t|"+c+"|"+sq); sqs.add(sq)
        else: sqs.add(tok)
    if use_pair:
        sl=sorted(sqs)
        for i in range(len(sl)):
            for j in range(i+1,len(sl)):
                fs.append("p|"+c+"|"+sl[i]+"~"+sl[j])
    if use_addr:
        ar=N.addr_reprs(addr)
        for p in ar["postals"]: fs.append("z|"+c+"|"+p)
        # street number + a name token
        for nm in ar["nums"]:
            for sq in list(sqs)[:2]:
                fs.append("zn|"+c+"|"+nm+"|"+sq)
    return fs

def run(tag, use_pair, use_addr, K, cap=15000):
    t=time.time()
    postings=defaultdict(list)
    dfeat=[None]*len(docs)
    for idx,(i,nm,ad,co) in enumerate(docs):
        ff=set(feats(nm,ad,co,use_pair,use_addr))
        for fk in ff: postings[fk].append(idx)
    Nd=len(docs)
    idf={k:math.log(1.0+Nd/len(v)) for k,v in postings.items()}
    recalls=[]; india=[]; us=[]; fr=[]; ncand=[]; cov=0; tot=0
    for (i,nm,ad,co) in s1_rows:
        score=defaultdict(float)
        for fk in set(feats(nm,ad,co,use_pair,use_addr)):
            post=postings.get(fk)
            if not post or len(post)>cap: continue
            w=idf[fk]
            for d in post: score[d]+=w
        if score:
            if len(score)>K:
                top=sorted(score.items(),key=lambda x:-x[1])[:K]
            else:
                top=score.items()
            cand={docs[d][0] for d,_ in top}
        else: cand=set()
        ncand.append(len(cand)); truth=set(gt[i])
        if truth:
            f=len(truth&cand); recalls.append(f/len(truth)); cov+=f; tot+=len(truth)
            if co=="India": india.append(f/len(truth))
            elif co=="US": us.append(f/len(truth))
    nd=np.array(ncand)
    print(f"[{tag}] K={K}: macroR={np.mean(recalls):.4f} microR={cov/tot:.4f} avgCand={nd.mean():.1f} "
          f"p50={np.median(nd):.0f} p99={np.percentile(nd,99):.0f} indiaR={np.mean(india):.4f} usR={np.mean(us):.4f} {time.time()-t:.1f}s")

run("tok",       False, False, 50)
run("tok+pair",  True,  False, 50)
run("tok+pair+addr", True, True, 50)
run("tok+pair+addr", True, True, 30)
run("tok+pair+addr", True, True, 80)

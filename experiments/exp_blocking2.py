"""Experiment 2: IDF-weighted inverted-index retrieval with char n-grams.

Instead of hard-dropping common keys, score each candidate by the summed IDF of
shared features and keep top-K. Features = name tokens + squeezed tokens +
char n-grams of the squeezed name (char n-grams give fuzzy / cross-script
recall). Everything is country-prefixed so blocking stays within-country.
"""
import sys, os, time, random, math
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import normalization as N
from collections import defaultdict
import numpy as np

DATA = "utils/datasets/train"
random.seed(42)

def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, tab, rest = line.partition("\t"); rest = rest.rstrip("\n")
            gt[s1] = [x for x in rest.split(",") if x] if rest.strip() else []
    return gt

print("loading GT..."); t=time.time()
gt = load_gt(os.path.join(DATA, "train_ground_truth.tsv"))
all_s1 = list(gt.keys()); random.shuffle(all_s1)
val_s1 = set(all_s1[:20000])
needed = set()
for s1 in val_s1: needed.update(gt[s1])
print(f"  GT {len(gt)}, valS1 {len(val_s1)}, needed {len(needed)} in {time.time()-t:.1f}s")

def load_src(path, keep=None, need=None, rate=0.0):
    rows=[]
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)<4: continue
            i=p[0]
            if (keep and i in keep) or (need and i in need) or (rate>0 and random.random()<rate):
                rows.append((i,p[1],p[2],p[3]))
    return rows

t=time.time()
s1_rows = load_src(os.path.join(DATA,"train_source1.tsv"), keep=val_s1)
docs = load_src(os.path.join(DATA,"train_source2.tsv"), need=needed, rate=0.12)
docs += load_src(os.path.join(DATA,"train_source3.tsv"), need=needed, rate=0.12)
print(f"  s1={len(s1_rows)} docs={len(docs)} in {time.time()-t:.1f}s")


def feats(name, country, use_gram, gram_n):
    nr = N.name_reprs(name)
    c = country
    fs = []
    for tok in nr["tokset"]:
        if len(tok) >= 3:
            fs.append("t|"+c+"|"+tok)
            sq = N.squeeze(tok)
            if sq != tok and len(sq) >= 3:
                fs.append("t|"+c+"|"+sq)
    if use_gram:
        s = "#" + nr["sqz"] + "#"
        for k in range(len(s)-gram_n+1):
            fs.append("g|"+c+"|"+s[k:k+gram_n])
    return fs

def run(use_gram, gram_n, K, cap=6000):
    t=time.time()
    # build index + df
    postings=defaultdict(list)
    for idx,(i,nm,ad,co) in enumerate(docs):
        for fkey in set(feats(nm,co,use_gram,gram_n)):
            postings[fkey].append(idx)
    Ndoc=len(docs)
    idf={k:math.log(1.0+Ndoc/len(v)) for k,v in postings.items()}
    # query
    recalls=[]; india=[]; us=[]; ncand=[]; covered=0; total=0
    for (i,nm,ad,co) in s1_rows:
        score=defaultdict(float)
        for fkey in set(feats(nm,co,use_gram,gram_n)):
            post=postings.get(fkey)
            if post is None: continue
            if len(post)>cap: continue
            w=idf[fkey]
            for d in post: score[d]+=w
        if score:
            top=sorted(score.items(), key=lambda x:-x[1])[:K]
            cand={docs[d][0] for d,_ in top}
        else:
            cand=set()
        ncand.append(len(cand))
        truth=set(gt[i])
        if truth:
            f=len(truth&cand); recalls.append(f/len(truth))
            covered+=f; total+=len(truth)
            if co=="India": india.append(f/len(truth))
            elif co=="US": us.append(f/len(truth))
    nd=np.array(ncand)
    print(f"gram={use_gram}(n{gram_n}) K={K} cap={cap}: macroR={np.mean(recalls):.4f} microR={covered/total:.4f} "
          f"avgCand={nd.mean():.1f} p99={np.percentile(nd,99):.0f} indiaR={np.mean(india):.4f} usR={np.mean(us):.4f} "
          f"t={time.time()-t:.1f}s")

run(False,0,40)
run(True,3,40)
run(True,4,40)
run(True,3,60)
run(True,4,60)
run(True,4,100)

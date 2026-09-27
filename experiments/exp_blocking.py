"""Blocking-key recall experiment on a held-out training sample.

Builds a doc index from (all true matches of the sampled S1) + random
distractors, then measures candidate recall / counts for several key designs.
This drives the blocking design decision before scaling to the full test set.
"""
import sys, os, time, random
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import normalization as N

DATA = "utils/datasets/train"
random.seed(42)

def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, tab, rest = line.partition("\t")
            rest = rest.rstrip("\n")
            gt[s1] = [x for x in rest.split(",") if x] if rest.strip() else []
    return gt

print("loading GT..."); t=time.time()
gt = load_gt(os.path.join(DATA, "train_ground_truth.tsv"))
print(f"  {len(gt)} rows in {time.time()-t:.1f}s")

# sample validation S1
all_s1 = list(gt.keys())
random.shuffle(all_s1)
VAL_N = 20000
val_s1 = set(all_s1[:VAL_N])
needed_docs = set()
for s1 in val_s1:
    needed_docs.update(gt[s1])
print(f"val S1={len(val_s1)}, needed true-match docs={len(needed_docs)}")

# load val S1 records
def load_source_filtered(path, keep_ids=None, sample_rate=0.0, need_set=None):
    """Return list of (id,name,addr,country). Keep if id in keep_ids OR
    (id in need_set) OR random<sample_rate."""
    rows=[]
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p)<4: continue
            i=p[0]
            take=False
            if keep_ids is not None and i in keep_ids: take=True
            elif need_set is not None and i in need_set: take=True
            elif sample_rate>0 and random.random()<sample_rate: take=True
            if take: rows.append((i,p[1],p[2],p[3]))
    return rows

print("loading S1 val records..."); t=time.time()
s1_rows = load_source_filtered(os.path.join(DATA,"train_source1.tsv"), keep_ids=val_s1)
print(f"  {len(s1_rows)} in {time.time()-t:.1f}s")

print("loading S2/S3 docs (true matches + 12% distractors)..."); t=time.time()
docs = load_source_filtered(os.path.join(DATA,"train_source2.tsv"), need_set=needed_docs, sample_rate=0.12)
docs += load_source_filtered(os.path.join(DATA,"train_source3.tsv"), need_set=needed_docs, sample_rate=0.12)
print(f"  {len(docs)} docs in {time.time()-t:.1f}s")


from collections import defaultdict

def rec_keys(name, addr, country, cfg):
    """Generate blocking keys for one record per config flags in cfg (set)."""
    nr = N.name_reprs(name)
    keys=[]
    c=country
    core=nr["core"]; sqz=nr["sqz"]; toks=nr["tokens"]
    if "core" in cfg and core:
        keys.append("NC|"+c+"|"+core)
    if "sqz" in cfg and len(sqz)>=5:
        keys.append("NS|"+c+"|"+sqz)
    if "tok" in cfg:
        for tkn in set(toks):
            if len(tkn)>=4:
                keys.append("NT|"+c+"|"+tkn)
    if "stok" in cfg:
        for tkn in set(toks):
            sq=N.squeeze(tkn)
            if len(sq)>=4:
                keys.append("NST|"+c+"|"+sq)
    if "pair" in cfg:
        st=sorted(set(t for t in toks if len(t)>=3))
        for i in range(len(st)):
            for j in range(i+1,len(st)):
                keys.append("NP|"+c+"|"+st[i]+"~"+st[j])
    if ("apost" in cfg) or ("aname" in cfg):
        ar=N.addr_reprs(addr)
        sq_toks=sorted(set(N.squeeze(t) for t in toks if len(t)>=4))
        if "apost" in cfg:
            for p in ar["postals"]:
                # postal + rarest-ish name token
                for tkn in sq_toks[:2]:
                    keys.append("AP|"+c+"|"+p+"|"+tkn)
        if "aname" in cfg:
            # address word set core + top name token
            aw=sorted(ar["words"])
            if aw and sq_toks:
                keys.append("AN|"+c+"|"+aw[0]+"|"+sq_toks[0])
    return keys

CONFIGS = {
  "core":            {"core"},
  "core+sqz":        {"core","sqz"},
  "core+stok":       {"core","stok"},
  "core+sqz+stok":   {"core","sqz","stok"},
  "core+sqz+stok+pair": {"core","sqz","stok","pair"},
  "all":             {"core","sqz","stok","pair","apost"},
}

# precompute name reprs for docs once (reuse across configs) is hard w/ cfg; just run each
CAP=400  # skip keys with more than CAP docs (non-discriminative)

for cname,cfg in CONFIGS.items():
    t=time.time()
    # build doc index
    index=defaultdict(list)
    for idx,(i,nm,ad,co) in enumerate(docs):
        for k in rec_keys(nm,ad,co,cfg):
            index[k].append(idx)
    # drop oversized posting lists
    dropped=sum(1 for k,v in index.items() if len(v)>CAP)
    # query
    tot_cand=0; recalls=[]; india_rec=[]; us_rec=[]; fr=[]
    covered_pos=0; total_pos=0; lost=0
    ncand_dist=[]
    for (i,nm,ad,co) in s1_rows:
        cset=set()
        for k in rec_keys(nm,ad,co,cfg):
            post=index.get(k)
            if post and len(post)<=CAP:
                cset.update(post)
        cand_ids={docs[d][0] for d in cset}
        truth=set(gt[i])
        ncand_dist.append(len(cand_ids))
        tot_cand+=len(cand_ids)
        if truth:
            found=len(truth & cand_ids)
            recalls.append(found/len(truth))
            total_pos+=len(truth); covered_pos+=found; lost+=len(truth)-found
            if co=="India": india_rec.append(found/len(truth))
            elif co=="US": us_rec.append(found/len(truth))
    import numpy as np
    nd=np.array(ncand_dist)
    print(f"[{cname}] macroRecall={np.mean(recalls):.4f} microRecall={covered_pos/total_pos:.4f} "
          f"avgCand={nd.mean():.1f} medCand={np.median(nd):.0f} p99Cand={np.percentile(nd,99):.0f} "
          f"lostPos={lost} droppedKeys={dropped} indiaR={np.mean(india_rec):.4f} usR={np.mean(us_rec):.4f} "
          f"build+query={time.time()-t:.1f}s")

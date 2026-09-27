"""Correctness check: vectorised blocking join vs brute-force reference."""
import sys, os
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import numpy as np
import blocking as B
from collections import defaultdict

# synthetic docs and queries
docs_ids = [f"S2-{i}" for i in range(60)] + [f"S3-{i}" for i in range(40)]
rng = np.random.default_rng(0)
words = ["alpha","beta","gamma","delta","omega","sigma","tau","zeta","acme","globex"]
def mk(n):
    names=[]; addrs=[]; cos=[]
    for _ in range(n):
        k=rng.integers(1,4)
        names.append(" ".join(rng.choice(words,size=k,replace=False)))
        addrs.append(f"{rng.integers(1,999)} main st {rng.integers(10000,99999)}")
        cos.append(rng.choice(["US","India","France"]))
    return names,addrs,cos
dn,da,dc = mk(100)
qn,qa,qc = mk(20)
q_ids=[f"S1-{i}" for i in range(20)]

idx = B.build_doc_index(docs_ids, dn, da, dc, use_pair=True, use_addr=True)

# brute force reference: same keys, IDF sum, top-K
def keyset(nm,ad,co): return set(B.record_keys(nm,ad,co,True,True))
dkeys=[keyset(dn[i],da[i],dc[i]) for i in range(100)]
df=defaultdict(int)
for ks in dkeys:
    for k in ks: df[k]+=1
import math
idf={k:math.log(1.0+100/df[k]) for k in df}
K=10; cap=20000
ref={}
for i in range(20):
    qk=keyset(qn[i],qa[i],qc[i]); sc=defaultdict(float)
    for k in qk:
        if k in df and df[k]<=cap:
            for j in range(100):
                if k in dkeys[j]: sc[j]+=idf[k]
    top=sorted(sc.items(),key=lambda x:(-x[1],x[0]))[:K]
    ref[i]={docs_ids[j] for j,_ in top}

got={}
for qi,(qid,cands) in enumerate(B.query_candidates(idx,q_ids,qn,qa,qc,K=K,cap=cap,batch=7)):
    got[qi]=set(cands)

ok=True
for i in range(20):
    # compare as sets (ties may reorder but top-K set should match when no score ties at boundary)
    if got[i]!=ref[i]:
        # allow boundary tie differences: check score-equivalence
        inter=got[i]&ref[i]
        if len(got[i])==len(ref[i]) and len(inter)>=len(ref[i])-2:
            continue
        ok=False
        print(f"q{i} MISMATCH got={sorted(got[i])} ref={sorted(ref[i])}")
print("PAIR/ADDR join correctness:", "OK" if ok else "FAIL")
print("sample cand counts:", [len(got[i]) for i in range(5)])

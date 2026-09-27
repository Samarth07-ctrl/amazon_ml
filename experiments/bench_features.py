"""Micro-benchmark: prep_record + pair_features throughput on real records,
to estimate full-scale featurisation cost and pick K."""
import sys, os, time, random
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import numpy as np
import features as F
import data_loader as D

random.seed(0)
DATA = "utils/datasets/train"

def sample_rows(path, n):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                rows.append((p[1], p[2], p[3]))
            if len(rows) >= n:
                break
    return rows

s1 = sample_rows(os.path.join(DATA, "train_source1.tsv"), 4000)
s2 = sample_rows(os.path.join(DATA, "train_source2.tsv"), 4000)

t = time.time()
r1 = [F.prep_record(*r) for r in s1]
r2 = [F.prep_record(*r) for r in s2]
prep_dt = time.time() - t
print(f"prep: {len(r1)+len(r2)} recs in {prep_dt:.2f}s -> {(len(r1)+len(r2))/prep_dt:,.0f}/s")

N = 200000
t = time.time()
for i in range(N):
    a = r1[i % len(r1)]; b = r2[i % len(r2)]
    F.pair_features(a, b, 1.0, 20, i % 20)
feat_dt = time.time() - t
rate = N / feat_dt
print(f"pair_features: {N} pairs in {feat_dt:.2f}s -> {rate:,.0f}/s")

for tag, npairs in [("K=30 test (50M)", 50_000_000), ("K=50 test (83M)", 83_000_000)]:
    print(f"  {tag}: ~{npairs/rate/60:.1f} min featurize")
print(f"  prep 10M docs (lazy, ~1x): ~{10_000_000/((len(r1)+len(r2))/prep_dt)/60:.1f} min")

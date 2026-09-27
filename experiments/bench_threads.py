"""Does thread-parallel featurisation help? rapidfuzz releases the GIL during
its C computation, so threads may parallelise the heavy part without the
Windows spawn/memory cost of multiprocessing."""
import sys, os, time, random
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import features as F

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

r1 = [F.prep_record(*r) for r in sample_rows(os.path.join(DATA, "train_source1.tsv"), 3000)]
r2 = [F.prep_record(*r) for r in sample_rows(os.path.join(DATA, "train_source2.tsv"), 3000)]

N = 300000
pairs = [(r1[i % len(r1)], r2[i % len(r2)], i % 20) for i in range(N)]

def chunk_feat(sub):
    out = np.empty((len(sub), F.N_FEATURES), np.float32)
    for k, (a, b, rk) in enumerate(sub):
        out[k] = F.pair_features(a, b, 1.0, 20, rk)
    return out

t = time.time()
_ = chunk_feat(pairs)
serial = time.time() - t
print(f"serial: {N/serial:,.0f}/s ({serial:.1f}s)")

for W in [4, 8, 12]:
    sz = (N + W - 1) // W
    chunks = [pairs[i:i+sz] for i in range(0, N, sz)]
    t = time.time()
    with ThreadPoolExecutor(max_workers=W) as ex:
        res = list(ex.map(chunk_feat, chunks))
    dt = time.time() - t
    print(f"threads={W}: {N/dt:,.0f}/s ({dt:.1f}s)  speedup {serial/dt:.1f}x")

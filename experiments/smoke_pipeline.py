"""End-to-end plumbing smoke test on a tiny synthetic dataset.

Verifies training-pair construction, model fit, threshold sweep, validation
metric, and streaming test output writing -- all on data small enough to run in
seconds. Not a quality test; purely that the wiring is correct.
"""
import sys, os, tempfile, csv
sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
import numpy as np
import pipeline as P
import data_loader as D
import model as M

rng = np.random.default_rng(0)
words = ["acme", "globex", "initech", "umbrella", "wayne", "stark", "wonka",
         "soylent", "tyrell", "hooli", "pied", "piper", "vandelay"]
countries = ["US", "India", "France"]


def rand_name():
    k = rng.integers(1, 3)
    return " ".join(rng.choice(words, size=k, replace=False))


def make_dataset(root, split, n_s1=400):
    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    sid = did = 0
    for _ in range(n_s1):
        co = rng.choice(countries)
        nm = rand_name()
        zipc = str(rng.integers(10000, 99999))
        s1_id = f"S1-{sid}"; sid += 1
        s1.append((s1_id, nm, f"{rng.integers(1,999)} main st {zipc}", co))
        matches = []
        # ~85% get 1-2 matches (mirrors the real high match rate), rest singleton
        if rng.random() < 0.85:
            for _ in range(rng.integers(1, 3)):
                pre = rng.choice(["S2", "S3"])
                mid = f"{pre}-{did}"; did += 1
                # noisy variant of the same name/addr
                noisy = nm.replace("a", "aa") if rng.random() < 0.4 else nm
                row = (mid, noisy, f"{rng.integers(1,999)} main st {zipc}", co)
                (s2 if pre == "S2" else s3).append(row)
                matches.append(mid)
        # distractors
        for _ in range(rng.integers(1, 4)):
            pre = rng.choice(["S2", "S3"])
            mid = f"{pre}-{did}"; did += 1
            (s2 if pre == "S2" else s3).append(
                (mid, rand_name(), f"{rng.integers(1,999)} oak rd {rng.integers(10000,99999)}", rng.choice(countries)))
        gt.append((s1_id, ",".join(matches)))

    def wr(name, rows, header):
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(header) + "\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
    wr(f"{split}_source1.tsv", s1, ["entity_id", "business_name", "business_address", "country"])
    wr(f"{split}_source2.tsv", s2, ["entity_id", "business_name", "business_address", "country"])
    wr(f"{split}_source3.tsv", s3, ["entity_id", "business_name", "business_address", "country"])
    if split == "train":
        wr("train_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])


root = tempfile.mkdtemp(prefix="ber_smoke_")
make_dataset(root, "train", 400)
make_dataset(root, "test", 120)
out = os.path.join(root, "output")
os.makedirs(out, exist_ok=True)

res, model, tau = P.train_and_validate(
    root, os.path.join(out, "model.joblib"), os.path.join(out, "run_log.jsonl"),
    train_n=300, val_n=60, K=20, cap=100000)
print("VAL F0.5=%.3f P=%.3f R=%.3f tau=%s candR=%.3f" % (
    res["val_f0.5"], res["val_precision"], res["val_recall"], tau,
    res["candidate_recall"]["macro_recall"]))

stats = P.predict_test(root, out, model, tau, K=20, cap=100000, batch=50)
print("TEST", {k: stats[k] for k in ("n_test_s1", "n_pred_matches", "n_singletons", "n_candidates_total")})

# validate output structure
mp = os.path.join(out, "matching_results.tsv")
cp = os.path.join(out, "candidate_pairs.tsv")
with open(mp, encoding="utf-8") as f:
    mh = f.readline().strip().split("\t"); mrows = [l for l in f]
with open(cp, encoding="utf-8") as f:
    ch = f.readline().strip().split("\t"); crows = [l for l in f]
assert mh == ["source1_entity_id", "matched_entity_ids"], mh
assert ch == ["source1_entity_id", "candidate_entity_ids"], ch
assert len(mrows) == stats["n_test_s1"], (len(mrows), stats["n_test_s1"])
assert len(crows) == stats["n_test_s1"]
print("OUTPUT STRUCTURE OK  rows=%d" % len(mrows))
print("SMOKE OK")

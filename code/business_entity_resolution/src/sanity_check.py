"""Independent sanity checker for the submission files.

Deliberately re-implements the structural checks (rather than importing the
official validator) so the two can disagree. Verifies both TSVs against the
test Source-1 ids and the S2/S3 id universe, and reports match/candidate
statistics. Exit code 0 == all checks passed.

Usage:
    python sanity_check.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir utils/datasets/test
"""
import argparse
import os
import sys


def read_test_ids(test_dir):
    s1, docs = [], set()
    for fn, bucket in (("test_source1.tsv", None),
                       ("test_source2.tsv", docs), ("test_source3.tsv", docs)):
        path = os.path.join(test_dir, fn)
        with open(path, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                i = line.split("\t", 1)[0].strip()
                if not i:
                    continue
                if bucket is None:
                    s1.append(i)
                else:
                    bucket.add(i)
    return s1, docs


def read_two_col(path, expect_header):
    rows = {}
    dup = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != expect_header:
            return None, header, dup
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            ids = [x for x in val.split(",") if x] if val.strip() else []
            if sid in rows:
                dup.append(sid)
            rows[sid] = ids
    return rows, header, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--candidate", default="output/candidate_pairs.tsv")
    ap.add_argument("--test-dir", default="utils/datasets/test")
    a = ap.parse_args()

    errs, warns = [], []
    s1_ids, doc_ids = read_test_ids(a.test_dir)
    s1_set = set(s1_ids)

    match, mh, mdup = read_two_col(a.matching, ["source1_entity_id", "matched_entity_ids"])
    cand, ch, cdup = read_two_col(a.candidate, ["source1_entity_id", "candidate_entity_ids"])
    if match is None:
        errs.append(f"matching header wrong: {mh}")
    if cand is None:
        errs.append(f"candidate header wrong: {ch}")
    if errs:
        for e in errs:
            print("FAIL:", e)
        sys.exit(1)

    if mdup:
        errs.append(f"{len(mdup)} duplicate S1 rows in matching")
    if cdup:
        errs.append(f"{len(cdup)} duplicate S1 rows in candidate")

    miss_m = s1_set - set(match)
    miss_c = s1_set - set(cand)
    extra_m = set(match) - s1_set
    if miss_m:
        errs.append(f"{len(miss_m)} test S1 missing from matching")
    if miss_c:
        errs.append(f"{len(miss_c)} test S1 missing from candidate")
    if extra_m:
        errs.append(f"{len(extra_m)} unknown S1 in matching")

    n_match = n_singleton = n_cand = 0
    bad_prefix = subset_viol = intra_dup = 0
    for sid in match:
        m = match[sid]
        c = set(cand.get(sid, []))
        if len(m) != len(set(m)):
            intra_dup += 1
        for x in m:
            if not (x.startswith("S2-") or x.startswith("S3-")):
                bad_prefix += 1
            if x not in c:
                subset_viol += 1
        n_match += len(m)
        if not m:
            n_singleton += 1
    for sid in cand:
        cc = cand[sid]
        if len(cc) != len(set(cc)):
            intra_dup += 1
        n_cand += len(cc)

    if bad_prefix:
        errs.append(f"{bad_prefix} matched ids without S2-/S3- prefix")
    if subset_viol:
        errs.append(f"{subset_viol} matched ids not in candidate set (subset violation)")
    if intra_dup:
        errs.append(f"{intra_dup} rows with intra-list duplicates")

    print(f"test S1={len(s1_ids)}  doc universe={len(doc_ids)}")
    print(f"matching rows={len(match)}  candidate rows={len(cand)}")
    print(f"predicted matches={n_match}  singletons(empty)={n_singleton}"
          f"  ({100*n_singleton/max(1,len(match)):.1f}%)")
    print(f"total candidates={n_cand}  avg cand/entity={n_cand/max(1,len(cand)):.1f}")
    if warns:
        for w in warns:
            print("WARN:", w)
    if errs:
        for e in errs:
            print("FAIL:", e)
        sys.exit(1)
    print("SANITY CHECK PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""End-to-end orchestration: training + validation + test inference.

Two entry points:
  * ``train_and_validate`` -- entity-level split, build candidates via blocking,
    construct positive/hard-negative training pairs, fit the model, tune the
    F0.5 threshold on held-out entities, and report metrics.
  * ``predict_test`` -- index the test documents, block, score every candidate,
    apply the tuned threshold, and write the two submission TSVs.

Records are kept as parallel arrays (ids/names/addrs/countries) plus an
id->row map; feature reprs are computed lazily per batch so we never hold ~10M
prepped dicts in memory at once.
"""
import os
import numpy as np

import data_loader as D
import blocking as B
import features as F
import model as M
import matching as MT
import evaluation as E
import utils as U


def _index_ids(ids):
    return {v: i for i, v in enumerate(ids)}


def _prep_rows(src, rows):
    """Prep (feature reprs) for a set of row indices of a source dict."""
    out = {}
    for r in rows:
        out[r] = F.prep_record(src["names"][r], src["addrs"][r], src["countries"][r])
    return out


def _merged_docs(s2, s3):
    """Concatenate S2 and S3 into one document source (ids keep their prefix)."""
    return {
        "ids": s2["ids"] + s3["ids"],
        "names": s2["names"] + s3["names"],
        "addrs": s2["addrs"] + s3["addrs"],
        "countries": s2["countries"] + s3["countries"],
    }


class DocPrep:
    """Lazy id -> prepped Rec cache over a document source.

    ``clearable`` cache lets test inference bound memory by dropping the cache
    between query batches (docs recurring across batches are re-prepped)."""

    def __init__(self, docs):
        self.docs = docs
        self.row_of = _index_ids(docs["ids"])
        self.cache = {}

    def get(self, cid):
        r = self.row_of[cid]
        v = self.cache.get(r)
        if v is None:
            d = self.docs
            v = F.prep_record(d["names"][r], d["addrs"][r], d["countries"][r])
            self.cache[r] = v
        return v

    def clear(self):
        self.cache.clear()


def _q_recs(src, ids):
    """Prep a set of query records into {id: Rec}."""
    row = _index_ids(src["ids"])
    out = {}
    for qid in ids:
        r = row[qid]
        out[qid] = F.prep_record(src["names"][r], src["addrs"][r], src["countries"][r])
    return out


def build_training_pairs(q_recs, cand_iter, gt, docprep, neg_cap=40):
    """Construct labelled training pairs.

    Positives: every true GT match (features computed even if blocking missed
    it, so the classifier learns positive patterns regardless of recall).
    Negatives: blocking candidates not in GT (hard negatives), capped per query.
    Returns (X float32[n, F], y int8[n]).
    """
    rows, labels = [], []
    for qid, cands, scores in cand_iter:
        qr = q_recs[qid]
        truth = gt.get(qid, [])
        tset = set(truth)
        nk = len(cands)
        seen_pos = set()
        negs = 0
        for rank, (cid, sc) in enumerate(zip(cands, scores)):
            dr = docprep.get(cid)
            feat = F.pair_features(qr, dr, float(sc), nk, rank)
            if cid in tset:
                rows.append(feat); labels.append(1); seen_pos.add(cid)
            elif negs < neg_cap:
                rows.append(feat); labels.append(0); negs += 1
        # positives the blocker missed -- add with neutral provenance
        for cid in tset - seen_pos:
            try:
                dr = docprep.get(cid)
            except KeyError:
                continue
            rows.append(F.pair_features(qr, dr, 0.0, nk, nk)); labels.append(1)
    if not rows:
        return np.zeros((0, F.N_FEATURES), np.float32), np.zeros(0, np.int8)
    return np.vstack(rows).astype(np.float32), np.asarray(labels, np.int8)


def _sub_arrays(src, ids):
    """Return (ids, names, addrs, countries) parallel lists for a subset of ids."""
    row = _index_ids(src["ids"])
    rr = [row[i] for i in ids]
    return (list(ids),
            [src["names"][r] for r in rr],
            [src["addrs"][r] for r in rr],
            [src["countries"][r] for r in rr])


def candidate_recall(cand_list, truth):
    """Macro / micro candidate recall over non-singleton entities."""
    macro, cov, tot = [], 0, 0
    for qid, cands, _ in cand_list:
        t = truth.get(qid, set())
        if not t:
            continue
        cset = set(cands)
        hit = len(t & cset)
        macro.append(hit / len(t))
        cov += hit; tot += len(t)
    return {
        "macro_recall": float(np.mean(macro)) if macro else 0.0,
        "micro_recall": (cov / tot) if tot else 0.0,
        "n_nonsingleton": len(macro),
    }


def score_candidates(model, q_recs, cand_list, docprep):
    """Predict match prob for every candidate. Returns {qid: [(cid, prob)]}."""
    rows, idx_map = [], []
    for qid, cands, scores in cand_list:
        qr = q_recs[qid]
        nk = len(cands)
        for rank, (cid, sc) in enumerate(zip(cands, scores)):
            rows.append(F.pair_features(qr, docprep.get(cid), float(sc), nk, rank))
            idx_map.append((qid, cid))
    if not rows:
        return {}
    X = np.vstack(rows).astype(np.float32)
    probs = M.predict_proba(model, X)
    grouped = {}
    for (qid, cid), p in zip(idx_map, probs):
        grouped.setdefault(qid, []).append((cid, float(p)))
    return grouped


def train_and_validate(data_dir, model_path, run_log, seed=42, val_frac=0.15,
                       train_n=150000, val_n=40000, K=50, cap=20000,
                       use_pair=True, use_addr=True, neg_cap=40):
    """Fit the model on train entities and report held-out macro-F0.5."""
    import random
    rng = random.Random(seed)
    s1p, s2p, s3p = D.source_paths(data_dir, "train")
    with U.Timer("load train sources"):
        s1 = D.read_source(s1p)
        docs = _merged_docs(D.read_source(s2p), D.read_source(s3p))
        gt = D.read_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    U.log(f"train S1={len(s1['ids'])} docs={len(docs['ids'])} gt={len(gt)}")

    train_ids, val_ids = U.entity_split(s1["ids"], val_frac, seed)
    train_ids = [i for i in train_ids if i in gt]
    val_ids = [i for i in val_ids if i in gt]
    train_sub = train_ids if len(train_ids) <= train_n else rng.sample(train_ids, train_n)
    val_sub = list(val_ids) if len(val_ids) <= val_n else rng.sample(list(val_ids), val_n)

    with U.Timer("build doc index"):
        idx = B.build_doc_index(docs["ids"], docs["names"], docs["addrs"],
                                docs["countries"], use_pair, use_addr)
    docprep = DocPrep(docs)

    with U.Timer("training pairs + fit"):
        tids, tn, ta, tc = _sub_arrays(s1, train_sub)
        cand_tr = list(B.query_candidates(idx, tids, tn, ta, tc, K=K, cap=cap,
                                          use_pair=use_pair, use_addr=use_addr))
        q_recs_tr = _q_recs(s1, train_sub)
        X, y = build_training_pairs(q_recs_tr, cand_tr, gt, docprep, neg_cap)
        U.log(f"train pairs X={X.shape} pos={int(y.sum())} neg={int((y == 0).sum())}")
        model = M.train(X, y, seed)
        M.save(model, model_path)

    with U.Timer("validate"):
        vids, vn, va, vc = _sub_arrays(s1, val_sub)
        cand_va = list(B.query_candidates(idx, vids, vn, va, vc, K=K, cap=cap,
                                          use_pair=use_pair, use_addr=use_addr))
        truth_va = {q: set(gt.get(q, [])) for q in val_sub}
        crec = candidate_recall(cand_va, truth_va)
        q_recs_va = _q_recs(s1, val_sub)
        grouped = score_candidates(model, q_recs_va, cand_va, docprep)
        best_tau, best_f, table = MT.sweep_threshold(grouped, truth_va, val_sub)
        pred = MT.decide(grouped, best_tau, val_sub)
        m = E.macro_f_beta(pred, truth_va)
        meta = {q: vc[k] for k, q in enumerate(vids)}
        report = E.detailed_report(pred, truth_va, meta)

    fi = None
    try:
        n_s = min(20000, len(X))
        sel = rng.sample(range(len(X)), n_s) if len(X) > n_s else range(len(X))
        sel = list(sel)
        fi = M.feature_importance(model, F.FEATURE_NAMES, X[sel], y[sel])
    except Exception as e:
        U.log(f"feature importance skipped: {e}")

    result = {
        "phase": "train_and_validate", "seed": seed, "K": K, "cap": cap,
        "use_pair": use_pair, "use_addr": use_addr, "neg_cap": neg_cap,
        "n_train_s1": len(train_sub), "n_val_s1": len(val_sub),
        "train_pairs": int(X.shape[0]), "train_pos": int(y.sum()),
        "candidate_recall": crec, "best_tau": best_tau,
        "val_f0.5": m["f_beta"], "val_precision": m["precision"],
        "val_recall": m["recall"], "detailed": report,
        "sweep_table": table,
        "top_features": fi[:15] if fi else None,
    }
    U.append_run_log(run_log, result)
    return result, model, best_tau


def predict_test(data_dir, out_dir, model, tau, K=50, cap=20000,
                 use_pair=True, use_addr=True, batch=100000):
    """Run full test inference and write the two submission TSVs (streaming)."""
    os.makedirs(out_dir, exist_ok=True)
    s1p, s2p, s3p = D.source_paths(data_dir, "test")
    with U.Timer("load test sources"):
        s1 = D.read_source(s1p)
        docs = _merged_docs(D.read_source(s2p), D.read_source(s3p))
    U.log(f"test S1={len(s1['ids'])} docs={len(docs['ids'])}")
    with U.Timer("build test doc index"):
        idx = B.build_doc_index(docs["ids"], docs["names"], docs["addrs"],
                                docs["countries"], use_pair, use_addr)
    docprep = DocPrep(docs)

    ids, names = s1["ids"], s1["names"]
    addrs, countries = s1["addrs"], s1["countries"]
    n = len(ids)
    match_path = os.path.join(out_dir, "matching_results.tsv")
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    n_match, n_singleton, n_cand_total = 0, 0, 0

    with open(match_path, "w", encoding="utf-8", newline="") as mf, \
         open(cand_path, "w", encoding="utf-8", newline="") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for b0 in range(0, n, batch):
            b1 = min(n, b0 + batch)
            bids, bn, ba, bc = (ids[b0:b1], names[b0:b1], addrs[b0:b1], countries[b0:b1])
            cand_batch = list(B.query_candidates(
                idx, bids, bn, ba, bc, K=K, cap=cap,
                batch=b1 - b0, use_pair=use_pair, use_addr=use_addr))
            q_recs = {}
            for i, qid in enumerate(bids):
                q_recs[qid] = F.prep_record(bn[i], ba[i], bc[i])
            grouped = score_candidates(model, q_recs, cand_batch, docprep)
            for qid, cands, scores in cand_batch:
                cand_ids = list(dict.fromkeys(cands))          # de-dupe, keep order
                n_cand_total += len(cand_ids)
                scored = grouped.get(qid, [])
                matched = [cid for cid, p in scored if p >= tau]
                matched = list(dict.fromkeys(matched))
                cf.write(qid + "\t" + ",".join(cand_ids) + "\n")
                mf.write(qid + "\t" + ",".join(matched) + "\n")
                n_match += len(matched)
                if not matched:
                    n_singleton += 1
            docprep.clear()
            U.log(f"  test {b1}/{n}  matches={n_match} singletons={n_singleton}")
    return {
        "n_test_s1": n, "n_pred_matches": n_match, "n_singletons": n_singleton,
        "n_candidates_total": n_cand_total, "tau": tau, "K": K,
        "matching_path": match_path, "candidate_path": cand_path,
    }

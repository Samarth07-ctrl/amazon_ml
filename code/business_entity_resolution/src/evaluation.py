"""Evaluation: the official macro F0.5 metric, computed per Source-1 entity.

F0.5 = (1.25 * P * R) / (0.25 * P + R), computed per S1 entity then averaged
over ALL S1 entities (singletons included). A true singleton scores 1.0 iff we
predict the empty set, else 0.0.
"""
import numpy as np


def entity_f_beta(pred_set, true_set, beta=0.5):
    """F_beta for one entity given predicted / true match ID sets."""
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0  # correct singleton
    if not pred_set:
        return 0.0, 0.0, 1.0  # missed everything (recall 0); precision undefined->0
    if not true_set:
        return 0.0, 1.0, 0.0  # predicted matches for a true singleton
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    b2 = beta * beta
    f = (1 + b2) * precision * recall / (b2 * precision + recall)
    return f, precision, recall


def macro_f_beta(pred, truth, beta=0.5):
    """pred, truth: dict {s1_id: set(ids)}. Averaged over truth's keys."""
    fs, ps, rs = [], [], []
    for s1, tset in truth.items():
        pset = pred.get(s1, set())
        f, p, r = entity_f_beta(pset, tset, beta)
        fs.append(f); ps.append(p); rs.append(r)
    return {
        "f_beta": float(np.mean(fs)),
        "precision": float(np.mean(ps)),
        "recall": float(np.mean(rs)),
        "n": len(fs),
    }


def detailed_report(pred, truth, meta=None, beta=0.5):
    """Break F0.5 down by singleton / non-singleton / country / source mix."""
    groups = {"all": [], "singleton": [], "non_singleton": [],
              "multi_match": [], "has_s2": [], "has_s3": []}
    country_groups = {}
    for s1, tset in truth.items():
        pset = pred.get(s1, set())
        f, p, r = entity_f_beta(pset, tset, beta)
        groups["all"].append(f)
        if not tset:
            groups["singleton"].append(f)
        else:
            groups["non_singleton"].append(f)
            if len(tset) > 1:
                groups["multi_match"].append(f)
            if any(x.startswith("S2-") for x in tset):
                groups["has_s2"].append(f)
            if any(x.startswith("S3-") for x in tset):
                groups["has_s3"].append(f)
        if meta is not None:
            co = meta.get(s1, "?")
            country_groups.setdefault(co, []).append(f)
    out = {k: (float(np.mean(v)) if v else None, len(v)) for k, v in groups.items()}
    out["by_country"] = {k: (float(np.mean(v)), len(v)) for k, v in country_groups.items()}
    return out

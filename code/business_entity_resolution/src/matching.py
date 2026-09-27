"""Per-entity decision / thresholding and singleton handling.

The model outputs a match probability per (S1, candidate) pair. Because the
metric is macro-F0.5 (precision-weighted) computed per S1 entity, we choose the
decision rule that maximises validation F0.5 rather than pair accuracy:

  * accept every candidate with prob >= tau  (multi-match friendly),
  * an entity with no accepted candidate is predicted a singleton (empty set).

We tune a single global tau (simple + robust), and also support an optional
high-precision override for near-exact evidence. Thresholds are swept against
the true F0.5 metric on held-out entities.
"""
import numpy as np
from collections import defaultdict

from evaluation import macro_f_beta


def group_scores(s1_ids, cand_ids, probs):
    """Return {s1_id: [(cand_id, prob), ...]}."""
    g = defaultdict(list)
    for s, c, p in zip(s1_ids, cand_ids, probs):
        g[s].append((c, p))
    return g


def decide(grouped, tau, all_s1, margin=None, min_keep=0.0):
    """Apply threshold tau. grouped: {s1:[(cand,prob)]}. all_s1: iterable of
    every S1 id that must appear. Returns {s1: set(matched)}."""
    pred = {}
    for s1 in all_s1:
        cands = grouped.get(s1, [])
        keep = {c for c, p in cands if p >= tau}
        pred[s1] = keep
    return pred


def sweep_threshold(grouped, truth, all_s1, taus=None, beta=0.5):
    """Sweep tau, return (best_tau, best_score, table)."""
    if taus is None:
        taus = np.round(np.arange(0.10, 0.951, 0.025), 3)
    table = []
    best = (None, -1.0)
    for tau in taus:
        pred = decide(grouped, tau, all_s1)
        m = macro_f_beta(pred, truth, beta)
        table.append((float(tau), m["f_beta"], m["precision"], m["recall"]))
        if m["f_beta"] > best[1]:
            best = (float(tau), m["f_beta"])
    return best[0], best[1], table


def sweep_threshold_per_source(grouped, truth, all_s1, taus=None, beta=0.5):
    """Independent tau for S2 vs S3 candidates. Returns (tau2, tau3, score)."""
    if taus is None:
        taus = np.round(np.arange(0.15, 0.901, 0.05), 3)
    best = (None, None, -1.0)
    for t2 in taus:
        for t3 in taus:
            pred = {}
            for s1 in all_s1:
                keep = set()
                for c, p in grouped.get(s1, []):
                    thr = t2 if c.startswith("S2-") else t3
                    if p >= thr:
                        keep.add(c)
                pred[s1] = keep
            m = macro_f_beta(pred, truth, beta)
            if m["f_beta"] > best[2]:
                best = (float(t2), float(t3), m["f_beta"])
    return best

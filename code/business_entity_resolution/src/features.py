"""Pairwise feature engineering for (Source-1, candidate) record pairs.

Features are computed from the multiple normalized representations produced by
``normalization``. We rely on rapidfuzz (MIT) for fast C-level string metrics.
The output is a fixed-order float vector per pair; ``FEATURE_NAMES`` documents
the columns so the model and any analysis stay in sync.
"""
import numpy as np
from collections import namedtuple
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import normalization as N

FEATURE_NAMES = [
    # name
    "n_core_eq", "n_sqz_eq", "n_tok_jaccard", "n_tok_shared", "n_tok_containment",
    "n_ratio", "n_token_set_ratio", "n_token_sort_ratio", "n_partial_ratio",
    "n_jw", "n_len_absdiff", "n_len_ratio", "n_prefix", "n_num_overlap",
    # address
    "a_both_null", "a_one_null", "a_tok_jaccard", "a_tok_shared", "a_ratio",
    "a_token_set_ratio", "a_postal_eq", "a_postal_any", "a_num_overlap",
    "a_word_shared", "a_len_ratio",
    # cross / provenance
    "country_eq", "name_x_addr", "name_strong", "addr_strong",
    "both_strong", "both_weak", "block_score", "block_nkeys", "cand_rank",
]
N_FEATURES = len(FEATURE_NAMES)


def _jacc(a, b):
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _containment(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _num_overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / (max(len(a), len(b)) + 1e-6)


# Lightweight (tuple-backed) record representation -- far cheaper than a dict
# when millions of prepped documents are cached in memory.
Rec = namedtuple("Rec", [
    "core", "sqz", "tokset", "nums", "country",
    "a_tokset", "a_core", "a_postals", "a_nums", "a_words", "a_null",
])


def prep_record(name, addr, country):
    """Precompute all representations for one record (cache these)."""
    nr = N.name_reprs(name)
    ar = N.addr_reprs(addr)
    nums = frozenset(t for t in nr["tokens"] if t.isdigit())
    return Rec(
        core=nr["core"], sqz=nr["sqz"], tokset=nr["tokset"],
        nums=nums, country=country,
        a_tokset=ar["tokset"], a_core=ar["core"], a_postals=ar["postals"],
        a_nums=ar["nums"], a_words=ar["words"], a_null=ar["is_null"],
    )


def pair_features(r1, r2, block_score=0.0, block_nkeys=0, cand_rank=0):
    """Return a float32 feature vector (length N_FEATURES) for a record pair."""
    f = np.empty(N_FEATURES, dtype=np.float32)
    c1, c2 = r1.core, r2.core
    s1, s2 = r1.sqz, r2.sqz
    t1, t2 = r1.tokset, r2.tokset
    # ---- name ----
    f[0] = 1.0 if c1 == c2 and c1 else 0.0
    f[1] = 1.0 if s1 == s2 and s1 else 0.0
    f[2] = _jacc(t1, t2)
    shared = len(t1 & t2)
    f[3] = shared
    f[4] = _containment(t1, t2)
    f[5] = fuzz.ratio(c1, c2) / 100.0
    f[6] = fuzz.token_set_ratio(c1, c2) / 100.0
    f[7] = fuzz.token_sort_ratio(c1, c2) / 100.0
    f[8] = fuzz.partial_ratio(s1, s2) / 100.0
    f[9] = JaroWinkler.normalized_similarity(s1, s2)
    f[10] = abs(len(s1) - len(s2))
    f[11] = min(len(s1), len(s2)) / (max(len(s1), len(s2)) + 1e-6)
    f[12] = _prefix_len(s1, s2)
    f[13] = _num_overlap(r1.nums, r2.nums)
    # ---- address ----
    a1n, a2n = r1.a_null, r2.a_null
    f[14] = 1.0 if a1n and a2n else 0.0
    f[15] = 1.0 if (a1n ^ a2n) else 0.0
    at1, at2 = r1.a_tokset, r2.a_tokset
    f[16] = _jacc(at1, at2)
    f[17] = len(at1 & at2)
    f[18] = fuzz.ratio(r1.a_core, r2.a_core) / 100.0 if not (a1n or a2n) else 0.0
    f[19] = fuzz.token_set_ratio(r1.a_core, r2.a_core) / 100.0 if not (a1n or a2n) else 0.0
    p1, p2 = r1.a_postals, r2.a_postals
    f[20] = 1.0 if (p1 & p2) else 0.0
    f[21] = 1.0 if (p1 and p2) else 0.0
    f[22] = _num_overlap(r1.a_nums, r2.a_nums)
    f[23] = len(r1.a_words & r2.a_words)
    la1, la2 = len(r1.a_core), len(r2.a_core)
    f[24] = min(la1, la2) / (max(la1, la2) + 1e-6)
    # ---- cross / provenance ----
    name_sim = f[9]  # jaro-winkler on squeezed name
    addr_sim = f[16]
    f[25] = 1.0 if r1.country == r2.country else 0.0
    f[26] = name_sim * addr_sim
    ns = 1.0 if name_sim >= 0.9 else 0.0
    as_ = 1.0 if addr_sim >= 0.6 else 0.0
    f[27] = ns
    f[28] = as_
    f[29] = 1.0 if (ns and as_) else 0.0
    f[30] = 1.0 if (name_sim < 0.7 and addr_sim < 0.3) else 0.0
    f[31] = block_score
    f[32] = block_nkeys
    f[33] = cand_rank
    return f


"""Scalable multi-stage blocking / candidate generation.

Pure-Python inverted-index scoring does not scale to the ~1.7M Source-1 queries
against ~10M Source-2/3 documents in the test split. We use a vectorised,
sort/join-based blocker built on numpy int64 arrays:

  1. Feature keys per record are hashed to int64 (blake2b, 8 bytes), so all
     downstream work is integer array ops rather than Python dict lookups.
  2. The document side is indexed once: (feat_key, doc_idx) postings sorted by
     feat_key, with per-key document frequency and IDF.
  3. Queries are scored in batches. For each batch we join query postings
     against the sorted doc index on feat_key, expand matched blocks into
     candidate (query, doc) pairs weighted by IDF, sum weights per pair, and
     keep the top-K docs per query.

Very common keys (df above `cap`) are skipped for scoring: they add candidates
without discriminating and blow up the join. Country is encoded into every key,
which is open-set safe -- no country is ever enumerated, an unseen country such
as France simply gets its own key namespace.

Keys emitted per record (all prefixed by the record's raw country string):
  t|<co>|<token>        name token (>=3 chars)
  t|<co>|<squeezed>     squeezed token (collapses repeated chars; bridges
                        transliteration noise across scripts / reorderings)
  p|<co>|<a>~<b>        sorted pair of squeezed tokens (bounded count)
  z|<co>|<postal>       address postal code
  zn|<co>|<num>|<tok>   street number + a name token
"""
import hashlib
import numpy as np

import normalization as N

MAX_PAIR_TOKS = 5   # form pairs from at most this many squeezed tokens
MIN_TOK_LEN = 3
MAX_PAIRS = 40_000_000   # cap materialised (query,doc) pairs per join step


def _h(s):
    """8-byte blake2b hash of a key string -> signed int64."""
    d = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big", signed=True)


def record_keys(name, addr, country, use_pair=True, use_addr=True):
    """Return a de-duplicated list of int64 feature keys for one record."""
    co = country or "?"
    nr = N.name_reprs(name)
    keys = set()
    sqs = []
    for tok in nr["tokset"]:
        if len(tok) < MIN_TOK_LEN:
            continue
        keys.add(_h("t|" + co + "|" + tok))
        sq = N.squeeze(tok)
        if len(sq) >= MIN_TOK_LEN:
            keys.add(_h("t|" + co + "|" + sq))
            sqs.append(sq)
        else:
            sqs.append(tok)
    if use_pair and len(sqs) > 1:
        sl = sorted(set(sqs))[:MAX_PAIR_TOKS]
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                keys.add(_h("p|" + co + "|" + sl[i] + "~" + sl[j]))
    if use_addr:
        ar = N.addr_reprs(addr)
        for p in ar["postals"]:
            keys.add(_h("z|" + co + "|" + p))
        top_sqs = sorted(set(sqs))[:2]
        for nm in ar["nums"]:
            for sq in top_sqs:
                keys.add(_h("zn|" + co + "|" + nm + "|" + sq))
    return list(keys)


def _all_postings(names, addrs, countries, use_pair, use_addr, chunk=500_000):
    """Generate flat (key, row) postings arrays for a set of records.

    Accumulates into flat Python lists per chunk and converts to numpy once per
    chunk, so we never hold millions of tiny numpy arrays (memory) nor billions
    of boxed Python ints (also memory) at once.
    """
    key_arrs, row_arrs = [], []
    kbuf, rbuf = [], []
    n = len(names)
    for row in range(n):
        ks = record_keys(names[row], addrs[row], countries[row], use_pair, use_addr)
        if ks:
            kbuf.extend(ks)
            rbuf.extend([row] * len(ks))
        if len(kbuf) >= chunk:
            key_arrs.append(np.asarray(kbuf, dtype=np.int64))
            row_arrs.append(np.asarray(rbuf, dtype=np.int32))
            kbuf, rbuf = [], []
    if kbuf:
        key_arrs.append(np.asarray(kbuf, dtype=np.int64))
        row_arrs.append(np.asarray(rbuf, dtype=np.int32))
    if not key_arrs:
        return np.empty(0, np.int64), np.empty(0, np.int32)
    return np.concatenate(key_arrs), np.concatenate(row_arrs)


class DocIndex:
    """Sorted inverted index over the document (S2+S3) side."""

    def __init__(self, ukeys, kstart, kcount, docs, idf, ids):
        self.ukeys = ukeys      # int64[U], sorted unique feature keys
        self.kstart = kstart    # int64[U], offset of each key's block in docs
        self.kcount = kcount    # int64[U], df of each key
        self.docs = docs        # int32[P], doc rows grouped by key
        self.idf = idf          # float32[U]
        self.ids = ids          # list[str], doc row -> entity id

    @property
    def n_docs(self):
        return len(self.ids)


def build_doc_index(ids, names, addrs, countries, use_pair=True, use_addr=True):
    """Build a DocIndex from parallel document lists.

    Written to bound peak RAM on the ~10M-doc / ~80M-posting side: intermediate
    arrays are freed with explicit ``del`` as soon as they are consumed, so we
    never hold the raw postings, the sort permutation, and the sorted copy all
    at once.
    """
    keys, rows = _all_postings(names, addrs, countries, use_pair, use_addr)
    order = np.argsort(keys, kind="quicksort")   # grouping only; stability unneeded,
                                                  # and quicksort skips timsort's O(n) buffer
    keys = keys[order]
    docs = rows[order]
    del rows, order                       # free raw postings + permutation (~1 GB)
    ukeys, kstart, kcount = np.unique(keys, return_index=True, return_counts=True)
    del keys                              # sorted keys no longer needed
    n_docs = len(ids)
    idf = np.log(1.0 + n_docs / kcount.astype(np.float64)).astype(np.float32)
    return DocIndex(ukeys, kstart.astype(np.int64), kcount.astype(np.int64),
                    docs, idf, list(ids))


def _score_batch(index, qkeys, qrows, n_q, K, cap):
    """Vectorised IDF join + top-K for one query batch.

    qkeys/qrows: flat query postings (local query row in 0..n_q-1).
    Returns list length n_q of (doc_rows int32[], scores float64[]) top-K each,
    each ordered by descending score.
    """
    empty = (np.empty(0, np.int32), np.empty(0, np.float64))
    if len(qkeys) == 0:
        return [empty for _ in range(n_q)]
    pos = np.searchsorted(index.ukeys, qkeys)
    pos = np.clip(pos, 0, len(index.ukeys) - 1)
    valid = (index.ukeys[pos] == qkeys) & (index.kcount[pos] <= cap)
    if not valid.any():
        return [empty for _ in range(n_q)]
    total = int(index.kcount[pos[valid]].sum())
    # guard: if the batch would materialise too many pairs, split the queries
    if total > MAX_PAIRS and n_q > 1:
        mid = n_q // 2
        left_m = qrows < mid
        left = _score_batch(index, qkeys[left_m], qrows[left_m], mid, K, cap)
        right_m = ~left_m
        right = _score_batch(index, qkeys[right_m], qrows[right_m] - mid,
                             n_q - mid, K, cap)
        return left + right
    pos = pos[valid]
    qr = qrows[valid]
    counts = index.kcount[pos]                       # docs contributed per posting
    total = int(counts.sum())
    # ragged expansion of each key's doc block
    out_start = np.zeros(len(counts) + 1, np.int64)
    np.cumsum(counts, out=out_start[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(out_start[:-1], counts)
    doc_slot = np.repeat(index.kstart[pos], counts) + within
    d = index.docs[doc_slot].astype(np.int64)        # global doc rows
    q = np.repeat(qr.astype(np.int64), counts)
    w = np.repeat(index.idf[pos], counts).astype(np.float64)
    # sum weights per (query, doc)
    code = q * index.n_docs + d
    order = np.argsort(code, kind="stable")
    code_s = code[order]
    w_s = w[order]
    bnd = np.empty(len(code_s), bool)
    bnd[0] = True
    np.not_equal(code_s[1:], code_s[:-1], out=bnd[1:])
    seg_start = np.nonzero(bnd)[0]
    score = np.add.reduceat(w_s, seg_start)
    ucode = code_s[seg_start]
    uq = ucode // index.n_docs
    ud = (ucode % index.n_docs).astype(np.int32)
    # top-K per query: sort by (query asc, score desc)
    order2 = np.lexsort((-score, uq))
    uq_s = uq[order2]
    ud_s = ud[order2]
    sc_s = score[order2]
    gb = np.empty(len(uq_s), bool)
    gb[0] = True
    np.not_equal(uq_s[1:], uq_s[:-1], out=gb[1:])
    grp_start_idx = np.nonzero(gb)[0]
    rank = np.arange(len(uq_s), dtype=np.int64) - np.repeat(
        grp_start_idx, np.diff(np.append(grp_start_idx, len(uq_s))))
    keep = rank < K
    uq_s = uq_s[keep]
    ud_s = ud_s[keep]
    sc_s = sc_s[keep]
    out = [empty for _ in range(n_q)]
    kb = np.empty(len(uq_s), bool)
    if len(uq_s):
        kb[0] = True
        np.not_equal(uq_s[1:], uq_s[:-1], out=kb[1:])
        starts = np.nonzero(kb)[0]
        ends = np.append(starts[1:], len(uq_s))
        for s, e in zip(starts, ends):
            out[int(uq_s[s])] = (ud_s[s:e], sc_s[s:e])
    return out


def query_candidates(index, ids, names, addrs, countries, K=50, cap=20000,
                     batch=100000, use_pair=True, use_addr=True):
    """Yield (query_id, [candidate_doc_ids], [scores]) for every query record.

    candidate_doc_ids / scores are ordered by descending block score, so the
    position of a candidate in the list is its ``cand_rank``.
    """
    n = len(names)
    for b0 in range(0, n, batch):
        b1 = min(n, b0 + batch)
        qkeys, qrows = _all_postings(
            names[b0:b1], addrs[b0:b1], countries[b0:b1], use_pair, use_addr)
        tops = _score_batch(index, qkeys, qrows, b1 - b0, K, cap)
        for i in range(b1 - b0):
            rows, scores = tops[i]
            yield ids[b0 + i], [index.ids[r] for r in rows], scores

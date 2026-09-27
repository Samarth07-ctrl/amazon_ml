"""Small shared helpers: timing, logging, and an experiment/run log."""
import sys
import time
import json
import os


def log(msg):
    """Timestamped stderr log (stdout is reserved for results)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        log(f"{self.label}: {time.time() - self.t:.1f}s")


def entity_split(s1_ids, val_frac=0.15, seed=42):
    """Deterministic entity-level train/val split of Source-1 ids.

    Split is on the S1 entity, so no S1 appears in both sides -- this prevents
    leakage of an entity's match decisions between train and validation.
    """
    import random
    rng = random.Random(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val = set(ids[:n_val])
    train = ids[n_val:]
    return train, val


def append_run_log(path, record):
    """Append a JSON line to the experiment/run log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(record)
    record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tsv(path, header, rows):
    """Write rows (iterable of tuples) as a TSV with the given header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

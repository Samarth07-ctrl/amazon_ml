"""Data loading utilities for the entity-resolution pipeline.

All files are tab-separated with columns:
    entity_id, business_name, business_address, country
Ground truth: source1_entity_id, matched_entity_ids (comma list, may be empty).

The dataset is large (~10M Source-2/3 records), so loaders return compact
column-parallel Python lists / numpy arrays rather than pandas frames where it
matters for memory, and support streaming.
"""
import csv
import os

csv.field_size_limit(10_000_000)


def read_source(path):
    """Read a source TSV. Returns dict of parallel lists: ids, names, addrs, countries."""
    ids, names, addrs, countries = [], [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()  # skip header
        for line in f:
            # manual split is ~2x faster than csv for this simple 4-col schema
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                # pad malformed rows (missing address etc.)
                parts = parts + [""] * (4 - len(parts))
            ids.append(parts[0])
            names.append(parts[1])
            addrs.append(parts[2])
            countries.append(parts[3])
    return {"ids": ids, "names": names, "addrs": addrs, "countries": countries}


def read_ground_truth(path):
    """Return dict {source1_id: [matched ids...]} (empty list for singletons)."""
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, tab, rest = line.partition("\t")
            if not tab and not s1.strip():
                continue
            rest = rest.rstrip("\n")
            gt[s1.strip()] = [x for x in rest.split(",") if x] if rest.strip() else []
    return gt


def source_paths(data_dir, split):
    """Return (s1, s2, s3) paths for split in {'train','test'}."""
    pre = f"{split}_"
    d = os.path.join(data_dir, split)
    return (
        os.path.join(d, pre + "source1.tsv"),
        os.path.join(d, pre + "source2.tsv"),
        os.path.join(d, pre + "source3.tsv"),
    )

# Business Entity Resolution

Matches each **Source-1** business entity to zero, one, or many records in
**Source-2** and **Source-3**, optimising **macro F0.5 per Source-1 entity**
(precision-weighted; a correct empty prediction for a true singleton scores
1.0). Uses only the provided challenge data — no external databases, geocoders,
or network lookups anywhere in the pipeline, including during experimentation.

## Approach

```
 raw TSVs ─▶ normalization ─▶ blocking (candidate generation)
                                   │
                                   ▼
                         pairwise features ─▶ GBDT match model
                                   │
                                   ▼
                     F0.5-tuned threshold ─▶ per-entity decision
                                   │
                                   ▼
              matching_results.tsv  +  candidate_pairs.tsv
```

1. **Normalization** (`normalization.py`) — Unicode NFKC, transliteration
   (Unidecode romanizes native Indic/other scripts to Latin, matching the
   Latin-only Source-1), accent stripping, legal-suffix removal (multi-locale:
   ltd/pvt/llc/inc/sarl/sas/…), and a `squeeze` representation that collapses
   runs of repeated characters to bridge transliteration noise. Several
   representations are kept per record (tokens, token set, core string,
   squeezed, sorted-join) so we never over-collapse.

2. **Blocking** (`blocking.py`) — a scalable, vectorised inverted index built
   entirely on numpy int64 arrays (feature keys hashed with blake2b). Keys are
   country-prefixed name tokens, squeezed tokens, token pairs, postal codes,
   and street-number+token combinations. Candidates are scored by summed IDF
   and the top-K per entity are kept. Very common keys (df > cap) are skipped.
   Country is encoded into every key, so the blocker is **open-set safe** — an
   unseen country (e.g. France in the test split) simply gets its own key
   namespace; nothing is hard-coded to {US, India}.

3. **Features** (`features.py`) — 34 pairwise features over the multiple
   representations: name similarity (exact/squeezed equality, token Jaccard /
   containment, rapidfuzz ratio / token_set / token_sort / partial, Jaro-
   Winkler, length ratios, prefix, numeric overlap), address similarity
   (null-handling, token Jaccard, postal equality, number/word overlap), and
   cross/provenance features (country match, name×address interaction, block
   score, block key count, candidate rank).

4. **Model** (`model.py`) — scikit-learn `HistGradientBoostingClassifier`
   (BSD-3; well within the MIT/Apache-2.0, ≤8B-parameter rule). GBDTs handle
   the mixed-scale, non-linear similarity features with no feature scaling.
   Trained on GT positives plus hard negatives mined from blocking.

5. **Matching** (`matching.py`) — the decision threshold τ is swept directly
   against the true macro-F0.5 metric on a held-out, **entity-level**
   validation split (no S1 entity appears in both train and validation).
   Every candidate with prob ≥ τ is accepted (multi-match friendly); an entity
   with no accepted candidate is predicted a singleton (empty set).

6. **Evaluation** (`evaluation.py`) — the official macro-F0.5, plus a breakdown
   by singleton / multi-match / source-mix / country.

## Layout

```
src/
  normalization.py   text normalization + multiple representations
  data_loader.py     TSV / ground-truth readers
  blocking.py        vectorised IDF top-K candidate generation
  features.py        34 pairwise features (rapidfuzz)
  model.py           HistGradientBoosting train / predict / persist
  matching.py        F0.5-tuned thresholding + singleton handling
  evaluation.py      macro-F0.5 metric + detailed report
  pipeline.py        train_and_validate + predict_test orchestration
  utils.py           timing, entity split, run log, TSV writer
  main.py            CLI entry point
  sanity_check.py    independent submission-structure checker
requirements.txt
README.md
```

## Reproduce

From the repository root (the folder containing `utils/` and `output/`):

```bash
pip install -r code/business_entity_resolution/requirements.txt

# Train + validate (entity-level split), then run full test inference.
# Writes output/model.joblib, output/matching_results.tsv, output/candidate_pairs.tsv
python code/business_entity_resolution/src/main.py --mode all --K 50

# Validate the submission (official + independent checker)
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  utils/datasets/test
python code/business_entity_resolution/src/sanity_check.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  utils/datasets/test
```

`--mode train` fits and validates only; `--mode predict` runs inference from a
saved model. `--K` sets candidates per entity; `--tau` overrides the tuned
threshold. All randomness is seeded (`--seed`, default 42).

## Scale notes

Train has 2.21M Source-1 and 10.32M Source-2/3 records; test has 1.73M and
9.97M. Blocking is fully vectorised (sort/join over integer arrays) to handle
this within ~16 GB RAM; feature reprs for documents are computed lazily and
cached per batch so the ~10M records are never all materialised as objects at
once.

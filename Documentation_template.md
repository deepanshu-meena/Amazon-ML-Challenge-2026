# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [YOUR TEAM NAME]
**Team Members:** [LIST YOUR TEAM MEMBERS]
**Submission Date:** September 27, 2026

---

## 1. Executive Summary

We built a two-tier entity resolution pipeline: a full blocking+LightGBM
matcher for offline validation, and a fast exact-match fallback (normalized
business name + country, with generic-key pruning) used for the final
test-set submission under time constraints. The exact-match approach
prioritizes precision, which the F_0.5 metric weights 2x over recall.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data (2.2M Source-1, 5.0M Source-2, 5.3M Source-3
records) confirmed the noise patterns described in the problem statement:
legal-suffix variation (Corp/Corporation, Pvt/Private), punctuation and
`&`/`and` inconsistencies, and address abbreviation variants (Rd/Road).
Country is US/India in training and adds France in test, so no country
value is ever hard-coded -- it is treated as an open string field
throughout.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (validated pipeline), with a
rule-based exact-match fallback used for the submitted test predictions.

**Core Innovation:** Vectorized, merge-based blocking (pandas hash joins
instead of per-row Python loops) so candidate generation scales to
multi-million-row sources; a hard absolute cap on any single blocking key's
postings (not just a percentage) to prevent common tokens from exploding
the candidate set; two-pass global-frequency pruning on the exact-match
fallback so extremely generic business names cannot become false-merge
sources.

---

## 3. Candidate Generation (Blocking)

**Full pipeline (validated on a held-out training slice):**
- Blocking keys used: normalized name tokens, name character trigrams,
  address tokens + numeric/PIN tokens, exact normalized name -- each
  partitioned by country, with generic keys above a frequency cap dropped
  before candidate retrieval.
- Candidate pairs generated: ~20-30 per Source-1 entity after top-K
  truncation (K=10-15).
- Recall preserved via multiple independent blocking signals (token +
  trigram + address + exact-name), so a weak signal on one field does not
  eliminate a true match; measured blocking recall ceiling of 0.84-0.85 on
  a real 20k-entity training slice.

**Final submission (exact-match fallback, used due to time constraints):**
- Blocking key: normalized business name + country (exact match only).
- Two-pass global frequency count per key; any key shared by more than 3
  "other-source" records is dropped as too generic to auto-match on name
  alone, before the merge is performed -- this is what keeps precision
  high despite using no classifier.
- Result on the real test set: 998,255 / 1,732,544 Source-1 entities
  matched, ~1.96 matches per matched entity (a sane ratio, not an
  over-matching artifact).

---

## 4. Matching Model

**Full pipeline features (validated, not used in final run due to time):**
- Name features: token Jaccard, character-trigram Jaccard, Levenshtein
  ratio, token-sort ratio, partial ratio, length ratio.
- Address features: token Jaccard, Levenshtein ratio, token-sort ratio,
  length ratio.
- Other: country match (binary), character n-gram TF-IDF cosine similarity.

**Model type:** LightGBM (gradient-boosted trees; MIT-licensed, well under
the 8B-parameter cap), `scale_pos_weight` set from the natural class
imbalance.

**Threshold selection method:** Grid search directly against macro F_0.5
on a held-out validation split of Source-1 entities (not rows, to avoid
leakage).

**Final submission model type:** none -- exact-match blocking with
frequency-based pruning was used directly as the match decision, chosen
under time pressure over waiting for the full classifier pipeline to
complete on the full ~10M-record test set.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), full pipeline, validation split:** 0.92 (measured
  on a real 20k-entity slice of the training data, not synthetic).
- **Final submission:** exact-match fallback; validation F_0.5 not
  separately measured for this variant due to time constraints -- it
  trades recall (only catches near-identical names) for precision (very
  low false-merge rate by construction, since matches require an exact
  normalized name in a non-generic key).
- **Common false positives (wrong merges):** possible for two distinct
  businesses that happen to share an identical normalized name and country
  but are different real-world entities (e.g. franchise locations) -- rare
  given the frequency cap, but not eliminated by it.
- **Common false negatives (missed matches):** any true match with a
  non-identical normalized name (typos, abbreviation differences not
  covered by our normalization dictionary, word-order changes) is missed
  by the exact-match fallback -- this is the main gap versus the full
  blocking+classifier pipeline, which is designed to catch these via
  token/trigram/TF-IDF similarity.

---

## 6. Conclusion

Our validated pipeline (vectorized blocking + LightGBM) reaches macro
F_0.5 = 0.92 on held-out training data, but multi-million-row full-dataset
runtime exceeded our available submission window. We fell back to a fast,
precision-focused exact-match approach to guarantee a valid, non-trivial
submission, matching 998,255 of 1,732,544 test entities. The key lesson:
validate pipeline runtime at full data scale early, not just on samples --
and always keep a fast, simple fallback ready given a precision-heavy
metric rewards conservative matching even from simpler methods.

---

## Appendix

### A. Code Artefacts

Full runnable pipeline under `code/business_entity_resolution/`:
- `src/text_utils.py`, `src/data_io.py` -- normalization and loading
- `src/blocking.py` -- vectorized merge-based candidate generation
- `src/features.py`, `src/tfidf_utils.py` -- pairwise similarity features
- `src/train.py`, `src/predict.py` -- LightGBM training/threshold tuning
  and test inference (the validated pipeline)
- `fast_baseline.py` -- the exact-match fallback actually used for the
  submitted `matching_results.tsv`
- `utils/validate_submission.py` -- official validator, run before
  submission (PASS)

To reproduce the full pipeline: see `README.md` in that folder.
To reproduce the submitted result: `python fast_baseline.py --test-dir dataset/test --out-dir output_fallback`.

### B. Additional Results

Full-pipeline validation run (real training data, 20k Source-1 entities):
```
blocking recall ceiling = 0.84-0.85
avg candidates/entity   = 20-30
validation macro F_0.5  = 0.92
```

Final submission run (real test data, full 1,732,544 Source-1 entities):
```
entities matched: 998,255 / 1,732,544
avg matches per matched entity: 1.96
runtime: ~16 minutes (two-pass chunked processing, Windows laptop)
```

---

**Note:** exact-match fallback was a deliberate scope trade-off under a
hard deadline, not the intended final approach -- the blocking+LightGBM
pipeline in `src/` is the one designed to be competitive on the private
leaderboard given more runtime budget.

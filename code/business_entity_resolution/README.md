# Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline: normalize → block (candidate generation) → feature
engineer → classify → threshold-tune for macro F_0.5 → write
`matching_results.tsv` + `candidate_pairs.tsv`.

## 1. Setup

All commands below assume you're running from **this directory**
(`code/business_entity_resolution/`).

```bash
pip install -r requirements.txt
```

Drop the real competition data in:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

(Not included here — the dataset is only accessible to team leaders via the
challenge portal.)

## 2. (Optional) Smoke-test on synthetic data first

Before you have the real dataset, or any time you change the code, sanity
check the whole pipeline runs cleanly:

```bash
python tests/make_synthetic_data.py
python -m src.train   --train-dir dataset/train --model-out artifacts/model.joblib
python -m src.predict --test-dir dataset/test   --model artifacts/model.joblib --out-dir output
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

This overwrites `dataset/` with tiny fake data — just to prove the plumbing
(normalization → blocking → features → model → output format) works, not to
estimate real performance. Delete `dataset/train` and `dataset/test` and
re-populate with the real files before your actual run.

## 3. Train

```bash
python -m src.train --train-dir dataset/train --model-out artifacts/model.joblib --top-k 15
```

Prints, in order:
- **blocking recall ceiling** — the max recall possible given your candidate
  set, *before* the classifier even runs. If this is low, no amount of model
  tuning will fix it — raise `--top-k` or loosen `--no-country-filter` first.
- **avg candidates/entity** — what you're being scored on beyond raw F_0.5.
  Lower is better as long as the recall ceiling stays high.
- **val_macro_F0.5** at the tuned threshold — your best local estimate of
  leaderboard score.

## 4. Predict on test set

```bash
python -m src.predict --test-dir dataset/test --model artifacts/model.joblib --out-dir output
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (inside
this directory). Copy those two files into the top-level `output/` folder of
the submission package before zipping — that top-level copy is what gets
uploaded to the leaderboard / reviewed.

## 5. Validate before you spend a submission

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Prefer the **official** `utils/validate_submission.py` shipped with the real
dataset if it differs from this reimplementation — this one mirrors the
rules in the problem statement PDF but hasn't been checked against their
exact script.

## 6. Local scoring (on a held-out labeled split, not the real test set)

`src/train.py` already reports val-split F_0.5 during training. If you want
to score an arbitrary prediction file against a ground-truth file directly:

```bash
python -m src.evaluate --pred output/matching_results.tsv --truth dataset/train/train_ground_truth.tsv
```

## Design summary

| Stage | Approach |
|---|---|
| Normalization | Legal-suffix expansion (Corp→Corporation, Pvt→Private, & → and), address abbreviation expansion (Rd→Road), diacritic folding for transliteration, lowercase/punctuation cleanup |
| Blocking | Inverted index on name tokens, name character-trigrams (typo tolerance), and address tokens/numerics; soft country-match filter (generic equality — no hard-coded country values, so unseen labels like France work unchanged); cheap Jaccard-based re-rank keeps only top-K per Source-1 entity |
| Features | Token Jaccard, trigram Jaccard, Levenshtein ratio, token-sort ratio, partial ratio (name + address), country match, length ratios, char n-gram TF-IDF cosine |
| Model | LightGBM (falls back to sklearn HistGradientBoosting if unavailable) with `scale_pos_weight` for the natural class imbalance |
| Threshold | Grid search on a held-out validation split of Source-1 **entities** (not rows, to avoid leakage) to directly maximize macro F_0.5 |
| Fair play | No external APIs/lookups anywhere in the pipeline — every signal comes from the provided TSVs |

## Tuning knobs if you have time before the deadline

- `--top-k`: raise if blocking recall ceiling < ~0.95; lower (after confirming
  ceiling stays high) since the challenge explicitly ranks smaller candidate
  sets higher.
- `--no-country-filter`: try if you suspect country labels are noisy/missing
  in some records and it's costing recall.
- Add more features in `src/features.py` (e.g. a small open-license sentence
  embedding model's cosine similarity) if you want to push F_0.5 further —
  keep any such model ≤8B params and MIT/Apache-2.0 licensed per the rules.

## Validated on the real data (not synthetic)

Ran `train.py` on a real 20k-entity slice of `train_source1.tsv` against the
real `train_source2.tsv`/`train_source3.tsv`:

```
blocking recall ceiling = 0.85   avg candidates/entity = 30
validation macro F_0.5  = 0.92
```

Full data is ~110x this slice (train_source1 = 2.2M rows, source2 = 5.0M,
source3 = 5.3M; test_source1 = 1.7M, source2 = 4.9M, source3 = 5.1M).

**Run the real thing on a proper machine, not a tiny sandbox.** An earlier
test attempt here hit an OOM trying to load two ~500MB TSVs into pandas at
once with only ~4GB RAM available — that's a hardware ceiling, not a code
bug. On a normal 8GB+ RAM machine it should run fine. If you do hit memory
pressure: lower `--top-k` first (directly shrinks the feature matrix), then
`--max-df-frac` (more aggressive generic-token pruning).

Given a hard deadline: start `train.py` on the real full dataset right away
-- candidate generation and feature building scale roughly linearly with
row count, so expect noticeably more than the 108 seconds the slice took
(exact multiplier depends on your CPU/RAM; a multi-core machine helps since
LightGBM's training step parallelizes). Kick it off, then use the wait time
to fill in `Documentation_template.md`.

### Recommended run for time pressure

```bash
python -m src.train   --train-dir dataset/train --model-out artifacts/model.joblib --top-k 10
python -m src.predict --test-dir dataset/test   --model artifacts/model.joblib --out-dir output
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

`--top-k 10` (vs the default 15) trades a little recall for meaningfully
fewer candidate pairs to score -- worth it under time pressure, and the
challenge already rewards a smaller candidate set. If train.py's printed
recall ceiling comes back uncomfortably low (well under 0.85), raise
`--top-k` back up before touching anything else -- blocking recall is the
hard ceiling on your final score, no matter how good the classifier is.

The bundled `utils/validate_submission.py` is the **real one from the
challenge package** (not a reimplementation) -- run it with `--check-ids`
only if you have memory to spare, since it loads all Source-2/3 IDs into
memory to do the check (off by default for exactly that reason).

"""
Train the matching model end-to-end from the training split.

Usage:
    python -m src.train --train-dir dataset/train --model-out artifacts/model.joblib --top-k 15

What it does:
  1. Loads train_source1/2/3 + train_ground_truth.
  2. Runs the SAME blocking code used at prediction time against Source2 and
     Source3 separately, merges candidates per Source-1 entity.
  3. Labels every (s1, candidate) pair using ground truth (1 if truly
     matched, 0 otherwise) -- this also measures the blocking recall
     ceiling, printed to stdout.
  4. Splits Source-1 ENTITIES (not rows -- avoids leakage) into train/val.
  5. Fits a TF-IDF vectorizer on train names, builds features for all pairs.
  6. Trains PairClassifier, then searches a decision threshold on the
     validation split to maximize macro F_0.5.
  7. Persists model + vectorizer + threshold + blocking config to disk.
"""
import argparse
import os
import random

import joblib
import numpy as np

from src.data_io import load_source, load_ground_truth, to_lookup
from src.blocking import generate_candidates, merge_candidate_dicts
from src.features import build_feature_matrix
from src.tfidf_utils import fit_vectorizer, cosine_lookup_for_pairs
from src.model_utils import PairClassifier, best_threshold, f_beta_macro


def build_labeled_pairs(cand_dict, truth):
    """cand_dict: s1_id -> [(cand_id, cheap_score), ...] -> list of (s1_id, cand_id, label)"""
    out = []
    for s1_id, cands in cand_dict.items():
        true_set = truth.get(s1_id, set())
        for cand_id, _ in cands:
            out.append((s1_id, cand_id, 1 if cand_id in true_set else 0))
    return out


def recall_ceiling(cand_dict, truth):
    hit, total = 0, 0
    for s1_id, true_set in truth.items():
        if not true_set:
            continue
        cand_ids = {c for c, _ in cand_dict.get(s1_id, [])}
        hit += len(true_set & cand_ids)
        total += len(true_set)
    return hit / total if total else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--model-out", default="artifacts/model.joblib")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-country-filter", action="store_true")
    ap.add_argument("--max-df-frac", type=float, default=0.03,
                     help="drop tokens appearing in more than this fraction of a country partition (generic-token guard at scale)")
    args = ap.parse_args()

    country_filter = not args.no_country_filter

    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    truth = load_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))

    print(f"[train] source1={len(s1)} source2={len(s2)} source3={len(s3)} labeled_s1={len(truth)}")

    cand_s2 = generate_candidates(s1, s2, "S2", top_k=args.top_k, country_filter=country_filter, max_df_frac=args.max_df_frac)
    cand_s3 = generate_candidates(s1, s3, "S3", top_k=args.top_k, country_filter=country_filter, max_df_frac=args.max_df_frac)
    cand_all = merge_candidate_dicts(cand_s2, cand_s3)

    ceiling = recall_ceiling(cand_all, truth)
    avg_cands = np.mean([len(v) for v in cand_all.values()]) if cand_all else 0.0
    print(f"[train] blocking recall ceiling={ceiling:.4f}  avg candidates/entity={avg_cands:.2f}")

    # --- group split by S1 entity id ---
    ids = list(s1["entity_id"])
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * args.val_frac)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    labeled = build_labeled_pairs(cand_all, truth)
    train_pairs = [(a, b, y) for (a, b, y) in labeled if a in train_ids]
    val_pairs = [(a, b, y) for (a, b, y) in labeled if a in val_ids]
    print(f"[train] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)}")

    s1_lookup = to_lookup(s1)
    other_lookup = to_lookup(s2)
    other_lookup.update(to_lookup(s3))

    vec = fit_vectorizer(list(s1["name_norm"]) + list(s2["name_norm"]) + list(s3["name_norm"]))

    def make_xy(pairs):
        pr = [(a, b) for a, b, _ in pairs]
        y = np.array([lab for _, _, lab in pairs], dtype=np.int32)
        cos_lookup = cosine_lookup_for_pairs(pr, s1_lookup, other_lookup, vec)
        X = build_feature_matrix(pr, s1_lookup, other_lookup, cos_lookup)
        return X, y

    X_train, y_train = make_xy(train_pairs)
    X_val, y_val = make_xy(val_pairs)

    clf = PairClassifier().fit(X_train, y_train)
    print(f"[train] trained backend={clf.backend}")

    val_probs = clf.predict_proba(X_val)
    probs_by_entity = {}
    for (a, b, _), p in zip(val_pairs, val_probs):
        probs_by_entity.setdefault(a, []).append((b, float(p)))

    val_truth = {k: v for k, v in truth.items() if k in val_ids}
    t, score = best_threshold(probs_by_entity, val_truth)
    print(f"[train] best_threshold={t:.2f}  val_macro_F0.5={score:.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({
        "classifier": clf,
        "vectorizer": vec,
        "threshold": t,
        "top_k": args.top_k,
        "country_filter": country_filter,
        "max_df_frac": args.max_df_frac,
    }, args.model_out)
    print(f"[train] saved -> {args.model_out}")


if __name__ == "__main__":
    main()

"""
Run the trained pipeline on the test set and write both required outputs:
    output/matching_results.tsv   (final matches -- scored)
    output/candidate_pairs.tsv    (blocking candidate set -- reviewed, not scored)

Usage:
    python -m src.predict --test-dir dataset/test --model artifacts/model.joblib --out-dir output
"""
import argparse
import os

import joblib

from src.data_io import load_source, to_lookup, write_id_list_tsv
from src.blocking import generate_candidates, merge_candidate_dicts
from src.features import build_feature_matrix
from src.tfidf_utils import fit_vectorizer, cosine_lookup_for_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--top-k", type=int, default=None, help="override saved top_k")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    clf = bundle["classifier"]
    threshold = bundle["threshold"]
    top_k = args.top_k or bundle["top_k"]
    country_filter = bundle["country_filter"]
    max_df_frac = bundle.get("max_df_frac", 0.03)

    s1 = load_source(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(args.test_dir, "test_source3.tsv"))

    print(f"[predict] source1={len(s1)} source2={len(s2)} source3={len(s3)}")

    cand_s2 = generate_candidates(s1, s2, "S2", top_k=top_k, country_filter=country_filter, max_df_frac=max_df_frac)
    cand_s3 = generate_candidates(s1, s3, "S3", top_k=top_k, country_filter=country_filter, max_df_frac=max_df_frac)
    cand_all = merge_candidate_dicts(cand_s2, cand_s3)

    s1_lookup = to_lookup(s1)
    other_lookup = to_lookup(s2)
    other_lookup.update(to_lookup(s3))

    # Fit a fresh TF-IDF space on the test corpus for the cosine feature
    # (see tfidf_utils docstring for why this is fine to refit per stage).
    vec = fit_vectorizer(list(s1["name_norm"]) + list(s2["name_norm"]) + list(s3["name_norm"]))

    all_pairs = []
    for s1_id, cands in cand_all.items():
        for cand_id, _ in cands:
            all_pairs.append((s1_id, cand_id))

    cos_lookup = cosine_lookup_for_pairs(all_pairs, s1_lookup, other_lookup, vec)
    X = build_feature_matrix(all_pairs, s1_lookup, other_lookup, cos_lookup) if all_pairs else None
    probs = clf.predict_proba(X) if X is not None and len(X) else []

    prob_map = {pair: p for pair, p in zip(all_pairs, probs)}

    os.makedirs(args.out_dir, exist_ok=True)

    candidate_rows = []
    matching_rows = []
    for row in s1.itertuples(index=False):
        s1_id = row.entity_id
        cands = cand_all.get(s1_id, [])
        cand_ids = sorted({c for c, _ in cands})  # dedup + stable order
        candidate_rows.append((s1_id, cand_ids))

        matched = sorted({
            cand_id for cand_id, _ in cands
            if prob_map.get((s1_id, cand_id), 0.0) >= threshold
        })
        matching_rows.append((s1_id, matched))

    write_id_list_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"),
                       "source1_entity_id", "candidate_entity_ids", candidate_rows)
    write_id_list_tsv(os.path.join(args.out_dir, "matching_results.tsv"),
                       "source1_entity_id", "matched_entity_ids", matching_rows)

    n_matched = sum(1 for _, m in matching_rows if m)
    print(f"[predict] wrote {len(matching_rows)} rows ({n_matched} with >=1 match) to {args.out_dir}/")


if __name__ == "__main__":
    main()

"""
Run this FIRST, before touching blocking/model code, once you have the real
data. Answers exactly the questions you need before choosing top_k,
max_df_frac, and whether the country filter is safe to use:

  - row counts per source, duplicate entity_ids, missing name/address/country
  - country label distribution (confirms the train/test open-set claim)
  - matches per Source-1 entity + singleton percentage (tells you how
    aggressive your threshold needs to be)
  - rough token-frequency check (which tokens would get pruned by
    --max-df-frac, so you can sanity check the cutoff isn't too aggressive)

Usage:
    python -m src.eda --train-dir dataset/train
"""
import argparse
import os
from collections import Counter

from src.data_io import load_source, load_ground_truth
from src.text_utils import tokenize


def describe_source(name, df):
    print(f"\n--- {name} ---")
    print(f"rows: {len(df)}")
    dup = df["entity_id"].duplicated().sum()
    print(f"duplicate entity_id: {dup}")
    for col in ("business_name", "business_address", "country"):
        blank = (df[col].astype(str).str.strip() == "").sum()
        print(f"blank {col}: {blank} ({blank/len(df):.2%})")
    print("country distribution:")
    for country, count in df["country"].value_counts().head(10).items():
        print(f"  {country or '(blank)'}: {count} ({count/len(df):.2%})")


def describe_ground_truth(truth, s1_ids):
    n = len(truth)
    n_singleton = sum(1 for v in truth.values() if not v)
    match_counts = [len(v) for v in truth.values()]
    avg_matches = sum(match_counts) / n if n else 0.0
    missing = set(s1_ids) - set(truth.keys())
    print(f"\n--- ground truth ---")
    print(f"labeled S1 entities: {n}  (S1 rows with no ground-truth row: {len(missing)})")
    print(f"singleton rate: {n_singleton}/{n} ({n_singleton/n:.2%})" if n else "no labels")
    print(f"avg matches per non-singleton entity: "
          f"{sum(match_counts)/max(1, n - n_singleton):.2f}")
    print(f"max matches for one entity: {max(match_counts) if match_counts else 0}")


def token_frequency_report(df, top=20):
    counter = Counter()
    for name in df["name_norm"]:
        for tok in set(tokenize(name, drop_stopwords=True, min_len=3)):
            counter[tok] += 1
    print(f"\n--- most common name tokens (n={len(df)}) ---")
    for tok, count in counter.most_common(top):
        print(f"  {tok}: {count} ({count/len(df):.2%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="dataset/train")
    args = ap.parse_args()

    s1 = load_source(os.path.join(args.train_dir, "train_source1.tsv"))
    s2 = load_source(os.path.join(args.train_dir, "train_source2.tsv"))
    s3 = load_source(os.path.join(args.train_dir, "train_source3.tsv"))
    truth_path = os.path.join(args.train_dir, "train_ground_truth.tsv")

    describe_source("Source 1", s1)
    describe_source("Source 2", s2)
    describe_source("Source 3", s3)

    if os.path.exists(truth_path):
        truth = load_ground_truth(truth_path)
        describe_ground_truth(truth, s1["entity_id"])

    token_frequency_report(s2, top=15)


if __name__ == "__main__":
    main()

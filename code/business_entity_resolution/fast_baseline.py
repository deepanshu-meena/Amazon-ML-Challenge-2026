"""
Fast, no-ML fallback: exact normalized-name + country matching.
Run this if the full train.py/predict.py pipeline won't finish in time.
No model training needed -- pure pandas merges, should finish in a few
minutes even on the full ~1.7M/4.9M/5.1M-row test set on real hardware.

Usage (run from code/business_entity_resolution/):
    python fast_baseline.py --test-dir dataset/test --out-dir output_fallback
"""
import argparse
import os
import time
from collections import Counter

import pandas as pd
from src.text_utils import normalize_series_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--out-dir", default="output_fallback")
    ap.add_argument("--chunk-size", type=int, default=500_000)
    ap.add_argument("--cap", type=int, default=3,
                     help="drop exact-name keys shared by more than this many 'other' records (too generic to auto-match)")
    args = ap.parse_args()

    t0 = time.time()
    s1 = pd.read_csv(os.path.join(args.test_dir, "test_source1.tsv"), sep="\t",
                      dtype=str, keep_default_na=False)
    s1["country_k"] = s1["country"].fillna("").str.strip().str.lower()
    s1["name_k"] = normalize_series_name(s1["business_name"])
    s1_key = s1[s1["name_k"] != ""][["entity_id", "country_k", "name_k"]]
    print(f"[fast_baseline] s1 ready: {len(s1)} rows, {time.time()-t0:.1f}s")

    matches = {eid: [] for eid in s1["entity_id"]}

    for fname in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(args.test_dir, fname)
        t1 = time.time()
        counter = Counter()
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                  chunksize=args.chunk_size,
                                  usecols=["entity_id", "business_name", "country"]):
            chunk["country_k"] = chunk["country"].fillna("").str.strip().str.lower()
            chunk["name_k"] = normalize_series_name(chunk["business_name"])
            vc = chunk[chunk["name_k"] != ""].groupby(["country_k", "name_k"]).size()
            for k, v in vc.items():
                counter[k] += v
        generic_keys = {k for k, v in counter.items() if v > args.cap}
        print(f"[fast_baseline] {fname} pass1: {len(counter)} distinct keys, "
              f"{len(generic_keys)} generic, {time.time()-t1:.1f}s")

        n_found = 0
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                  chunksize=args.chunk_size,
                                  usecols=["entity_id", "business_name", "country"]):
            chunk["country_k"] = chunk["country"].fillna("").str.strip().str.lower()
            chunk["name_k"] = normalize_series_name(chunk["business_name"])
            chunk_f = chunk[chunk["name_k"] != ""][["entity_id", "country_k", "name_k"]]
            idx = pd.MultiIndex.from_frame(chunk_f[["country_k", "name_k"]])
            chunk_f = chunk_f[~idx.isin(generic_keys)]
            merged = s1_key.merge(chunk_f, on=["country_k", "name_k"], suffixes=("_s1", "_o"))
            for s1id, oid in zip(merged["entity_id_s1"], merged["entity_id_o"]):
                matches[s1id].append(oid)
            n_found += len(merged)
        print(f"[fast_baseline] {fname} pass2: {n_found} matches, {time.time()-t1:.1f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "matching_results.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1["entity_id"]:
            ids = sorted(set(matches.get(eid, [])))
            f.write(f"{eid}\t{','.join(ids)}\n")
    with open(os.path.join(args.out_dir, "candidate_pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1["entity_id"]:
            ids = sorted(set(matches.get(eid, [])))
            f.write(f"{eid}\t{','.join(ids)}\n")

    n_matched = sum(1 for v in matches.values() if v)
    print(f"[fast_baseline] DONE in {time.time()-t0:.1f}s -- "
          f"{n_matched}/{len(s1)} entities matched -> {args.out_dir}/")


if __name__ == "__main__":
    main()

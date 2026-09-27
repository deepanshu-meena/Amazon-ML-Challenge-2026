"""
Dev-only helper: generates a tiny synthetic train+test set with the exact
column/format contract from the problem statement, so you can smoke-test
the whole pipeline (train -> predict -> evaluate -> validate_submission)
before the real dataset is available or while iterating on code.

This file is NOT part of the scored submission -- it's a testing aid.

Usage:
    python tests/make_synthetic_data.py
"""
import os
import random

NAMES = [
    ("Acme Corporation", "Acme Corp", "ACME CORP.", "Acme Cor"),
    ("Sunrise Bakery Pvt Ltd", "Sunrise Bakery Private Limited", "Sunrise Bakery", "Sunrize Bakery"),
    ("Global Tech Solutions Inc", "Global Tech Solutions", "Global Tech Solns Inc.", "GlobalTech Solutions"),
    ("Sharma & Sons Traders", "Sharma and Sons Traders", "Sharma Sons Trading Co", "Sharma Bros Traders"),
    ("Blue Ocean Logistics LLC", "Blue Ocean Logistics", "BlueOcean Logistcs", "Blue Ocean Freight LLC"),
]

ADDRESSES = [
    ("221B Baker Street, London", "221B Baker St, London", "221 B Baker Street London", "Near Baker St, London"),
    ("45 MG Road, Bengaluru", "45 M.G. Rd, Bangalore", "MG Road 45, Bengaluru", "45, MG Rd Bengaluru 560001"),
    ("500 5th Ave, New York", "500 Fifth Avenue, New York", "500 5th Av NY", "Five Hundred 5th Ave, New York"),
    ("12 Sector 18, Noida", "Sector 18, Plot 12, Noida", "12 Sec-18 Noida UP", "Near Metro, Sector 18 Noida"),
    ("77 Rue de Paris, Paris", "77 Rue De Paris", "77 R. de Paris, Paris FR", "Rue de Paris 77"),
]

COUNTRIES = ["US", "India", "US", "India", "France"]


def _rows(prefix, name_variants_idx, addr_variants_idx, country, n_start):
    return {
        "entity_id": f"{prefix}-{n_start:05d}",
        "business_name": NAMES[name_variants_idx][0 if prefix == "S1" else random.randint(1, 3)],
        "business_address": ADDRESSES[addr_variants_idx][0 if prefix == "S1" else random.randint(1, 3)],
        "country": country,
    }


def build_split(out_dir, n_entities=40, seed=0):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []

    for i in range(n_entities):
        name_idx = i % len(NAMES)
        addr_idx = i % len(ADDRESSES)
        country = COUNTRIES[i % len(COUNTRIES)]

        s1 = _rows("S1", name_idx, addr_idx, country, i + 1)
        s1_rows.append(s1)

        matched_ids = []
        # ~70% of entities get 1-2 matches in S2/S3, rest are singletons
        if random.random() < 0.7:
            s2 = _rows("S2", name_idx, addr_idx, country, i + 1)
            s2_rows.append(s2)
            matched_ids.append(s2["entity_id"])
            if random.random() < 0.4:
                s3 = _rows("S3", name_idx, addr_idx, country, i + 1)
                s3_rows.append(s3)
                matched_ids.append(s3["entity_id"])
        else:
            # add a noisy unrelated S2 record for negative signal
            other_idx = (name_idx + 1) % len(NAMES)
            s2 = _rows("S2", other_idx, addr_idx, country, i + 1)
            s2_rows.append(s2)

        gt_rows.append((s1["entity_id"], matched_ids))

    return s1_rows, s2_rows, s3_rows, gt_rows


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def write(rows, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                f.write(f"{r['entity_id']}\t{r['business_name']}\t{r['business_address']}\t{r['country']}\n")

    # --- train split ---
    random.seed(1)
    s1, s2, s3, gt = build_split(os.path.join(root, "dataset", "train"), n_entities=60, seed=1)
    train_dir = os.path.join(root, "dataset", "train")
    write(s1, os.path.join(train_dir, "train_source1.tsv"))
    write(s2, os.path.join(train_dir, "train_source2.tsv"))
    write(s3, os.path.join(train_dir, "train_source3.tsv"))
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matched in gt:
            f.write(f"{s1_id}\t{','.join(matched)}\n")

    # --- test split (no ground truth exposed to the pipeline) ---
    random.seed(2)
    s1t, s2t, s3t, gtt = build_split(os.path.join(root, "dataset", "test"), n_entities=25, seed=2)
    test_dir = os.path.join(root, "dataset", "test")
    write(s1t, os.path.join(test_dir, "test_source1.tsv"))
    write(s2t, os.path.join(test_dir, "test_source2.tsv"))
    write(s3t, os.path.join(test_dir, "test_source3.tsv"))
    # keep a private copy of test truth ONLY for our own smoke-test scoring
    with open(os.path.join(test_dir, "_synthetic_test_truth_DO_NOT_SUBMIT.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matched in gtt:
            f.write(f"{s1_id}\t{','.join(matched)}\n")

    print("Synthetic train/test data written under dataset/")


if __name__ == "__main__":
    main()

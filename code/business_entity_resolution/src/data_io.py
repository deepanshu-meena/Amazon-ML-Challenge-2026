import pandas as pd

from src.text_utils import normalize_series_name, normalize_series_address


def load_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("").str.strip()
    df["name_norm"] = normalize_series_name(df["business_name"])
    df["addr_norm"] = normalize_series_address(df["business_address"])
    return df


def load_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    def parse_ids(s):
        s = (s or "").strip()
        return set() if not s else set(x.strip() for x in s.split(",") if x.strip())

    gt = {}
    for row in df.itertuples(index=False):
        gt[row.source1_entity_id] = parse_ids(row.matched_entity_ids)
    return gt


def to_lookup(df):
    """entity_id -> {"name_norm":..., "addr_norm":..., "country":...}"""
    return df.set_index("entity_id")[["name_norm", "addr_norm", "country"]].to_dict("index")


def write_id_list_tsv(path, id_col_name, list_col_name, rows):
    """rows: list of (s1_entity_id, sorted list of matched/candidate ids)"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{id_col_name}\t{list_col_name}\n")
        for s1_id, ids in rows:
            f.write(f"{s1_id}\t{','.join(ids)}\n")

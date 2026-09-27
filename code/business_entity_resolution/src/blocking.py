"""
Blocking / candidate generation -- vectorized, merge-based (rewritten for
multi-million-row scale: train_source1 ~2.2M, source2/3 ~5M each).

Why this replaces the earlier per-row Python-loop + dict-of-sets index:
at this scale, iterating every Source-1 row in Python and doing per-
candidate rapidfuzz/jaccard calls inside that loop is the actual
bottleneck (tens of millions to billions of Python-level operations).
This version instead:

  1. Explodes each blocking signal (name tokens, name trigrams, address
     tokens/numerics, exact-normalized-name) into long
     "(entity_id, country_key, key)" tables for BOTH Source-1 and the
     other source, using vectorized pandas ops.
  2. Prunes overly generic keys (max_df_frac) via a vectorized groupby
     count -- this is what stops one common word from exploding the
     candidate set.
  3. Generates ALL candidate pairs for a given blocking signal with a
     single pandas merge on (country_key, key) -- a C-level hash join,
     not a Python loop.
  4. Concatenates the signals, and uses "how many independent signals
     agreed" as a free, vectorized coarse ranking score (more agreeing
     signals = more likely a real match), then truncates to top-K per
     Source-1 entity with a single sort + groupby.head().

Only after this truncation (Source-1 count x K pairs, not Source-1 x
Source-2/3) do the more expensive per-pair similarity features in
features.py run -- bounded and fast.
"""
from collections import defaultdict

import pandas as pd

from src.text_utils import tokenize, char_ngrams

UNKNOWN_COUNTRY_KEY = "_unknown"


def _country_key_series(country_series, country_filter):
    if not country_filter:
        return pd.Series([""] * len(country_series), index=country_series.index)
    ck = country_series.fillna("").astype(str).str.strip().str.lower()
    return ck.where(ck != "", UNKNOWN_COUNTRY_KEY)


def _numeric_tokens_list(text):
    return [t for t in text.split(" ") if any(ch.isdigit() for ch in t)]


def _explode_tokens(df, id_col, text_col, min_len, drop_stopwords=True):
    toks = df[text_col].apply(lambda s: tokenize(s, drop_stopwords=drop_stopwords, min_len=min_len))
    long = pd.DataFrame({
        id_col: df[id_col].values,
        "country_key": df["country_key"].values,
        "key": toks,
    }).explode("key")
    return long.dropna(subset=["key"])


def _explode_trigrams(df, id_col, text_col):
    tris = df[text_col].apply(lambda s: list(char_ngrams(s, n=3)))
    long = pd.DataFrame({
        id_col: df[id_col].values,
        "country_key": df["country_key"].values,
        "key": tris,
    }).explode("key")
    return long.dropna(subset=["key"])


def _explode_numeric(df, id_col, text_col):
    nums = df[text_col].apply(_numeric_tokens_list)
    long = pd.DataFrame({
        id_col: df[id_col].values,
        "country_key": df["country_key"].values,
        "key": nums,
    }).explode("key")
    return long.dropna(subset=["key"])


def _prune_generic_keys(long_other, max_df_frac, absolute_cap=150):
    """Drop (country_key, key) groups that cover too large a share of that
    country partition, OR exceed a hard absolute cap regardless of fraction
    -- vectorized groupby, no Python loop. The absolute cap is what actually
    prevents a merge blow-up on large country partitions (a 3% fractional
    cap on a 500k-record partition is still 15,000 postings per side,
    which squares into hundreds of millions of merged rows for a single
    common key)."""
    if long_other.empty:
        return long_other
    sizes = long_other.groupby("country_key")["entity_id"].transform("size")
    counts = long_other.groupby(["country_key", "key"])["entity_id"].transform("size")
    cap = (sizes * max_df_frac).clip(lower=15, upper=absolute_cap)
    return long_other[counts <= cap]


def _block_pairs(long_s1, long_other):
    """Inner-merge on (country_key, key) -> raw (s1_id, cand_id) pairs for one signal."""
    if long_s1.empty or long_other.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id"])
    merged = long_s1.merge(long_other, on=["country_key", "key"], suffixes=("_s1", "_o"))
    out = merged[["entity_id_s1", "entity_id_o"]].rename(
        columns={"entity_id_s1": "s1_id", "entity_id_o": "cand_id"}
    )
    return out.drop_duplicates()


def generate_candidates(s1_df, other_df, other_prefix, top_k=15,
                         country_filter=True, min_score=0.0, max_df_frac=0.01):
    """
    Returns: dict[s1_entity_id] -> list[(candidate_entity_id, score)]
    where score = number of independent blocking signals that agreed
    (an integer coarse-ranking proxy, not a similarity score -- the real
    similarity features are computed downstream in features.py on this
    truncated set).
    """
    s1 = s1_df.copy()
    other = other_df.copy()
    s1["country_key"] = _country_key_series(s1["country"], country_filter)
    other["country_key"] = _country_key_series(other["country"], country_filter)

    signals_s1 = {
        "name_tok": _explode_tokens(s1, "entity_id", "name_norm", min_len=3),
        "name_tri": _explode_trigrams(s1, "entity_id", "name_norm"),
        "addr_tok": pd.concat([
            _explode_tokens(s1, "entity_id", "addr_norm", min_len=4),
            _explode_numeric(s1, "entity_id", "addr_norm"),
        ], ignore_index=True),
    }
    signals_other = {
        "name_tok": _explode_tokens(other, "entity_id", "name_norm", min_len=3),
        "name_tri": _explode_trigrams(other, "entity_id", "name_norm"),
        "addr_tok": pd.concat([
            _explode_tokens(other, "entity_id", "addr_norm", min_len=4),
            _explode_numeric(other, "entity_id", "addr_norm"),
        ], ignore_index=True),
    }

    # exact-normalized-name block: strong, cheap, high-precision signal
    exact_s1 = s1[s1["name_norm"] != ""][["entity_id", "country_key", "name_norm"]].rename(
        columns={"name_norm": "key"})
    exact_other = other[other["name_norm"] != ""][["entity_id", "country_key", "name_norm"]].rename(
        columns={"name_norm": "key"})

    all_pairs = []
    for name in ("name_tok", "name_tri", "addr_tok"):
        pruned_other = _prune_generic_keys(signals_other[name], max_df_frac)
        pairs = _block_pairs(signals_s1[name], pruned_other)
        if not pairs.empty:
            all_pairs.append(pairs)

    exact_pairs_other = _prune_generic_keys(exact_other, max_df_frac)
    exact_pairs = _block_pairs(exact_s1, exact_pairs_other)
    if not exact_pairs.empty:
        all_pairs.append(exact_pairs)

    if not all_pairs:
        return {eid: [] for eid in s1_df["entity_id"]}

    combined = pd.concat(all_pairs, ignore_index=True)
    scored = combined.groupby(["s1_id", "cand_id"]).size().reset_index(name="score")

    if min_score > 0:
        scored = scored[scored["score"] >= min_score]

    scored = scored.sort_values(["s1_id", "score"], ascending=[True, False])
    truncated = scored.groupby("s1_id").head(top_k)

    results = defaultdict(list)
    for row in truncated.itertuples(index=False):
        results[row.s1_id].append((row.cand_id, int(row.score)))

    # ensure every Source-1 entity has an entry, even with zero candidates
    for eid in s1_df["entity_id"]:
        if eid not in results:
            results[eid] = []

    return dict(results)


def merge_candidate_dicts(dict_a, dict_b):
    """Merge two {s1_id: [(cand_id, score), ...]} dicts (union of keys)."""
    merged = defaultdict(list)
    for d in (dict_a, dict_b):
        for k, v in d.items():
            merged[k].extend(v)
    return dict(merged)

"""
Pairwise feature engineering.

Computes a feature vector for each (source1_entity_id, candidate_entity_id)
pair. Designed to be cheap enough to run over the *candidate* set (already
small thanks to blocking), not the full cross product.
"""
import numpy as np
from rapidfuzz import fuzz

from src.text_utils import tokenize, char_ngrams, jaccard

FEATURE_NAMES = [
    "name_token_jaccard",
    "name_trigram_jaccard",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "addr_token_jaccard",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "country_match",
    "name_len_ratio",
    "addr_len_ratio",
    "tfidf_cosine",
]


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


def compute_pair_features(s1_name, s1_addr, s1_country,
                           c_name, c_addr, c_country,
                           tfidf_cosine=0.0):
    f = {
        "name_token_jaccard": jaccard(set(tokenize(s1_name)), set(tokenize(c_name))),
        "name_trigram_jaccard": jaccard(char_ngrams(s1_name), char_ngrams(c_name)),
        "name_levenshtein_ratio": fuzz.ratio(s1_name, c_name) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(s1_name, c_name) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(s1_name, c_name) / 100.0,
        "addr_token_jaccard": jaccard(set(tokenize(s1_addr)), set(tokenize(c_addr))),
        "addr_levenshtein_ratio": fuzz.ratio(s1_addr, c_addr) / 100.0,
        "addr_token_sort_ratio": fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0,
        "country_match": 1.0 if (s1_country and c_country and s1_country == c_country) else 0.0,
        "name_len_ratio": _len_ratio(s1_name, c_name),
        "addr_len_ratio": _len_ratio(s1_addr, c_addr),
        "tfidf_cosine": tfidf_cosine,
    }
    return [f[name] for name in FEATURE_NAMES]


def build_feature_matrix(pairs, s1_lookup, other_lookup, tfidf_cosine_lookup=None):
    """
    pairs: list of (s1_id, cand_id)
    s1_lookup / other_lookup: dict[entity_id] -> {"name_norm":..., "addr_norm":..., "country":...}
    tfidf_cosine_lookup: optional dict[(s1_id, cand_id)] -> float
    Returns: np.ndarray of shape (len(pairs), len(FEATURE_NAMES))
    """
    rows = []
    for s1_id, cand_id in pairs:
        s1 = s1_lookup[s1_id]
        c = other_lookup[cand_id]
        cos = 0.0
        if tfidf_cosine_lookup is not None:
            cos = tfidf_cosine_lookup.get((s1_id, cand_id), 0.0)
        rows.append(compute_pair_features(
            s1["name_norm"], s1["addr_norm"], s1["country"],
            c["name_norm"], c["addr_norm"], c["country"],
            tfidf_cosine=cos,
        ))
    return np.array(rows, dtype=np.float32)

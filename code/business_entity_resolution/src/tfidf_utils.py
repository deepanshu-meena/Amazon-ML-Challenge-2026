"""
Character n-gram TF-IDF cosine similarity, computed only for the pairs we
actually need (the candidate set), not the full cross product.

We fit one TfidfVectorizer per pipeline run over the union of Source-1,
Source-2 and Source-3 normalized names available at that stage (train-time
fit uses only train data; predict-time fit uses only test data -- the
vectorizer is a lightweight feature, not a persisted piece of "the model",
so refitting per stage is fine and keeps things simple/self-contained).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def fit_vectorizer(all_names):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vec.fit(all_names)
    return vec


def cosine_lookup_for_pairs(pairs, s1_lookup, other_lookup, vectorizer):
    """
    pairs: list of (s1_id, cand_id)
    Returns dict[(s1_id, cand_id)] -> cosine similarity (float)
    """
    if not pairs:
        return {}

    s1_ids = [p[0] for p in pairs]
    cand_ids = [p[1] for p in pairs]

    s1_names = [s1_lookup[i]["name_norm"] for i in s1_ids]
    cand_names = [other_lookup[i]["name_norm"] for i in cand_ids]

    s1_mat = vectorizer.transform(s1_names)
    cand_mat = vectorizer.transform(cand_names)

    # row-wise cosine similarity between s1_mat[i] and cand_mat[i]
    sims = np.asarray(s1_mat.multiply(cand_mat).sum(axis=1)).flatten()
    s1_norm = np.sqrt(np.asarray(s1_mat.multiply(s1_mat).sum(axis=1)).flatten())
    cand_norm = np.sqrt(np.asarray(cand_mat.multiply(cand_mat).sum(axis=1)).flatten())
    denom = s1_norm * cand_norm
    denom[denom == 0] = 1e-9
    sims = sims / denom

    return {(s1_ids[i], cand_ids[i]): float(sims[i]) for i in range(len(pairs))}

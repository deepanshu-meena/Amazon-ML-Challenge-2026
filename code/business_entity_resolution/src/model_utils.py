"""
Model wrapper: LightGBM if available (fast, handles imbalance well via
scale_pos_weight, MIT-licensed, well under the 8B-parameter cap by many
orders of magnitude), falling back to sklearn's HistGradientBoostingClassifier
(BSD-licensed, zero extra install) so the pipeline runs anywhere.
"""
import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from sklearn.ensemble import HistGradientBoostingClassifier


class PairClassifier:
    def __init__(self):
        self.backend = "lightgbm" if _HAS_LGB else "sklearn_hgb"
        self.model = None

    def fit(self, X, y):
        if _HAS_LGB:
            n_pos = max(1, int(y.sum()))
            n_neg = max(1, len(y) - n_pos)
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                scale_pos_weight=n_neg / n_pos,
                random_state=42,
            )
            self.model.fit(X, y)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                random_state=42,
                class_weight="balanced",
            )
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def f_beta_macro(pred_dict, truth_dict, beta=0.5):
    """
    pred_dict / truth_dict: dict[s1_entity_id] -> set(matched_ids)
    Every key in truth_dict must be scored (missing -> treated as empty pred).
    """
    beta2 = beta * beta
    scores = []
    for s1_id, true_set in truth_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        f = (1 + beta2) * precision * recall / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0
        scores.append(f)
    return float(np.mean(scores)) if scores else 0.0


def best_threshold(probs_by_entity, truth_dict, thresholds=None):
    """
    probs_by_entity: dict[s1_id] -> list[(cand_id, prob)]
    Search thresholds to maximize macro F_0.5.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = {s1: {cid for cid, p in cands if p >= t} for s1, cands in probs_by_entity.items()}
        score = f_beta_macro(pred, truth_dict)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score

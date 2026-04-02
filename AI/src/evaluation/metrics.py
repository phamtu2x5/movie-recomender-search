from math import sqrt

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def hit_rate_at_k(relevant_items: set[int], ranked_items: list[int], k: int = 10) -> float:
    top_k = ranked_items[:k]
    return float(any(item in relevant_items for item in top_k))


def recall_at_k(relevant_items: set[int], ranked_items: list[int], k: int = 10) -> float:
    if not relevant_items:
        return 0.0
    top_k = set(ranked_items[:k])
    return float(len(relevant_items.intersection(top_k)) / len(relevant_items))


def ndcg_at_k(relevant_items: set[int], ranked_items: list[int], k: int = 10) -> float:
    top_k = ranked_items[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)

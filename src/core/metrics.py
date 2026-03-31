import math
import numpy as np


def recall_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    if not relevant:
        return None

    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def dcg_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    dcg = 0.0
    for idx, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1 / math.log2(idx + 1)
    return dcg


def ndcg_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    if not relevant:
        return None

    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0:
        return 0.0

    return dcg_at_k(recommended, relevant, k) / idcg


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def diversity_at_k(recommended: list, item_genre_vectors: dict, k: int = 10) -> float:
    recs = recommended[:k]

    if len(recs) < 2:
        return 0.0

    pairwise_distances = []

    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            item_i = recs[i]
            item_j = recs[j]

            vec_i = item_genre_vectors.get(item_i)
            vec_j = item_genre_vectors.get(item_j)

            if vec_i is None or vec_j is None:
                continue

            sim = cosine_similarity(vec_i, vec_j)
            dist = 1.0 - sim
            pairwise_distances.append(dist)

    if not pairwise_distances:
        return 0.0

    return float(np.mean(pairwise_distances))

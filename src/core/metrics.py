import math
import numpy as np


def recall_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    '''Calculates the recall@k metric for a single user.'''

    # If there are no relevant items, return None to indicate that recall is undefined
    if not relevant:
        return None

    # Consider only the top-K recommended items for the recall calculation
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def dcg_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    '''Calculates the DCG@k metric for a single user.'''

    # If there are no relevant items, return 0.0 since DCG is zero when
    # there are no relevant items
    dcg = 0.0
    if not relevant:
        return dcg

    # Calculating the DCG by iterating over the top-K recommended items and summing
    # the discounted gains for the relevant items based on their positions in the ranking
    for idx, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1 / math.log2(idx + 1)
    return dcg


def ndcg_at_k(recommended: list, relevant: list, k: int = 10) -> float:
    '''Calculates the NDCG@k metric for a single user.'''

    # If there are no relevant items, return None to indicate that NDCG is undefined
    if not relevant:
        return None

    # Calculating the ideal DCG (IDCG) for the user based on the number of relevant
    # items and the position of those items in an ideal ranking
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    # If the ideal DCG is zero (which can happen if there are no relevant items),
    # return 0.0 to avoid division by zero
    if idcg == 0:
        return 0.0

    # Calculating the DCG for the recommended items and normalizing it by the
    # ideal DCG to get NDCG
    return dcg_at_k(recommended, relevant, k) / idcg


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    '''Calculates the cosine similarity between two vectors.'''
    
    # Calculating the cosine similarity by computing the dot product of the two vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    # If either vector has zero norm, return 0.0 to indicate that the similarity
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Returning the cosine similarity as the dot product of the vectors divided
    # by the product of their norms
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def diversity_at_k(recommended: list, item_genre_vectors: dict, k: int = 10) -> float:
    '''
    Calculates the diversity@k metric for a single user based on the genre vectors
    of the recommended items. It computes the average pairwise distance between
    the genre vectors of the top-K recommended items, where the distance is defined
    as 1 minus the cosine similarity. A higher diversity score indicates that the
    recommended items are more diverse in terms of their genres.
    '''
    
    # Consider only the top-K recommended items for the diversity calculation
    recs = recommended[:k]

    # If there are fewer than 2 recommended items, return 0.0 since diversity
    # cannot be calculated for a single item
    if len(recs) < 2:
        return 0.0

    # Calculating the pairwise distances between the genre vectors of the recommended items
    pairwise_distances = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            # Getting the genre vectors for the two items being compared
            item_i = recs[i]
            item_j = recs[j]

            # If either item does not have a genre vector, skip the similarity calculation
            vec_i = item_genre_vectors.get(item_i)
            vec_j = item_genre_vectors.get(item_j)
            if vec_i is None or vec_j is None:
                continue

            # Calculating the cosine similarity between the genre vectors and converting
            # it to a distance and adding it to the list of pairwise distances
            sim = cosine_similarity(vec_i, vec_j)
            dist = 1.0 - sim
            pairwise_distances.append(dist)

    # If there are no valid pairwise distances, return 0.0 to indicate
    # that diversity cannot be calculated
    if not pairwise_distances:
        return 0.0

    # Returning the average pairwise distance as the diversity score
    return float(np.mean(pairwise_distances))

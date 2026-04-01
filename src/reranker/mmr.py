import pandas as pd

from core.config import Config

from core.metrics import cosine_similarity, diversity_at_k, ndcg_at_k, recall_at_k
from core.structs import (
    GeneralVariables,
    LTR_Variables,
    MF_Variables,
    PopularityVariables,
    ResultBundle,
)
from core.candidates import get_candidates
from rankers.mf_general import predict_mf
from rankers.pairwise_ltr import predict_pairwise_ltr


def get_top_m_candidates(
    config: Config,
    general_vars: GeneralVariables,
    method_vars: MF_Variables | LTR_Variables,
    user_id: int,
    predict_func: callable,
    popularity_vars: PopularityVariables | None = None,
) -> list[tuple[int, float]]:
    candidates = get_candidates(general_vars, user_id)

    scored_items = [None] * len(candidates)
    for i, item_id in enumerate(candidates):
        if popularity_vars is None:
            score = predict_func(general_vars, method_vars.model, user_id, item_id)
        else:
            score = predict_func(
                config, general_vars, popularity_vars, method_vars, user_id, item_id
            )

        scored_items[i] = (item_id, score)

    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return ranked_items[: config.TOP_M]


def item_similarity_by_genre(
    general_vars: GeneralVariables,
    item_a: int,
    item_b: int,
) -> float:
    vec_a = general_vars.item_genre_vectors.get(item_a)
    vec_b = general_vars.item_genre_vectors.get(item_b)

    if vec_a is None or vec_b is None:
        return 0.0

    return cosine_similarity(vec_a, vec_b)


def mmr_rerank(
    config: Config,
    general_vars: GeneralVariables,
    scored_candidates: list[tuple[int, float]],
    alpha: float,
) -> list[int]:
    selected = []
    selected_ids = set()

    target_k = config.TOP_K

    while len(selected) < target_k and len(selected) < len(scored_candidates):
        best_item = None
        best_mmr_score = -float("inf")

        for item_id, rel_score in scored_candidates:
            item_id = int(item_id)

            if item_id in selected_ids:
                continue

            if len(selected) == 0:
                mmr_score = rel_score
            else:
                max_sim = max(
                    item_similarity_by_genre(general_vars, item_id, selected_item)
                    for selected_item in selected
                )
                mmr_score = (1 - alpha) * rel_score - alpha * max_sim

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_item = item_id

        if best_item is None:
            break

        selected.append(best_item)
        selected_ids.add(best_item)

    return selected


def print_mmr_example(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables | None,
    method_vars: MF_Variables | LTR_Variables,
    user_id: int,
) -> None:
    top_m_mf = get_top_m_candidates(
        config,
        general_vars,
        method_vars,
        user_id,
        predict_func=predict_mf if popularity_vars is None else predict_pairwise_ltr,
        popularity_vars=popularity_vars,
    )

    print("User:", user_id)
    for alpha in config.MMR_ALPHA_VALUES:
        mmr_recs = mmr_rerank(config, general_vars, top_m_mf, alpha=alpha)
        print(f"{method_vars.method_name.upper()} + MMR alpha={alpha}:", mmr_recs)


def evaluate_mmr(
    config: Config,
    general_vars: GeneralVariables,
    method_vars: MF_Variables | LTR_Variables,
    predict_func: callable,
    popularity_vars: PopularityVariables | None = None,
) -> ResultBundle:

    mmr_results = []

    for alpha in config.MMR_ALPHA_VALUES:
        if config.PRINT_CONFIRM:
            print(f"Evaluating MMR with alpha={alpha} for method {method_vars.method_name}...")
        for user_id in general_vars.eligible_users:
            top_m_candidates = get_top_m_candidates(
                config, general_vars, method_vars, user_id, predict_func, popularity_vars
            )
            recommended = mmr_rerank(config, general_vars, top_m_candidates, alpha=alpha)

            relevant = general_vars.relevant_items_by_user[user_id]

            recall = recall_at_k(recommended, relevant, k=config.TOP_K)
            ndcg = ndcg_at_k(recommended, relevant, k=config.TOP_K)
            diversity = diversity_at_k(recommended, general_vars.item_genre_vectors, k=config.TOP_K)

            mmr_results.append(
                {
                    "user_id": int(user_id),
                    "method": f"mmr_{method_vars.method_name}",
                    "alpha": alpha,
                    "top_k": recommended,
                    "recall@10": recall,
                    "ndcg@10": ndcg,
                    "diversity@10": diversity,
                }
            )

    result_bundle = ResultBundle.from_rows(mmr_results)

    if config.PRINT_CONFIRM:
        print(result_bundle.df.head())

    return result_bundle

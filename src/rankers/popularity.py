from core.config import Config
from core.structs import GeneralVariables, PopularityVariables, ResultBundle

from core.candidates import get_candidates

from core.metrics import recall_at_k, ndcg_at_k, diversity_at_k


def get_item_popularity(general_vars: GeneralVariables) -> dict:
    item_popularity = general_vars.train_df.groupby("item_id").size().to_dict()
    return item_popularity


def popularity_score(popularity_vars: PopularityVariables, item_id: int) -> int:
    return popularity_vars.item_popularity.get(item_id, 0)


def recommend_popularity(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    user_id: int,
) -> list:
    candidates = get_candidates(general_vars, user_id)
    ranked_items = sorted(
        candidates, key=lambda item_id: popularity_score(popularity_vars, item_id), reverse=True
    )
    return ranked_items[: config.TOP_K]


def evaluate_popularity(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
) -> ResultBundle:
    results = [{} for _ in range(len(general_vars.eligible_users))]

    for user_id in general_vars.eligible_users:
        recommended = recommend_popularity(config, general_vars, popularity_vars, user_id)
        relevant = general_vars.relevant_items_by_user[user_id]

        recall = recall_at_k(recommended, relevant, k=config.TOP_K)
        ndcg = ndcg_at_k(recommended, relevant, k=config.TOP_K)
        diversity = diversity_at_k(recommended, general_vars.item_genre_vectors, k=config.TOP_K)

        results[general_vars.eligible_users.index(user_id)] = {
            "user_id": user_id,
            "method": "popularity",
            "top_k": recommended,
            "recall@10": recall,
            "ndcg@10": ndcg,
            "diversity@10": diversity,
        }

    result_bundle = ResultBundle.from_rows(results)
    if config.PRINT_CONFIRM:
        print(result_bundle.df.head())

    return result_bundle


def print_examples_popularity(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    user_id: int,
):
    example_recs = recommend_popularity(config, general_vars, popularity_vars, user_id)

    print("User:", user_id)
    print("Top-10 item ids:", example_recs)

    print(
        general_vars.items[general_vars.items["item_id"].isin(example_recs)][["item_id", "title"]]
    )

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
    """
    Retrieves the top-M candidate items for a given user based on the specified
    prediction function.
    """

    # Getting candidate items that the user has not interacted with
    candidates = get_candidates(general_vars, user_id)

    # Scoring each candidate item using the provided prediction function
    scored_items = [None] * len(candidates)
    for i, item_id in enumerate(candidates):
        if popularity_vars is None:
            score = predict_func(general_vars, method_vars.model, user_id, item_id)
        else:
            score = predict_func(
                config, general_vars, popularity_vars, method_vars, user_id, item_id
            )

        # Storing the item_id and its corresponding score as a tuple in the scored_items list
        scored_items[i] = (item_id, score)

    # Sorting the scored items by their relevance scores in descending order
    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    top_m = ranked_items[: config.TOP_M]

    # Optionally normalizing the relevance scores of the top-M candidates if specified in the config
    if config.MMR_NORMALIZE_REL:
        top_m = normalize_rel_scores(top_m)

    # Returning the top-M scored candidates as a list of tuples (item_id, score)
    return top_m


def normalize_rel_scores(
    scored_candidates: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Normalizes the relevance scores of the scored candidates to a [0, 1] range using min-max normalization."""

    # If there are no scored candidates, return the empty list
    if not scored_candidates:
        return scored_candidates

    # Extracting the relevance scores from the scored candidates to find the min and max
    scores = [score for _, score in scored_candidates]
    min_score = min(scores)
    max_score = max(scores)

    # If all scores are the same (max equals min), return a list of candidates with normalized score 0.0
    if max_score == min_score:
        return [(item_id, 0.0) for item_id, _ in scored_candidates]

    # Normalizing each score to the [0, 1] range using min-max normalization formula
    # and returning the list of tuples with item_id and normalized score
    denom = max_score - min_score
    return [(item_id, (score - min_score) / denom) for item_id, score in scored_candidates]


def item_similarity_by_genre(
    general_vars: GeneralVariables,
    item_a: int,
    item_b: int,
) -> float:
    """
    Calculates the cosine similarity between the genre vectors of two items.
    If either item does not have a genre vector, it returns 0.0 similarity.
    """

    # Retrieving the genre vectors for the two items from the general variables
    vec_a = general_vars.item_genre_vectors.get(item_a)
    vec_b = general_vars.item_genre_vectors.get(item_b)

    # If either item does not have a genre vector, we consider their similarity to be 0.0
    if vec_a is None or vec_b is None:
        return 0.0

    # Calculating and returning the cosine similarity between the two genre vectors
    return cosine_similarity(vec_a, vec_b)


def mmr_rerank(
    config: Config,
    general_vars: GeneralVariables,
    scored_candidates: list[tuple[int, float]],
    alpha: float,
) -> list[int]:
    """
    Applies Maximal Marginal Relevance (MMR) re-ranking to the scored candidates for a user.
    It iteratively selects items that balance relevance (based on the provided scores) and
    diversity (based on genre similarity) until it fills the top-K recommendations.
    """

    # Initializing the list of selected items and a set to keep
    # track of selected item ids for quick lookup
    selected = []
    selected_ids = set()

    # Iteratively selecting items until we have enough recommendations
    # or run out of candidates
    target_k = config.TOP_K
    while len(selected) < target_k and len(selected) < len(scored_candidates):
        best_item = None
        best_mmr_score = -float("inf")

        # Evaluating each candidate item to find the one with the highest MMR score
        for item_id, rel_score in scored_candidates:
            # Skipping items that have already been selected in the MMR process
            if item_id in selected_ids:
                continue

            # If no items have been selected yet, the MMR score is just the relevance score
            if len(selected) == 0:
                mmr_score = (1 - alpha) * rel_score
            else:
                # Calculating the maximum similarity of the candidate item to
                # any of the already selected items
                max_sim = max(
                    item_similarity_by_genre(general_vars, item_id, selected_item)
                    for selected_item in selected
                )
                mmr_score = (1 - alpha) * rel_score - alpha * max_sim

            # Updating the best item and best MMR score if this candidate has a
            # higher MMR score than the current best
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_item = item_id

        # If we couldn't find a best item (which can happen if all candidates
        # have been selected), break the loop
        if best_item is None:
            break

        # Adding the best item to the selected list and marking it as selected in the set
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
    """
    Prints an example of MMR re-ranking for a given user and method.
    It retrieves the top-M candidates for the user and applies MMR re-ranking.
    """

    # Getting the top-M candidates for the user
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
    """
    Evaluates the MMR re-ranking for a given method by generating recommendations
    for each eligible user, calculating the evaluation metrics (recall@10, ndcg@10,
    diversity@10), and returning the results in a ResultBundle.
    """

    print("\n-- Starting MMR evaluation --")
    print(f"Method: {method_vars.method_name}")
    print(f"Alpha values: {config.MMR_ALPHA_VALUES}")

    # Initializing a list to store results for each user and alpha value combination
    mmr_results = []

    # Iterating over each alpha value for MMR re-ranking and evaluating the
    # recommendations for each eligible user
    for alpha in config.MMR_ALPHA_VALUES:
        print(f"MMR eval for alpha={alpha}...")
        for user_id in general_vars.eligible_users:
            # Getting the top-M candidates for the user using the specified prediction function
            top_m_candidates = get_top_m_candidates(
                config, general_vars, method_vars, user_id, predict_func, popularity_vars
            )

            # Applying MMR re-ranking to the top-M candidates to get the final recommended items
            recommended = mmr_rerank(config, general_vars, top_m_candidates, alpha=alpha)

            # Calculating the evaluation metrics for the recommended items
            relevant = general_vars.relevant_items_by_user[user_id]
            recall = recall_at_k(recommended, relevant, k=config.TOP_K)
            ndcg = ndcg_at_k(recommended, relevant, k=config.TOP_K)
            diversity = diversity_at_k(recommended, general_vars.item_genre_vectors, k=config.TOP_K)

            # Storing the results for this user and alpha value combination in the mmr_results list
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

    # Creating a ResultBundle from the MMR evaluation results and returning it
    result_bundle = ResultBundle.from_rows(mmr_results)

    if config.PRINT_CONFIRM:
        print("\nMMR evaluation completed.")
        print("Average metrics across users and alpha values:")
        print(result_bundle.df[["recall@10", "ndcg@10", "diversity@10"]].mean())
        if config.ADVANCED_PRINT_MODE:
            print("DF head of MMR results:")
            print(result_bundle.df.head())

    return result_bundle

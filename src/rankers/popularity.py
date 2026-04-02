from core.config import Config
from core.structs import GeneralVariables, PopularityVariables, ResultBundle

from core.candidates import get_candidates

from core.metrics import recall_at_k, ndcg_at_k, diversity_at_k


def get_item_popularity(general_vars: GeneralVariables) -> dict:
    '''
    Calculates the popularity of each item based on the training data, which is
    defined as the number of interactions (ratings) an item has in the training set.
    Returns a dictionary mapping item_id to its popularity count.
    '''
    
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
    '''
    Generates recommendations for a given user based on item popularity.
    It retrieves the candidate items that the user has not interacted with,
    calculates their popularity scores, and returns the top-K most popular
    items as recommendations.
    '''
    
    # Getting candidate items that the user has not interacted with
    candidates = get_candidates(general_vars, user_id)
    
    # Sorting candidate items by their popularity scores in descending order
    ranked_items = sorted(
        candidates, key=lambda item_id: popularity_score(popularity_vars, item_id), reverse=True
    )
    
    # Returning the top-K item ids as recommendations
    return ranked_items[: config.TOP_K]


def evaluate_popularity(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
) -> ResultBundle:
    '''
    Evaluates the popularity-based recommender by generating recommendations
    for each eligible user, calculating the evaluation metrics (recall@10,
    ndcg@10, diversity@10), and returning the results in a ResultBundle.
    '''
    
    # Initializing a list to store results for each user
    results = [{} for _ in range(len(general_vars.eligible_users))]

    # Generating recommendations and calculate metrics for each eligible user
    for user_id in general_vars.eligible_users:
        # Generating recommendations for the user using the popularity-based recommender
        recommended = recommend_popularity(config, general_vars, popularity_vars, user_id)
        relevant = general_vars.relevant_items_by_user[user_id]

        # Calculating evaluation metrics for the recommendations
        recall = recall_at_k(recommended, relevant, k=config.TOP_K)
        ndcg = ndcg_at_k(recommended, relevant, k=config.TOP_K)
        diversity = diversity_at_k(recommended, general_vars.item_genre_vectors, k=config.TOP_K)

        # Storing the results for the user in the results list
        results[general_vars.eligible_users.index(user_id)] = {
            "user_id": user_id,
            "method": "popularity",
            "top_k": recommended,
            "recall@10": recall,
            "ndcg@10": ndcg,
            "diversity@10": diversity,
        }

    # Converting the list of results into a ResultBundle for easier analysis and logging
    result_bundle = ResultBundle.from_rows(results)
    if config.PRINT_CONFIRM:
        print("\nEvaluation done for popularity-based recommender.")
        print("Average metrics across users:")
        print(result_bundle.df[["recall@10", "ndcg@10", "diversity@10"]].mean())
        if config.ADVANCED_PRINT_MODE:
            print("Detailed results for each user:")
            print(f"results head: {result_bundle.df.head()}")

    return result_bundle


def print_examples_popularity(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    user_id: int,
):
    '''
    Prints the top-K recommended item ids and their corresponding titles
    for a given user using the popularity-based recommender.
    '''
    
    # Generating recommendations for the user using the popularity-based recommender
    example_recs = recommend_popularity(config, general_vars, popularity_vars, user_id)

    print("\nExample recommendations for user_id", user_id, "using popularity-based recommender:")
    print("User:", user_id)
    print("Top-10 item ids:", example_recs)

    print(
        general_vars.items[general_vars.items["item_id"].isin(example_recs)][["item_id", "title"]]
    )

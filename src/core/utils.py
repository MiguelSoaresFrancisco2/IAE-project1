import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from core.config import Config
from core.metrics import diversity_at_k, ndcg_at_k, recall_at_k
from core.structs import (
    GeneralVariables,
    LTR_Variables,
    MF_Variables,
    PopularityVariables,
    ResultBundle,
)

from rankers.mf_general import (
    prepare_mf_data,
    get_data,
    print_rmse,
    recommend_mf,
)

from rankers.mf_als import prepare_mf_als_data, train_mf_als
from rankers.mf_sgd import train_mf_sgd
from rankers.pairwise_ltr import (
    recommend_pairwise_ltr,
    train_pairwise_ltr,
)


def prepare_evaluation(
    config: Config,
    general_vars: GeneralVariables,
) -> tuple[list, set, dict, dict]:
    '''
    Prepares variables needed for evaluation:
    - all_users: list of all user ids
    - all_items: set of all item ids
    - train_items_by_user: dict mapping user_id to set of item_ids in their training
    - relevant_items_by_user: dict mapping user_id to set of relevant item_ids in test (rating >= 4)
    '''
    
    # Filtered test dataframe containing only relevant items (rating >= 4)
    test_relevant = general_vars.test_df[general_vars.test_df["rating"] >= 4].copy()

    # Dictionaries for train items and relevant test items by user
    train_items_by_user = general_vars.train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    relevant_items_by_user = test_relevant.groupby("user_id")["item_id"].apply(set).to_dict()

    # Sets of all users and items
    all_items = set(general_vars.items["item_id"].unique())
    all_users = sorted(general_vars.ratings["user_id"].unique())

    if config.PRINT_CONFIRM:
        print("\nEvaluation variables prepared.")
        print("Number of all users:", len(all_users))
        print("Number of all items:", len(all_items))
        print("Number of users with relevant items in test:", len(relevant_items_by_user))

    return all_users, all_items, train_items_by_user, relevant_items_by_user


def plot_training_history(
    config: Config,
    history: list,
    title: str,
    xlabel: str,
    ylabel: str,
    img_name: str,
    figsize: tuple = (8, 4),
    marker: str = "o",
):
    '''
    Plots the training history (RMSE for MF methods, loss for LTR) with appropriate titles and labels.
    Saves the plot as an image if SAVE_IMAGES is True and shows it if SHOW_PLOTS is True.
    
    Parameters:
    - history: list of metric values over epochs/iterations
    - title: title of the plot
    - xlabel: label for the x-axis
    - ylabel: label for the y-axis
    - img_name: filename for saving the image (if SAVE_IMAGES is True)
    - figsize: size of the plot figure
    - marker: marker style for the plot points (default is "o")
    '''
    
    plt.figure(figsize=figsize)
    if marker:
        plt.plot(range(1, len(history) + 1), history, marker=marker)
    else:
        plt.plot(range(1, len(history) + 1), history)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if config.SAVE_IMAGES:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{img_name}")
    if config.SHOW_PLOTS:
        plt.show()


def plot_mmr_tradeoff(
    config: Config,
    mmr_summary_df: pd.DataFrame,
    method_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(mmr_summary_df["diversity@10"], mmr_summary_df["ndcg@10"], marker="o")

    for _, row in mmr_summary_df.iterrows():
        plt.annotate(
            f"α={row['alpha']}",
            (row["diversity@10"], row["ndcg@10"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Diversity@10")
    plt.ylabel("NDCG@10")
    plt.title(f"MMR Tradeoff on {method_name.upper()}: NDCG@10 vs Diversity@10")
    plt.grid(True)
    if config.SAVE_IMAGES:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/mmr_tradeoff_{method_name}.png")
    if config.SHOW_PLOTS:
        plt.show()


def compare_methods(
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    methods_vars: dict[str, MF_Variables | LTR_Variables],
) -> None:
    '''
    Compares the evaluation results of all methods by printing their average
    recall@10, ndcg@10, and diversity@10.
    Also prints example recommendations for a specific user for each method.
    '''
    
    # Collecting the results DataFrames for all methods (popularity and MF/LTR) to compare their average metrics
    results_dfs = [popularity_vars.results.df] + [
        methods_vars[m].results.df for m in general_vars.done_methods_names
    ]
    method_names = ["popularity"] + [m for m in general_vars.done_methods_names]

    # Printing the average evaluation metrics across all eligible users for each method being compared
    print("\nComparison of Methods:")
    for i in range(len(method_names)):
        print(f"method: {method_names[i]}")
        print(f"recall@10: {results_dfs[i]['recall@10'].mean()}")
        print(f"ndcg@10: {results_dfs[i]['ndcg@10'].mean()}")
        print(f"diversity@10: {results_dfs[i]['diversity@10'].mean()}")
        print("-" * 40)


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Loads the MovieLens 100k dataset ratings and items dataframes.
    '''
    
    # Defining column names for the datasets
    _ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    _item_cols = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    # Load the ratings and items data
    ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=_ratings_cols, encoding="latin-1")
    items = pd.read_csv("data/ml-100k/u.item", sep="|", names=_item_cols, encoding="latin-1")

    if config.PRINT_CONFIRM:
        print("\n\nData loaded successfully.")
        print("Ratings shape:", ratings.shape)
        print("Items shape:", items.shape)
        if config.ADVANCED_PRINT_MODE:
            print("Ratings head:")
            print(ratings.head())
            print("Items head:")
            print(items.head())

    return ratings, items


def get_train_test_split(
    config: Config, general_vars: GeneralVariables
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Splits the ratings dataframe into train and test sets using sklearn's train_test_split.
    '''
    
    train_df, test_df = train_test_split(
        general_vars.ratings, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    if config.PRINT_CONFIRM:
        print("\nTrain/Test split completed.")
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)

    return train_df, test_df


def get_genre_vectors(general_vars: GeneralVariables) -> dict[int, list[float]]:
    '''
    Prepares a dictionary mapping item_id to its genre vector
    (list of 0s and 1s for each genre).
    '''
    
    # Defining the genre columns in the items dataframe
    genre_columns = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    # Extracting the genre columns and converting to a dictionary of item_id to genre vector
    item_genres = general_vars.items.set_index("item_id")[genre_columns].astype(float)

    # Converting to dictionary of item_id to genre vector (list of floats)
    item_genre_vectors = {item_id: item_genres.loc[item_id].values for item_id in item_genres.index}

    return item_genre_vectors


def get_eligible_users(config: Config, general_vars: GeneralVariables) -> list:
    '''
    Prepares a list of eligible users for evaluation, which are users that
    have at least one relevant item in the test set (rating >= 4).
    '''
    
    # Eligible users are those that have at least one relevant item in the test set (rating >= 4)
    eligible_users = [u for u in general_vars.all_users if u in general_vars.relevant_items_by_user]

    # Limiting the number of eligible users for faster evaluation
    if config.FAST_MODE and config.MAX_USERS_EVAL is not None:
        eligible_users = eligible_users[: config.MAX_USERS_EVAL]

    if config.PRINT_CONFIRM:
        print("\nEligible users for evaluation determined.")
        print(f"Eligible users for evaluation: {len(eligible_users)}")

    return eligible_users


def setup_general_vars(config: Config, general_vars: GeneralVariables) -> GeneralVariables:
    """
    Loads data, prepares train/test splits, computes relevant items,
    prepares index mapping and training data for MF methods.
    """

    # Load data and prepare train/test splits
    general_vars.ratings, general_vars.items = load_data(config)
    general_vars.train_df, general_vars.test_df = get_train_test_split(config, general_vars)

    # Prepare evaluation variables
    (
        general_vars.all_users,
        general_vars.all_items,
        general_vars.train_items_by_user,
        general_vars.relevant_items_by_user,
    ) = prepare_evaluation(config, general_vars)

    # Prepare other precomputed variables
    general_vars.item_genre_vectors = get_genre_vectors(general_vars)
    general_vars.eligible_users = get_eligible_users(config, general_vars)
    general_vars.index_map = prepare_mf_data(config, general_vars)

    # Prepare training data for MF methods
    general_vars.train_data, general_vars.test_data = get_data(config, general_vars)
    general_vars.user_ratings_train, general_vars.item_ratings_train = prepare_mf_als_data(
        config, general_vars
    )

    return general_vars


def save_logs(
    config: Config,
    general_vars: GeneralVariables,
    log_name: str,
    results: list[dict],
    hyperparameters: dict,
    ema_logs: bool = False,
) -> None:
    '''
    Saves the evaluation logs in a JSONL file in the logs directory. Each line
    in the file corresponds to a JSON object with the user_id, method, top_k
    recommendations, evaluation metrics, and hyperparameters. If ema_logs is
    True, it saves logs in a different format suitable for EMA sessions.
    '''

    # Helper function to convert user_id to internal index if index mapping is available
    def to_internal_user_id(user_id: int) -> int:
        '''
        Converts the original user_id to the internal index used for MF methods if
        index mapping is available. Otherwise, returns the original user_id as an integer.
        '''
        
        # If the index mapping is not available, return the original user_id as an integer
        if general_vars is None or general_vars.index_map is None:
            return int(user_id)
        
        # If the user_id is in the index mapping, return the corresponding internal index
        if user_id in general_vars.index_map.user_to_index:
            return int(general_vars.index_map.user_to_index[user_id])
    
        # If the user_id is not in the index mapping, return the original user_id as an integer
        return int(user_id)

    # Saving logs in a JSONL file, where each line is a JSON object with the user_id,
    # method, top_k recommendations, evaluation metrics, and hyperparameters
    with open(f"logs/{log_name}.jsonl", "w", encoding="utf-8") as f:
        # Non-EMA logs (popularity, MF, LTR)
        if not ema_logs:
            for row in results:
                log_entry = {
                    "user_id": to_internal_user_id(row["user_id"]),
                    "method": row["method"],
                    "top_k": [int(x) for x in row["top_k"]],
                    "metrics": {
                        "recall@10": None if row["recall@10"] is None else float(row["recall@10"]),
                        "ndcg@10": None if row["ndcg@10"] is None else float(row["ndcg@10"]),
                        "diversity@10": float(row["diversity@10"]),
                    },
                    "hyperparameters": hyperparameters,
                }
                if "alpha" in row:
                    log_entry["alpha"] = float(row["alpha"])
                f.write(json.dumps(log_entry) + "\n")
        # EMA logs
        else:
            for session in results:
                method_name = session.get("mf_method") or "ema"
                for log in session["logs"]:
                    log_entry = {
                        "user_id": to_internal_user_id(session["user_id"]),
                        "method": f"ema_{method_name}",
                        "round": int(log["round"]),
                        "recommended_items": [int(x) for x in log["recommended_items"]],
                        "chosen_item": int(log["chosen_item"]),
                        "chosen_title": log["chosen_title"],
                        "state_vector_head": [float(x) for x in log["state_vector_head"]],
                        "hyperparameters": hyperparameters,
                    }
                    f.write(json.dumps(log_entry) + "\n")

    if config.PRINT_CONFIRM:
        print("\nLogs saved successfully.")
        print(f"File created: logs/{log_name}.jsonl")
        if config.ADVANCED_PRINT_MODE:
            print("First 3 lines of the file:")
            with open(f"logs/{log_name}.jsonl", "r", encoding="utf-8") as f:
                for _ in range(3):
                    print(f.readline().strip())


def print_examples_recommendations(
    config: Config,
    general_vars: GeneralVariables,
    model_vars: MF_Variables | LTR_Variables,
    recommend_func: callable,
    user_id: int,
    popularity_vars: PopularityVariables | None = None,
) -> None:
    '''
    Prints example recommendations for a specific user using the given
    recommendation function and model variables.
    '''

    # Generating recommendations for the user using the appropriate recommendation function
    if popularity_vars is None:
        example_recs_mf = recommend_func(config, general_vars, model_vars, user_id)
    else:
        example_recs_mf = recommend_func(config, general_vars, popularity_vars, model_vars, user_id)

    # Printing the example recommendations for the user, including the
    # recommended item ids and their corresponding titles
    print(f"\nExample recommendations for user_id {user_id} using {model_vars.method_name.upper()} recommender:")
    print("User:", user_id)
    print(f"Top-10 {model_vars.method_name.upper()} item ids:", example_recs_mf)
    print(
        general_vars.items[general_vars.items["item_id"].isin(example_recs_mf)][
            ["item_id", "title"]
        ]
    )


def evaluate_method(
    config: Config,
    general_vars: GeneralVariables,
    method_vars: MF_Variables | LTR_Variables,
    popularity_vars: PopularityVariables | None = None,
    example_user_id: int | None = None,
) -> ResultBundle:
    '''
    Evaluates a recommendation method (MF SGD/ALS or pairwise LTR) by generating
    recommendations for each eligible user, calculating evaluation metrics
    (recall@10, ndcg@10, diversity@10), and returning the results in a ResultBundle.
    '''
    
    # Determining the recommendation function to use based on the method type (MF or LTR)
    method_name = method_vars.method_name
    recommend_func: callable = (
        recommend_mf if method_name != "pairwise_ltr" else recommend_pairwise_ltr
    )

    # Evaluating the method for each eligible user and storing the results
    # in a list of dictionaries
    method_results = [{} for _ in range(len(general_vars.eligible_users))]
    for user_id in general_vars.eligible_users:
        # Generating recommendations for the user using the appropriate recommendation function
        if method_name != "pairwise_ltr":
            recommended = recommend_func(config, general_vars, method_vars, user_id)
        else:
            recommended = recommend_func(
                config, general_vars, popularity_vars, method_vars, user_id
            )

        # Getting the relevant items for the user from the general variables
        relevant = general_vars.relevant_items_by_user[user_id]

        # Calculating evaluation metrics for the recommendations and storing them in the results list
        recall = recall_at_k(recommended, relevant, k=config.TOP_K)
        ndcg = ndcg_at_k(recommended, relevant, k=config.TOP_K)
        diversity = diversity_at_k(recommended, general_vars.item_genre_vectors, k=config.TOP_K)
        method_results[general_vars.eligible_users.index(user_id)] = {
            "user_id": int(user_id),
            "method": method_vars.method_name,
            "top_k": recommended,
            "recall@10": recall,
            "ndcg@10": ndcg,
            "diversity@10": diversity,
        }

    # Converting the list of results into a ResultBundle for easier analysis and logging
    result_bundle = ResultBundle.from_rows(method_results)

    if config.PRINT_CONFIRM:
        print("\nEvaluation done.")
        print_eval_summary(
            config, general_vars, popularity_vars, method_vars, result_bundle, example_user_id
        )

    return result_bundle


def setup_hyperparameters(
    config: Config,
    mf_sgd_vars: MF_Variables,
    mf_als_vars: MF_Variables,
    pairwise_ltr_vars: LTR_Variables,
) -> None:
    '''
    Sets up the hyperparameters for all methods in their respective variable containers.
    '''
    
    mf_sgd_vars.hyperparameters = {
        "k": config.TOP_K,
        "d": config.MF_SGD_DIM,
        "lr": config.MF_SGD_LR,
        "reg": config.MF_SGD_REG,
        "epochs": config.MF_SGD_EPOCHS,
    }
    mf_als_vars.hyperparameters = {
        "k": config.TOP_K,
        "d": config.MF_ALS_DIM,
        "reg": config.MF_ALS_REG,
        "iters": config.MF_ALS_ITERS,
    }
    pairwise_ltr_vars.hyperparameters = {
        "k": config.TOP_K,
        "epochs": config.LTR_EPOCHS,
        "lr": config.LTR_LR,
        "reg": config.LTR_REG,
        "max_pairs_per_user": config.LTR_MAX_PAIRS_PER_USER,
        "use_popularity": config.LTR_USE_POPULARITY,
    }


def train_method(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    methods_vars: dict[str, MF_Variables | LTR_Variables],
    method_name: str,
) -> ResultBundle | None:
    '''
    Trains a recommendation method and returns the trained model.
    - For MF methods, it trains the model and returns the training history.
    - For pairwise LTR, it trains the model using the specified MF method's
      embeddings and returns the training history and learned parameters.
    '''
    
    # Training the specified method and return the model and training history
    if method_name == "mf_sgd":
        model = train_mf_sgd(config, general_vars)
    elif method_name == "mf_als":
        model = train_mf_als(config, general_vars)
    elif method_name == "pairwise_ltr":
        # Verify that the LTR configuration is valid before training
        if not is_verify_ltr_config_ok(config, general_vars):
            print("LTR configuration is invalid. Skipping LTR training.")
            return None

        # Setting the MF model to be used for LTR training based on the specified LTR_MF_METHOD
        methods_vars[method_name].mf_model = methods_vars[config.LTR_MF_METHOD].model

        # Training the pairwise LTR model using the specified MF model's
        # embeddings and returning the training history and learned parameters
        model = train_pairwise_ltr(
            config,
            general_vars,
            popularity_vars,
            methods_vars[method_name],
        )

    if config.SHOW_PLOTS or config.SAVE_IMAGES:
        show_plots(config, model, method_name)

    return model


def is_verify_ltr_config_ok(
    config: Config,
    general_vars: GeneralVariables,
) -> bool:
    '''
    Verifies that the configuration for pairwise LTR training is valid:
    - The specified LTR_MF_METHOD must be in the METHODS list.
    - The specified LTR_MF_METHOD must have been trained before LTR training.
    Returns True if the configuration is valid, False otherwise.
    '''
    
    if config.LTR_MF_METHOD not in config.METHODS:
        print(f"Error: LTR_MF_METHOD '{config.LTR_MF_METHOD}' must be in METHODS for pairwise LTR.")
        return False
    elif config.LTR_MF_METHOD not in general_vars.done_methods_names:
        print(f"Error: LTR_MF_METHOD '{config.LTR_MF_METHOD}' must be trained before pairwise LTR.")
        return False

    return True


def show_plots(
    config: Config,
    model: dict,
    method_name: str,
) -> None:
    '''
    Shows and/or saves the training history plot for the specified method.
     - For MF methods, it plots the training RMSE over epochs/iterations.
     - For pairwise LTR, it plots the training loss over epochs.
    '''

    # Extracting the training history from the model and plotting it with appropriate titles and labels
    history = model["history"]
    title = f"{method_name.upper()} Training {'RMSE' if method_name != 'pairwise_ltr' else 'Loss'}"
    xlabel = "Epoch" if method_name != "mf_als" else "Iteration"
    y_label = "Loss" if method_name == "pairwise_ltr" else "Train RMSE"
    img_name = f"{method_name}_training_{'rmse' if method_name != 'pairwise_ltr' else 'loss'}.png"

    plot_training_history(config, history, title, xlabel, y_label, img_name)


def print_eval_summary(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    method_vars: MF_Variables | LTR_Variables,
    result_bundle: ResultBundle,
    example_user_id: int | None = None,
) -> None:
    '''
    Prints a summary of the evaluation results for a given method, including:
    - Average recall@10, ndcg@10, and diversity@10 across all eligible users.
    - Example recommendations for a specific user (if example_user_id is provided).
    - RMSE for MF methods (if applicable).
    '''
    
    print("-- Evaluation Summary --")
    method_name = method_vars.method_name

    print(f"\nAverage recall@10: {result_bundle.df['recall@10'].mean():.6f}")
    print(f"Average ndcg@10: {result_bundle.df['ndcg@10'].mean():.6f}")
    print(f"Average diversity@10: {result_bundle.df['diversity@10'].mean():.6f}")
    
    if method_name != "pairwise_ltr":
        print_rmse(general_vars, method_vars.model)

    print_examples_recommendations(
        config,
        general_vars,
        method_vars,
        recommend_func=recommend_mf if method_name != "pairwise_ltr" else recommend_pairwise_ltr,
        user_id=example_user_id if example_user_id is not None else 1,
        popularity_vars=popularity_vars if method_name == "pairwise_ltr" else None,
    )
    
    if config.ADVANCED_PRINT_MODE:
        print("Full evaluation results (first 5 rows):")
        print(result_bundle.df.head())
    
    print("-- End of Summary --")

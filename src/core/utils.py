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
)

from rankers.mf_als import prepare_mf_als_data


def prepare_evaluation(
    config: Config,
    general_vars: GeneralVariables,
) -> tuple[list, set, dict, dict]:
    test_relevant = general_vars.test_df[general_vars.test_df["rating"] >= 4].copy()

    train_items_by_user = general_vars.train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    relevant_items_by_user = test_relevant.groupby("user_id")["item_id"].apply(set).to_dict()

    all_items = set(general_vars.items["item_id"].unique())
    all_users = sorted(general_vars.ratings["user_id"].unique())

    if config.PRINT_CONFIRM:
        print("Número total de users:", len(all_users))
        print("Número total de items:", len(all_items))
        print("Users com itens relevantes no teste:", len(relevant_items_by_user))

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


def compare_methods(results_dfs: list[pd.DataFrame], method_names: list[str]):
    for i in range(len(method_names)):
        print(f"method: {method_names[i]}")
        print(f"recall@10: {results_dfs[i]['recall@10'].mean()}")
        print(f"ndcg@10: {results_dfs[i]['ndcg@10'].mean()}")
        print(f"diversity@10: {results_dfs[i]['diversity@10'].mean()}")


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=_ratings_cols, encoding="latin-1")
    items = pd.read_csv("data/ml-100k/u.item", sep="|", names=_item_cols, encoding="latin-1")

    if config.PRINT_CONFIRM:
        print("Ratings shape:", ratings.shape)
        print("Items shape:", items.shape)
        print(ratings.head())

    return ratings, items


def get_train_test_split(
    config: Config, general_vars: GeneralVariables
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        general_vars.ratings, test_size=0.2, random_state=config.RANDOM_STATE
    )

    if config.PRINT_CONFIRM:
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)

    return train_df, test_df


def get_genre_vectors(general_vars: GeneralVariables) -> dict[int, list[float]]:
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

    item_genres = general_vars.items.set_index("item_id")[genre_columns].astype(float)

    item_genre_vectors = {item_id: item_genres.loc[item_id].values for item_id in item_genres.index}

    return item_genre_vectors


def get_eligible_users(config: Config, general_vars: GeneralVariables) -> list:
    eligible_users = [u for u in general_vars.all_users if u in general_vars.relevant_items_by_user]

    if config.FAST_MODE and config.MAX_USERS_EVAL is not None:
        eligible_users = eligible_users[: config.MAX_USERS_EVAL]

    if config.PRINT_CONFIRM:
        print(f"Eligible users for evaluation: {len(eligible_users)}")

    return eligible_users


def setup_general_vars(config: Config, general_vars: GeneralVariables) -> GeneralVariables:
    general_vars.ratings, general_vars.items = load_data(config)
    general_vars.train_df, general_vars.test_df = get_train_test_split(config, general_vars)
    (
        general_vars.all_users,
        general_vars.all_items,
        general_vars.train_items_by_user,
        general_vars.relevant_items_by_user,
    ) = prepare_evaluation(config, general_vars)
    general_vars.item_genre_vectors = get_genre_vectors(general_vars)
    general_vars.eligible_users = get_eligible_users(config, general_vars)
    general_vars.index_map = prepare_mf_data(config, general_vars)
    general_vars.train_data, general_vars.test_data = get_data(config, general_vars)
    general_vars.user_ratings_train, general_vars.item_ratings_train = prepare_mf_als_data(
        general_vars
    )

    return general_vars


def save_logs(
    config: Config,
    log_name: str,
    results: list[dict],
    hyperparameters: dict,
    ema_logs: bool = False,
) -> None:
    os.makedirs("logs", exist_ok=True)

    with open(f"logs/{log_name}.jsonl", "w", encoding="utf-8") as f:
        if not ema_logs:
            for row in results:
                log_entry = {
                    "user_id": int(row["user_id"]),
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
        else:
            for session in results:
                method_name = session.get("mf_method") or "ema"
                for log in session["logs"]:
                    log_entry = {
                        "user_id": int(session["user_id"]),
                        "method": f"ema_{method_name}",
                        "round": int(log["round"]),
                        "recommended_items": [int(x) for x in log["recommended_items"]],
                        "chosen_item": int(log["chosen_item"]),
                        "chosen_title": log["chosen_title"],
                        "state_vector_head": [float(x) for x in log["state_vector_head"]],
                        "hyperparameters": hyperparameters,
                    }
                    f.write(json.dumps(log_entry) + "\n")

    print(f"File created: logs/{log_name}.jsonl")

    if config.PRINT_CONFIRM:
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

    if popularity_vars is None:
        example_recs_mf = recommend_func(config, general_vars, model_vars, user_id)
    else:
        example_recs_mf = recommend_func(config, general_vars, popularity_vars, model_vars, user_id)

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
    recommend_func: callable,
    popularity_vars: PopularityVariables | None = None,
) -> ResultBundle:
    method_results = [{} for _ in range(len(general_vars.eligible_users))]

    for user_id in general_vars.eligible_users:
        if popularity_vars is None:
            recommended = recommend_func(config, general_vars, method_vars, user_id)
        else:
            recommended = recommend_func(
                config, general_vars, popularity_vars, method_vars, user_id
            )

        relevant = general_vars.relevant_items_by_user[user_id]

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

    result_bundle = ResultBundle.from_rows(method_results)

    if config.PRINT_CONFIRM:
        print(result_bundle.df.head())

    return result_bundle


def setup_hyperparameters(
    config: Config,
    mf_sgd_vars: MF_Variables,
    mf_als_vars: MF_Variables,
    pairwise_ltr_vars: LTR_Variables,
) -> None:
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

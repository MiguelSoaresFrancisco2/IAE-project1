import pandas as pd
import numpy as np

from core.config import Config
from core.structs import GeneralVariables, MF_SGDVariables

from core.utils import get_candidates

from core.metrics import recall_at_k, ndcg_at_k, diversity_at_k


def prepare_md_sgd_data(config: Config, general_vars: GeneralVariables) -> tuple:
    unique_user_ids = sorted(general_vars.ratings["user_id"].unique())
    unique_item_ids = sorted(general_vars.ratings["item_id"].unique())

    user_to_index = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}

    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

    n_users = len(unique_user_ids)
    n_items = len(unique_item_ids)

    if config.PRINT_CONFIRM:
        print("n_users =", n_users)
        print("n_items =", n_items)

    return (
        unique_user_ids,
        unique_item_ids,
        user_to_index,
        item_to_index,
        index_to_user,
        index_to_item,
        n_users,
        n_items,
    )


def get_md_sgd_data(
    config: Config,
    general_vars: GeneralVariables,
):
    train_data = [
        (
            general_vars.user_to_index[row.user_id],
            general_vars.item_to_index[row.item_id],
            float(row.rating),
        )
        for row in general_vars.train_df.itertuples(index=False)
    ]

    test_data = [
        (
            general_vars.user_to_index[row.user_id],
            general_vars.item_to_index[row.item_id],
            float(row.rating),
        )
        for row in general_vars.test_df.itertuples(index=False)
    ]

    if config.PRINT_CONFIRM:
        print("Train tuples:", len(train_data))
        print("Test tuples:", len(test_data))
        print("Example:", train_data[:5])

    return train_data, test_data


def train_mf_sgd(
    config: Config,
    general_vars: GeneralVariables,
):
    rng = np.random.default_rng(config.RANDOM_STATE)

    mu = np.mean([rating for _, _, rating in general_vars.train_data])

    P = rng.normal(0, 0.1, size=(general_vars.n_users, config.MF_DIM))
    Q = rng.normal(0, 0.1, size=(general_vars.n_items, config.MF_DIM))

    bu = np.zeros(general_vars.n_users)
    bi = np.zeros(general_vars.n_items)

    history = []
    train_data_shuffled = general_vars.train_data.copy()

    for epoch in range(config.SGD_EPOCHS):
        rng.shuffle(train_data_shuffled)

        total_loss = 0.0

        for u_idx, i_idx, rating in train_data_shuffled:
            pred = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            err = rating - pred

            total_loss += err**2

            pu_old = P[u_idx].copy()
            qi_old = Q[i_idx].copy()

            P[u_idx] += config.LR_MF_SGD * (err * qi_old - config.REG_MF_SGD * pu_old)
            Q[i_idx] += config.LR_MF_SGD * (err * pu_old - config.REG_MF_SGD * qi_old)
            bu[u_idx] += config.LR_MF_SGD * (err - config.REG_MF_SGD * bu[u_idx])
            bi[i_idx] += config.LR_MF_SGD * (err - config.REG_MF_SGD * bi[i_idx])

        rmse = np.sqrt(total_loss / len(train_data_shuffled))
        history.append(rmse)

        if config.PRINT_CONFIRM:
            print(f"Epoch {epoch + 1}/{config.SGD_EPOCHS} - Train RMSE: {rmse:.4f}")

    return {
        "mu": mu,
        "P": P,
        "Q": Q,
        "bu": bu,
        "bi": bi,
        "history": history,
        "params": {
            "d": config.MF_DIM,
            "lr": config.LR_MF_SGD,
            "reg": config.REG_MF_SGD,
            "epochs": config.SGD_EPOCHS,
        },
    }


def predict_mf_sgd(
    general_vars: GeneralVariables,
    model: dict,
    user_id: int,
    item_id: int,
) -> float:
    u_idx = general_vars.user_to_index[user_id]
    i_idx = general_vars.item_to_index[item_id]

    pred = (
        model["mu"]
        + model["bu"][u_idx]
        + model["bi"][i_idx]
        + np.dot(model["P"][u_idx], model["Q"][i_idx])
    )

    pred = max(1.0, min(5.0, pred))
    return float(pred)


def recommend_mf_sgd(
    config: Config,
    general_vars: GeneralVariables,
    mf_sgd_vars: MF_SGDVariables,
    user_id: int,
) -> list:
    candidates = get_candidates(general_vars, user_id)

    scored_items = [None] * len(candidates)
    for i, item_id in enumerate(candidates):
        score = predict_mf_sgd(
            general_vars,
            mf_sgd_vars.model,
            user_id,
            item_id,
        )
        scored_items[i] = (item_id, score)

    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return [int(item_id) for item_id, _ in ranked_items[: config.TOP_K]]


def evaluate_mf_sgd(
    config: Config, general_vars: GeneralVariables, mf_sgd_vars: MF_SGDVariables
) -> tuple[list, pd.DataFrame]:
    mf_sgd_results = [{} for _ in range(len(general_vars.eligible_users))]

    for user_id in general_vars.eligible_users:
        recommended = recommend_mf_sgd(config, general_vars, mf_sgd_vars, user_id)
        relevant = general_vars.relevant_items_by_user[user_id]

        recall = recall_at_k(recommended, relevant, k=config.TOP_K)
        ndcg = ndcg_at_k(recommended, relevant, k=config.TOP_K)
        diversity = diversity_at_k(recommended, general_vars.item_genre_vectors, k=config.TOP_K)

        mf_sgd_results[general_vars.eligible_users.index(user_id)] = {
            "user_id": int(user_id),
            "method": "mf_sgd",
            "top_k": recommended,
            "recall@10": recall,
            "ndcg@10": ndcg,
            "diversity@10": diversity,
        }

    mf_sgd_results_df = pd.DataFrame(mf_sgd_results)

    if config.PRINT_CONFIRM:
        print(mf_sgd_results_df.head())

    return mf_sgd_results, mf_sgd_results_df


def print_examples_mf_sgd(
    config: Config,
    general_vars: GeneralVariables,
    mf_sgd_vars: MF_SGDVariables,
    user_id: int,
):
    example_recs_mf = recommend_mf_sgd(config, general_vars, mf_sgd_vars, user_id)

    print("User:", user_id)
    print("Top-10 MF-SGD item ids:", example_recs_mf)

    general_vars.items[general_vars.items["item_id"].isin(example_recs_mf)][["item_id", "title"]]

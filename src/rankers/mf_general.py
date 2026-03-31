import numpy as np

from core.config import Config
from core.structs import GeneralVariables, MF_Variables
from core.utils import get_candidates


def prepare_md_data(config: Config, general_vars: GeneralVariables) -> tuple:
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


def get_md_data(
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


def predict_mf(
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


def recommend_mf(
    config: Config,
    general_vars: GeneralVariables,
    mf_vars: MF_Variables,
    user_id: int,
) -> list:
    candidates = get_candidates(general_vars, user_id)

    scored_items = [None] * len(candidates)
    for i, item_id in enumerate(candidates):
        score = predict_mf(
            general_vars,
            mf_vars.model,
            user_id,
            item_id,
        )
        scored_items[i] = (item_id, score)

    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return [int(item_id) for item_id, _ in ranked_items[: config.TOP_K]]


def compute_rmse(
    general_vars: GeneralVariables,
    data: list[tuple[int, int, float]],
    model: dict,
) -> float:
    errors = [0.0] * len(data)
    i: int = 0
    for u_idx, i_idx, rating in data:
        user_id = general_vars.index_to_user[u_idx]
        item_id = general_vars.index_to_item[i_idx]

        pred = predict_mf(general_vars, model, user_id, item_id)

        errors[i] = (rating - pred) ** 2
        i += 1
    return float(np.sqrt(np.mean(errors)))


def print_rmse(
    general_vars: GeneralVariables,
    model: dict,
):
    test_rmse = compute_rmse(general_vars, general_vars.test_data, model)

    train_rmse = compute_rmse(general_vars, general_vars.train_data, model)

    print("Test RMSE:", test_rmse)
    print("Train RMSE:", train_rmse)

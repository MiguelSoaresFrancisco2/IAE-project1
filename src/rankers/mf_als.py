import numpy as np

from collections import defaultdict

from core.config import Config
from core.structs import GeneralVariables


def prepare_mf_als_data(general_vars: GeneralVariables) -> tuple[defaultdict, defaultdict]:
    user_ratings_train = defaultdict(list)
    item_ratings_train = defaultdict(list)

    for u_idx, i_idx, rating in general_vars.train_data:
        user_ratings_train[u_idx].append((i_idx, rating))
        item_ratings_train[i_idx].append((u_idx, rating))

    print("Example user ratings:", list(user_ratings_train.items())[:1])
    print("Example item ratings:", list(item_ratings_train.items())[:1])

    return user_ratings_train, item_ratings_train


def train_mf_als(
    config: Config,
    general_vars: GeneralVariables,
) -> dict:
    rng = np.random.default_rng(config.RANDOM_STATE)

    mu = np.mean([rating for _, _, rating in general_vars.train_data])

    P = rng.normal(0, 0.1, size=(general_vars.index_map.n_users, config.MF_ALS_DIM))
    Q = rng.normal(0, 0.1, size=(general_vars.index_map.n_items, config.MF_ALS_DIM))

    bu = np.zeros(general_vars.index_map.n_users)
    bi = np.zeros(general_vars.index_map.n_items)

    history = []

    I_d = np.eye(config.MF_ALS_DIM)

    for iteration in range(config.MF_ALS_ITERS):
        # Update user factors P
        for u_idx in range(general_vars.index_map.n_users):
            ratings_u = general_vars.user_ratings_train.get(u_idx, [])
            if not ratings_u:
                continue

            A = config.MF_ALS_REG * I_d
            b_vec = np.zeros(config.MF_ALS_DIM)

            for i_idx, rating in ratings_u:
                qi = Q[i_idx]
                residual = rating - (mu + bu[u_idx] + bi[i_idx])
                A += np.outer(qi, qi)
                b_vec += residual * qi

            P[u_idx] = np.linalg.solve(A, b_vec)

        # Update item factors Q
        for i_idx in range(general_vars.index_map.n_items):
            ratings_i = general_vars.item_ratings_train.get(i_idx, [])
            if not ratings_i:
                continue

            A = config.MF_ALS_REG * I_d
            b_vec = np.zeros(config.MF_ALS_DIM)

            for u_idx, rating in ratings_i:
                pu = P[u_idx]
                residual = rating - (mu + bu[u_idx] + bi[i_idx])
                A += np.outer(pu, pu)
                b_vec += residual * pu

            Q[i_idx] = np.linalg.solve(A, b_vec)

        # Update user biases
        for u_idx in range(general_vars.index_map.n_users):
            ratings_u = general_vars.user_ratings_train.get(u_idx, [])
            if not ratings_u:
                continue

            numerator = 0.0
            denominator = config.MF_ALS_REG

            for i_idx, rating in ratings_u:
                pred_wo_bu = mu + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
                numerator += rating - pred_wo_bu
                denominator += 1.0

            bu[u_idx] = numerator / denominator

        # Update item biases
        for i_idx in range(general_vars.index_map.n_items):
            ratings_i = general_vars.item_ratings_train.get(i_idx, [])
            if not ratings_i:
                continue

            numerator = 0.0
            denominator = config.MF_ALS_REG

            for u_idx, rating in ratings_i:
                pred_wo_bi = mu + bu[u_idx] + np.dot(P[u_idx], Q[i_idx])
                numerator += rating - pred_wo_bi
                denominator += 1.0

            bi[i_idx] = numerator / denominator

        # Compute train RMSE
        squared_errors = []
        for u_idx, i_idx, rating in general_vars.train_data:
            pred = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            pred = max(1.0, min(5.0, pred))
            squared_errors.append((rating - pred) ** 2)

        rmse = float(np.sqrt(np.mean(squared_errors)))
        history.append(rmse)

        if config.PRINT_CONFIRM:
            print(f"ALS Iteration {iteration + 1}/{config.MF_ALS_ITERS} - Train RMSE: {rmse:.4f}")

    return {
        "mu": mu,
        "P": P,
        "Q": Q,
        "bu": bu,
        "bi": bi,
        "history": history,
        "params": {"d": config.MF_ALS_DIM, "reg": config.MF_ALS_REG, "iters": config.MF_ALS_ITERS},
    }

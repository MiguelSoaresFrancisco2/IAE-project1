import numpy as np

from core.config import Config
from core.structs import GeneralVariables


def train_mf_sgd(
    config: Config,
    general_vars: GeneralVariables,
):
    rng = np.random.default_rng(config.RANDOM_STATE)

    mu = np.mean([rating for _, _, rating in general_vars.train_data])

    P = rng.normal(0, 0.1, size=(general_vars.n_users, config.MF_SGD_DIM))
    Q = rng.normal(0, 0.1, size=(general_vars.n_items, config.MF_SGD_DIM))

    bu = np.zeros(general_vars.n_users)
    bi = np.zeros(general_vars.n_items)

    history = []
    train_data_shuffled = general_vars.train_data.copy()

    for epoch in range(config.MF_SGD_EPOCHS):
        rng.shuffle(train_data_shuffled)

        total_loss = 0.0

        for u_idx, i_idx, rating in train_data_shuffled:
            pred = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            err = rating - pred

            total_loss += err**2

            pu_old = P[u_idx].copy()
            qi_old = Q[i_idx].copy()

            P[u_idx] += config.MF_SGD_LR * (err * qi_old - config.MF_SGD_REG * pu_old)
            Q[i_idx] += config.MF_SGD_LR * (err * pu_old - config.MF_SGD_REG * qi_old)
            bu[u_idx] += config.MF_SGD_LR * (err - config.MF_SGD_REG * bu[u_idx])
            bi[i_idx] += config.MF_SGD_LR * (err - config.MF_SGD_REG * bi[i_idx])

        rmse = np.sqrt(total_loss / len(train_data_shuffled))
        history.append(rmse)

        if config.PRINT_CONFIRM:
            print(f"Epoch {epoch + 1}/{config.MF_SGD_EPOCHS} - Train RMSE: {rmse:.4f}")

    return {
        "mu": mu,
        "P": P,
        "Q": Q,
        "bu": bu,
        "bi": bi,
        "history": history,
        "params": {
            "d": config.MF_SGD_DIM,
            "lr": config.MF_SGD_LR,
            "reg": config.MF_SGD_REG,
            "epochs": config.MF_SGD_EPOCHS,
        },
    }

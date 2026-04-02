import numpy as np

from collections import defaultdict

from core.config import Config
from core.structs import GeneralVariables
from rankers.mf_general import init_mf_params


def prepare_mf_als_data(
    config: Config, general_vars: GeneralVariables
) -> tuple[defaultdict, defaultdict]:
    """
    Prepares the training data for ALS, which are dictionaries mapping
    user_index to a list of (item_index, rating) and item_index to a
    list of (user_index, rating). This format is suitable for
    efficiently updating user and item factors in ALS.
    """

    # Using defaultdict to automatically handle users/items with no ratings
    user_ratings_train = defaultdict(list)
    item_ratings_train = defaultdict(list)

    # Populating the dictionaries with the training data
    for u_idx, i_idx, rating in general_vars.train_data:
        user_ratings_train[u_idx].append((i_idx, rating))
        item_ratings_train[i_idx].append((u_idx, rating))

    if config.PRINT_CONFIRM:
        print("\nData prepared for MF ALS.")
        print("Number of users with ratings:", len(user_ratings_train))
        print("Number of items with ratings:", len(item_ratings_train))
        if config.ADVANCED_PRINT_MODE:
            print(
                "Example user_index to (item_index, rating):",
                {k: v[:5] for k, v in list(user_ratings_train.items())[:5]},
            )
            print(
                "Example item_index to (user_index, rating):",
                {k: v[:5] for k, v in list(item_ratings_train.items())[:5]},
            )

    return user_ratings_train, item_ratings_train


def train_mf_als(
    config: Config,
    general_vars: GeneralVariables,
) -> dict:
    '''
    Trains a matrix factorization model using Alternating Least Squares (ALS).
    The model learns user and item latent factors, as well as user and item biases,
    to predict user-item interactions. The training process iteratively updates the
    parameters to minimize the prediction error on the training data.
    Returns a dictionary containing the learned parameters and training history.
    '''
    
    # Initializing the random number generator for reproducibility
    rng = np.random.default_rng(config.RANDOM_STATE)

    # Initializing the MF parameters (global mean, user factors, item factors, user biases, item biases)
    mu, P, Q, bu, bi = init_mf_params(general_vars, config.MF_ALS_DIM, rng)

    # Initializing a list to store the training history (RMSE for each epoch)
    history = [None] * config.MF_ALS_ITERS

    # Preparing the training data in a format suitable for ALS updates
    I_d = np.eye(config.MF_ALS_DIM)

    # ALS training loop for the specified number of iterations
    for iteration in range(config.MF_ALS_ITERS):
        # Update user factors P
        for u_idx in range(general_vars.index_map.n_users):
            # Getting the items rated by this user and their ratings
            ratings_u = general_vars.user_ratings_train.get(u_idx, [])
            if not ratings_u:
                continue

            # Building the A matrix and b vector for the least
            # squares problem to update P[u_idx]
            A = config.MF_ALS_REG * I_d
            b_vec = np.zeros(config.MF_ALS_DIM)

            # Iterating over the items rated by this user to populate A and b_vec
            for i_idx, rating in ratings_u:
                qi = Q[i_idx]
                residual = rating - (mu + bu[u_idx] + bi[i_idx])
                A += np.outer(qi, qi)
                b_vec += residual * qi

            # Solving the linear system A * P[u_idx] = b_vec to update the user factors
            P[u_idx] = np.linalg.solve(A, b_vec)

        # Update item factors Q
        for i_idx in range(general_vars.index_map.n_items):
            # Getting the users who rated this item and their ratings
            ratings_i = general_vars.item_ratings_train.get(i_idx, [])
            if not ratings_i:
                continue
            
            # Building the A matrix and b vector for the least
            # squares problem to update Q[i_idx]
            A = config.MF_ALS_REG * I_d
            b_vec = np.zeros(config.MF_ALS_DIM)

            # Iterating over the users who rated this item to populate A and b_vec
            for u_idx, rating in ratings_i:
                pu = P[u_idx]
                residual = rating - (mu + bu[u_idx] + bi[i_idx])
                A += np.outer(pu, pu)
                b_vec += residual * pu

            # Solving the linear system A * Q[i_idx] = b_vec to update the item factors
            Q[i_idx] = np.linalg.solve(A, b_vec)

        # Update user biases
        for u_idx in range(general_vars.index_map.n_users):
            # Getting the items rated by this user and their ratings to update the user bias
            ratings_u = general_vars.user_ratings_train.get(u_idx, [])
            if not ratings_u:
                continue

            # Calculating the numerator and denominator for the user bias update based on the residuals
            numerator = 0.0
            denominator = config.MF_ALS_REG
            for i_idx, rating in ratings_u:
                pred_wo_bu = mu + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
                numerator += rating - pred_wo_bu
                denominator += 1.0

            # Updating the user bias by solving the regularized least squares problem
            bu[u_idx] = numerator / denominator

        # Update item biases
        for i_idx in range(general_vars.index_map.n_items):
            # Getting the users who rated this item and their ratings to update the item bias
            ratings_i = general_vars.item_ratings_train.get(i_idx, [])
            if not ratings_i:
                continue

            # Calculating the numerator and denominator for the item bias update based on the residuals
            numerator = 0.0
            denominator = config.MF_ALS_REG
            for u_idx, rating in ratings_i:
                pred_wo_bi = mu + bu[u_idx] + np.dot(P[u_idx], Q[i_idx])
                numerator += rating - pred_wo_bi
                denominator += 1.0

            # Updating the item bias by solving the regularized least squares problem
            bi[i_idx] = numerator / denominator

        # Compute train RMSE
        squared_errors = []
        for u_idx, i_idx, rating in general_vars.train_data:
            pred = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            pred = max(1.0, min(5.0, pred))
            squared_errors.append((rating - pred) ** 2)

        # Calculating the RMSE for this iteration and storing it in the history
        rmse = float(np.sqrt(np.mean(squared_errors)))
        history[iteration] = rmse

        print(f"ALS Iteration {iteration + 1}/{config.MF_ALS_ITERS} - Train RMSE: {rmse:.4f}")

    # Returning the learned parameters and training history in a dictionary
    return {
        "mu": mu,
        "P": P,
        "Q": Q,
        "bu": bu,
        "bi": bi,
        "history": history,
        "params": {
            "d": config.MF_ALS_DIM,
            "reg": config.MF_ALS_REG,
            "iters": config.MF_ALS_ITERS
        },
    }

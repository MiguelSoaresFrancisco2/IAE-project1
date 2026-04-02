import numpy as np

from core.config import Config
from core.structs import GeneralVariables
from rankers.mf_general import init_mf_params


def train_mf_sgd(
    config: Config,
    general_vars: GeneralVariables,
):
    '''
    Trains a matrix factorization model using stochastic gradient descent (SGD).
    The model learns user and item latent factors, as well as user and item biases,
    to predict user-item interactions. The training process iteratively updates the
    parameters to minimize the prediction error on the training data.
    Returns a dictionary containing the learned parameters and training history.
    '''
    
    # Initializing the random number generator for reproducibility
    rng = np.random.default_rng(config.RANDOM_STATE)

    # Initializing the MF parameters (global mean, user factors, item factors, user biases, item biases)
    mu, P, Q, bu, bi = init_mf_params(general_vars, config.MF_SGD_DIM, rng)

    # Initializing a list to store the training history (RMSE for each epoch)
    history = [None] * config.MF_SGD_EPOCHS

    train_data_shuffled = general_vars.train_data.copy()

    # Training loop for the specified number of epochs
    for epoch in range(config.MF_SGD_EPOCHS):
        # Shuffling the training data at the beginning of each epoch to ensure randomness in SGD updates
        rng.shuffle(train_data_shuffled)

        # Initializing the total loss for this epoch to compute RMSE
        total_loss = 0.0
        
        # Iterating over each training example (user_index, item_index, rating)
        for u_idx, i_idx, rating in train_data_shuffled:
            # Predicting the rating using the current model parameters
            pred = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            
            # Calculating the prediction error (residual) for this training example
            err = rating - pred

            # Accumulating the squared error to compute RMSE
            total_loss += err**2

            # Storing the current user and item latent factors before updating
            pu_old = P[u_idx].copy()
            qi_old = Q[i_idx].copy()

            # Updating the user and item latent factors and biases using the SGD update rules
            P[u_idx] += config.MF_SGD_LR * (err * qi_old - config.MF_SGD_REG * pu_old)
            Q[i_idx] += config.MF_SGD_LR * (err * pu_old - config.MF_SGD_REG * qi_old)
            bu[u_idx] += config.MF_SGD_LR * (err - config.MF_SGD_REG * bu[u_idx])
            bi[i_idx] += config.MF_SGD_LR * (err - config.MF_SGD_REG * bi[i_idx])

        # Calculating the RMSE for this epoch and storing it in the history
        rmse = np.sqrt(total_loss / len(train_data_shuffled))
        history[epoch] = rmse

        print(f"Epoch {epoch + 1}/{config.MF_SGD_EPOCHS} - Train RMSE: {rmse:.4f}")

    # Returning the learned parameters and training history in a dictionary
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

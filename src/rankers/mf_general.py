import numpy as np

from core.config import Config
from core.structs import GeneralVariables, IndexMap, MF_Variables
from core.candidates import get_candidates


def prepare_mf_data(config: Config, general_vars: GeneralVariables) -> IndexMap:
    """
    Prepares the index mapping for users and items, which includes:
    - user_to_index: mapping from user_id to a unique index
    - item_to_index: mapping from item_id to a unique index
    - index_to_user: mapping from index back to user_id
    - index_to_item: mapping from index back to item_id
    Essential for matrix factorization models to work with integer indices.
    """

    # Extracting unique user_ids and item_ids from the ratings dataframe
    user_ids = sorted(general_vars.ratings["user_id"].unique())
    item_ids = sorted(general_vars.ratings["item_id"].unique())

    # Creating mappings from user_id/item_id to index and vice versa
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    # Creating reverse mappings from index back to user_id/item_id
    index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
    index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

    # Creating the IndexMap dataclass instance to hold all mappings and counts
    index_map = IndexMap(
        user_ids=user_ids,
        item_ids=item_ids,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )

    if config.PRINT_CONFIRM:
        print("\nIndex mapping for MF prepared.")
        print("Number of users:", index_map.n_users)
        print("Number of items:", index_map.n_items)
        if config.ADVANCED_PRINT_MODE:
            print("Example user_id to index:", list(index_map.user_to_index.items())[:5])
            print("Example item_id to index:", list(index_map.item_to_index.items())[:5])
            print("Example index to user_id:", list(index_map.index_to_user.items())[:5])
            print("Example index to item_id:", list(index_map.index_to_item.items())[:5])

    return index_map


def get_data(
    config: Config,
    general_vars: GeneralVariables,
):
    '''
    Prepares the training and testing data for matrix factorization,
    which are lists of tuples in the form (user_index, item_index, rating).
    This format is suitable for training MF models.
    '''
    
    # Converting user_id and item_id to their corresponding indices using the index mapping
    train_data = [
        (
            general_vars.index_map.user_to_index[row.user_id],
            general_vars.index_map.item_to_index[row.item_id],
            float(row.rating),
        )
        for row in general_vars.train_df.itertuples(index=False)
    ]

    # Converting user_id and item_id to their corresponding indices for the test set as well
    test_data = [
        (
            general_vars.index_map.user_to_index[row.user_id],
            general_vars.index_map.item_to_index[row.item_id],
            float(row.rating),
        )
        for row in general_vars.test_df.itertuples(index=False)
    ]

    if config.PRINT_CONFIRM:
        print("\nData prepared for MF.")
        print("Number of training tuples:", len(train_data))
        print("Number of test tuples:", len(test_data))
        if config.ADVANCED_PRINT_MODE:
            print("Example training data (user_index, item_index, rating):", train_data[:5])
            print("Example test data (user_index, item_index, rating):", test_data[:5])
            
    return train_data, test_data


def init_mf_params(
    general_vars: GeneralVariables,
    dim: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Initializes the parameters for matrix factorization models, including:
    - mu: global mean rating
    - P: user latent factor matrix (n_users x dim)
    - Q: item latent factor matrix (n_items x dim)
    - bu: user bias vector (n_users)
    - bi: item bias vector (n_items)
    The latent factors are initialized with small random values, while biases are initialized to zero.
    '''
    
    mu = np.mean([rating for _, _, rating in general_vars.train_data])
    P = rng.normal(0, 0.1, size=(general_vars.index_map.n_users, dim))
    Q = rng.normal(0, 0.1, size=(general_vars.index_map.n_items, dim))
    bu = np.zeros(general_vars.index_map.n_users)
    bi = np.zeros(general_vars.index_map.n_items)
    return mu, P, Q, bu, bi


def predict_mf(
    general_vars: GeneralVariables,
    model: dict,
    user_id: int,
    item_id: int,
) -> float:
    '''Predicts the rating for a given user_id and item_id using the MF model parameters.'''
    
    # Converting user_id and item_id to their corresponding indices using the index mapping
    u_idx = general_vars.index_map.user_to_index[user_id]
    i_idx = general_vars.index_map.item_to_index[item_id]

    # Calculating the predicted rating using the MF model parameters
    # (global mean, user bias, item bias, and dot product of latent factors)
    pred = (
        model["mu"]
        + model["bu"][u_idx]
        + model["bi"][i_idx]
        + np.dot(model["P"][u_idx], model["Q"][i_idx])
    )

    # Clipping the predicted rating to be within the valid range (1.0 to 5.0) and returning it
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
    '''Computes the Root Mean Squared Error (RMSE) for a given dataset and MF model.'''
    
    # Computing the squared errors for each (user_index, item_index, rating) tuple in the dataset
    errors = [0.0] * len(data)
    i: int = 0
    for u_idx, i_idx, rating in data:
        # Converting indices back to user_id and item_id using the index mapping
        user_id = general_vars.index_map.index_to_user[u_idx]
        item_id = general_vars.index_map.index_to_item[i_idx]

        # Predicting the rating using the MF model for this user-item pair
        pred = predict_mf(general_vars, model, user_id, item_id)

        # Calculating the squared error for this prediction and storing it in the errors list
        errors[i] = (rating - pred) ** 2
        i += 1
    
    # Calculating the RMSE by taking the square root of the mean of the squared errors
    return float(np.sqrt(np.mean(errors)))


def print_rmse(
    general_vars: GeneralVariables,
    model: dict,
):
    '''Computes and prints the RMSE for both training and test datasets.'''
    
    # Computing RMSE for the test set
    test_rmse = compute_rmse(general_vars, general_vars.test_data, model)

    # Computing RMSE for the training set
    train_rmse = compute_rmse(general_vars, general_vars.train_data, model)

    # Printing the RMSE values for both datasets
    print("Test RMSE:", test_rmse)
    print("Train RMSE:", train_rmse)

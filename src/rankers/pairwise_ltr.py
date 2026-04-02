import numpy as np

from core.config import Config
from core.structs import GeneralVariables, LTR_Variables, PopularityVariables
from core.candidates import get_candidates
from rankers.mf_general import predict_mf


def sigmoid(x: float) -> float:
    '''Computes the sigmoid function with clipping to prevent overflow.'''
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def build_ltr_feature_vector(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    mf_model: dict,
    user_id: int,
    item_id: int,
) -> np.ndarray:
    '''
    Builds the feature vector for the LTR model.
    The features are:
    - s_mf: the predicted rating from the MF model for this user-item pair
    - bu: the user bias from the MF model for this user
    - bi: the item bias from the MF model for this item
    - pop: the popularity of the item (if LTR_USE_POPULARITY is True, otherwise 0)
    - 1.0: a constant feature to allow the model to learn an overall bias term
    '''
    
    # Getting the user and item indices from the general variables' index maps
    u_idx = general_vars.index_map.user_to_index[user_id]
    i_idx = general_vars.index_map.item_to_index[item_id]

    # Calculating the features for this user-item pair using the MF model and popularity information
    s_mf = predict_mf(general_vars, mf_model, user_id, item_id)
    bu_val = float(mf_model["bu"][u_idx])
    bi_val = float(mf_model["bi"][i_idx])
    pop_val = (
        float(popularity_vars.item_popularity.get(item_id, 0)) if config.LTR_USE_POPULARITY else 0.0
    )

    # Returning the feature vector as a numpy array
    return np.array([s_mf, bu_val, bi_val, pop_val, 1.0], dtype=float)


def build_pairwise_training_data(
    config: Config,
    general_vars: GeneralVariables,
) -> list[tuple[int, int, int]]:
    """
    Builds the training data for pairwise LTR, which consists of tuples in
    the form (user_id, positive_item_id, negative_item_id). Positive items
    are those with ratings >= 4, while negative items have ratings < 4.
    The number of pairs per user can be limited to control the training time.
    """

    # Using a random generator to sample pairs if there are too many
    rng = np.random.default_rng(config.RANDOM_STATE)
    pairwise_examples = []

    # Grouping the training data by user_id to create pairs of positive and negative items
    grouped = general_vars.train_df.groupby("user_id")

    # For each user, identifying positive and negative items and creating pairs
    for user_id, user_ratings in grouped:
        # Identifying positive and negative items based on the rating threshold
        positives = user_ratings[user_ratings["rating"] >= 4]["item_id"].tolist()
        negatives = user_ratings[user_ratings["rating"] < 4]["item_id"].tolist()

        # If there are no positive or no negative items for this user, cannot create pairs, so skip
        if len(positives) == 0 or len(negatives) == 0:
            continue

        # Creating all possible pairs of (positive_item, negative_item) for this user
        possible_pairs = [(i_pos, i_neg) for i_pos in positives for i_neg in negatives]

        # If there are too many pairs, sample a subset to limit the training time
        if len(possible_pairs) > config.LTR_MAX_PAIRS_PER_USER:
            sampled_indices = rng.choice(
                len(possible_pairs), size=config.LTR_MAX_PAIRS_PER_USER, replace=False
            )
            selected_pairs = [possible_pairs[idx] for idx in sampled_indices]
        else:
            selected_pairs = possible_pairs

        # Adding the selected pairs to the training data with the user_id
        for i_pos, i_neg in selected_pairs:
            pairwise_examples.append((int(user_id), int(i_pos), int(i_neg)))

    if config.PRINT_CONFIRM:
        print("\nPairwise training data built.")
        print("Number of training pairs", len(pairwise_examples))
        if config.ADVANCED_PRINT_MODE:
            print(
                "Example training pairs (user_id, positive_item_id, negative_item_id): ",
                pairwise_examples[:5],
            )

    return pairwise_examples


def train_pairwise_ltr(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
) -> dict:
    """
    Trains a pairwise Learning to Rank (LTR) model using stochastic gradient descent.
    The model learns a weight vector (phi) that combines features such as MF score, user
    bias, item bias, and popularity to predict the relative preference of items for a user.

    The training data consists of pairs of items (positive and negative) for each user, and the model
    is optimized to score positive items higher than negative items. The function returns a dictionary
    containing the learned parameters, training history, and configuration details.
    """

    # Initializing the random number generator for reproducibility
    rng = np.random.default_rng(config.RANDOM_STATE)

    # Initializing the weight vector (phi) for the LTR model, which will be learned during training
    n_features = 5
    phi = np.zeros(n_features, dtype=float)

    history = [None] * config.LTR_EPOCHS
    train_data = ltr_vars.train_data.copy()

    # Training loop for the specified number of epochs
    for epoch in range(config.LTR_EPOCHS):
        # Shuffling the training data at the beginning of each epoch to ensure randomness in SGD updates
        rng.shuffle(train_data)

        # Initializing the total loss for this epoch to compute the average pairwise loss
        total_loss = 0.0

        # Iterating over each training example (user_id, positive_item_id, negative_item_id) to perform SGD updates
        for user_id, i_pos, i_neg in train_data:
            # Building the feature vectors for the positive and negative items
            # for this user
            f_pos = build_ltr_feature_vector(
                config, general_vars, popularity_vars, ltr_vars.mf_model, user_id, i_pos
            )
            f_neg = build_ltr_feature_vector(
                config, general_vars, popularity_vars, ltr_vars.mf_model, user_id, i_neg
            )
            
            # Calculating the score difference between the positive and negative
            # items using the current weight vector (phi)
            delta_f = f_pos - f_neg
            score = np.dot(phi, delta_f)

            # Calculating the pairwise loss using the logistic loss function and
            # accumulating it to compute the average loss for this epoch
            prob = sigmoid(score)
            loss = -np.log(max(prob, 1e-12))
            total_loss += loss

            # Calculating the gradient of the loss with respect to the weight vector (phi)
            grad = (prob - 1.0) * delta_f + config.LTR_REG * phi
            phi -= config.LTR_LR * grad

        # Calculating the average pairwise loss for this epoch and storing it in the history
        avg_loss = total_loss / len(train_data)
        history[epoch] = avg_loss

        print(f"LTR Epoch {epoch + 1}/{config.LTR_EPOCHS} - Avg Pairwise Loss: {avg_loss:.6f}")

    # Returning the learned parameters, training history, and configuration details in a dictionary
    return {
        "phi": phi,
        "history": history,
        "params": {"epochs": config.LTR_EPOCHS, "lr": config.LTR_LR, "reg": config.LTR_REG},
        "use_popularity": config.LTR_USE_POPULARITY,
    }


def predict_pairwise_ltr(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    user_id: int,
    item_id: int,
) -> float:
    features = build_ltr_feature_vector(
        config,
        general_vars,
        popularity_vars,
        ltr_vars.mf_model,
        user_id,
        item_id,
    )
    score = np.dot(ltr_vars.model["phi"], features)
    return float(score)


def recommend_pairwise_ltr(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    user_id: int,
) -> list[int]:
    candidates = get_candidates(general_vars, user_id)

    scored_items = []
    for item_id in candidates:
        score = predict_pairwise_ltr(
            config, general_vars, popularity_vars, ltr_vars, user_id, item_id
        )
        scored_items.append((item_id, score))

    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return [int(item_id) for item_id, _ in ranked_items[: config.TOP_K]]

import numpy as np

from core.config import Config
from core.structs import GeneralVariables, LTR_Variables, PopularityVariables
from core.utils import get_candidates
from rankers.mf_general import predict_mf


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def build_ltr_feature_vector(
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    mf_model: dict,
    user_id: int,
    item_id: int,
) -> np.ndarray:
    u_idx = general_vars.user_to_index[user_id]
    i_idx = general_vars.item_to_index[item_id]

    s_mf = predict_mf(general_vars, mf_model, user_id, item_id)
    bu_val = float(mf_model["bu"][u_idx])
    bi_val = float(mf_model["bi"][i_idx])
    pop_val = float(popularity_vars.item_popularity.get(item_id, 0))

    return np.array([s_mf, bu_val, bi_val, pop_val, 1.0], dtype=float)


def build_pairwise_training_data(
    config: Config,
    general_vars: GeneralVariables,
) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    pairwise_examples = []

    grouped = general_vars.train_df.groupby("user_id")

    for user_id, user_ratings in grouped:
        positives = user_ratings[user_ratings["rating"] >= 4]["item_id"].tolist()
        negatives = user_ratings[user_ratings["rating"] < 4]["item_id"].tolist()

        if len(positives) == 0 or len(negatives) == 0:
            continue

        possible_pairs = [(i_pos, i_neg) for i_pos in positives for i_neg in negatives]

        if len(possible_pairs) > config.LTR_MAX_PAIRS_PER_USER:
            sampled_indices = rng.choice(
                len(possible_pairs), size=config.LTR_MAX_PAIRS_PER_USER, replace=False
            )
            selected_pairs = [possible_pairs[idx] for idx in sampled_indices]
        else:
            selected_pairs = possible_pairs

        for i_pos, i_neg in selected_pairs:
            pairwise_examples.append((int(user_id), int(i_pos), int(i_neg)))

    return pairwise_examples


def print_examples_pairwise_ltr(pairwise_train_data: list[tuple[int, int, int]]) -> None:
    print("Number of training pairs", len(pairwise_train_data))
    print("Example:", pairwise_train_data[:5])


def train_pairwise_ltr(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    pairwise_vars: LTR_Variables,
    mf_model: dict,
) -> dict:
    rng = np.random.default_rng(config.RANDOM_STATE)

    n_features = 5
    phi = np.zeros(n_features, dtype=float)

    history = []
    train_data = pairwise_vars.train_data.copy()

    for epoch in range(config.LTR_EPOCHS):
        rng.shuffle(train_data)

        total_loss = 0.0

        for user_id, i_pos, i_neg in train_data:
            f_pos = build_ltr_feature_vector(
                general_vars, popularity_vars, mf_model, user_id, i_pos
            )
            f_neg = build_ltr_feature_vector(
                general_vars, popularity_vars, mf_model, user_id, i_neg
            )

            delta_f = f_pos - f_neg
            score = np.dot(phi, delta_f)

            prob = sigmoid(score)
            loss = -np.log(max(prob, 1e-12))
            total_loss += loss

            grad = (prob - 1.0) * delta_f + config.LTR_REG * phi
            phi -= config.LTR_LR * grad

        avg_loss = total_loss / len(train_data)
        history.append(avg_loss)

        if config.PRINT_CONFIRM:
            print(f"LTR Epoch {epoch + 1}/{config.LTR_EPOCHS} - Avg Pairwise Loss: {avg_loss:.6f}")

    return {
        "phi": phi,
        "history": history,
        "params": {"epochs": config.LTR_EPOCHS, "lr": config.LTR_LR, "reg": config.LTR_REG},
    }


def predict_pairwise_ltr(
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    user_id: int,
    item_id: int,
) -> float:
    features = build_ltr_feature_vector(general_vars, popularity_vars, ltr_vars.mf_model, user_id, item_id)
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
        score = predict_pairwise_ltr(general_vars, popularity_vars, ltr_vars, user_id, item_id)
        scored_items.append((item_id, score))

    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return [int(item_id) for item_id, _ in ranked_items[: config.TOP_K]]

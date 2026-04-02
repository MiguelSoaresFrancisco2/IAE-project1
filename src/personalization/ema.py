import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.config import Config

from core.structs import EMA_Variables, GeneralVariables, PopularityVariables
from core.candidates import get_candidates
from rankers.pairwise_ltr import predict_pairwise_ltr
from core.structs import LTR_Variables


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length. If the vector has zero norm, it is returned unchanged."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def normalize_scores_minmax(scores: list[float]) -> list[float]:
    """Normalizes a list of scores to the range [0, 1] using min-max scaling."""

    # If the list of scores is empty, return an empty list to avoid errors
    if not scores:
        return []

    # Calculating the minimum and maximum scores to perform min-max normalization
    min_score = min(scores)
    max_score = max(scores)

    # If all scores are the same (max_score == min_score), return a list of 0.0 to avoid division by zero
    if max_score == min_score:
        return [0.0 for _ in scores]

    # Normalizing each score to the range [0, 1] using the formula:
    # (score - min_score) / (max_score - min_score)
    denom = max_score - min_score
    return [(score - min_score) / denom for score in scores]


def normalize_affinity(affinity: float) -> float:
    """Maps cosine similarity from [-1, 1] to [0, 1]."""
    return (affinity + 1.0) / 2.0


def get_initial_user_state(
    general_vars: GeneralVariables,
    mf_model: dict,
    user_id: int,
) -> np.ndarray:
    """Initializes the user state vector for EMA based on the user's latent factors from the MF model."""
    u_idx = general_vars.index_map.user_to_index[user_id]
    return normalize_vector(mf_model["P"][u_idx].copy())


def ema_update_user_state(
    config: Config,
    general_vars: GeneralVariables,
    ema_vars: EMA_Variables,
    item_id: int,
) -> np.ndarray:
    i_idx = general_vars.index_map.item_to_index[item_id]
    item_vector = ema_vars.mf_model["Q"][i_idx]

    updated_state = (1 - config.EMA_RHO) * ema_vars.session_state + config.EMA_RHO * item_vector
    updated_state = normalize_vector(updated_state)

    return updated_state


def recommend_session_ltr(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
    user_id: int,
) -> list[int]:
    """
    Generates recommendations for a user during an EMA session by combining the
    base LTR scores with affinity scores based on the EMA-updated user state.
    """

    # Getting candidate items for the user and filtering out those already seen in the session
    candidates = get_candidates(general_vars, user_id)
    candidates = [item_id for item_id in candidates if item_id not in ema_vars.seen_in_session]

    # Calculating base LTR scores and affinity scores for each candidate item, then combining them
    base_scores = [None] * len(candidates)
    affinities = [None] * len(candidates)
    for i, item_id in enumerate(candidates):
        # Calculating the base LTR score for this user-item pair using the pairwise LTR model
        base_scores[i] = predict_pairwise_ltr(
            config,
            general_vars,
            popularity_vars,
            ltr_vars,
            user_id,
            item_id,
        )

        # Calculating the affinity score based on the cosine similarity between the
        # EMA-updated user state and the item latent factor vector from the MF model
        i_idx = general_vars.index_map.item_to_index[item_id]
        item_vector = normalize_vector(ema_vars.mf_model["Q"][i_idx])
        affinities[i] = float(np.dot(ema_vars.session_state, item_vector))

    # Normalizing the base LTR scores and affinity scores to ensure they are on
    # comparable scales before combining them
    base_scores_norm = normalize_scores_minmax(base_scores)
    affinity_scores_norm = [normalize_affinity(a) for a in affinities]

    # Combining the normalized base LTR scores and affinity scores by summing
    # them to get a final score for each candidate item
    scored_items = [
        (item_id, base_scores_norm[i] + affinity_scores_norm[i])
        for i, item_id in enumerate(candidates)
    ]

    # Sorting the candidate items by their combined scores in descending order
    # and returning the top-K item ids as recommendations
    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return [int(item_id) for item_id, _ in ranked_items[: config.TOP_K]]


def simulate_user_choice(
    general_vars: GeneralVariables,
    ema_vars: EMA_Variables,
) -> int:
    """
    Simulates the user's choice from the recommended items based on the affinity
    between the EMA-updated user state and the item vectors of the recommended items.
    """

    best_item = None
    best_score = -float("inf")

    # Iterating over the recommended items and calculating the affinity score for each item
    for item_id in ema_vars.recommended_items:
        i_idx = general_vars.index_map.item_to_index[item_id]
        item_vector = normalize_vector(ema_vars.mf_model["Q"][i_idx])
        affinity = float(np.dot(ema_vars.session_state, item_vector))

        # Keeping track of the item with the highest affinity score, which is considered the user's choice
        if affinity > best_score:
            best_score = affinity
            best_item = item_id

    return int(best_item)


def run_user_session(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
    user_id: int,
) -> dict:
    """
    Runs an interactive recommendation session for a single user, where the system
    generates recommendations, the user simulates a choice based on affinity, and
    the system updates its state using EMA. The session logs are collected and
    returned in a structured format.
    """

    # Initializing the session state, seen items, and logs for this user session
    ema_vars.session_state = get_initial_user_state(general_vars, ema_vars.mf_model, user_id)
    ema_vars.seen_in_session = set()

    # Running multiple rounds of recommendation, choice simulation, and state updates for this user session
    session_logs = [None] * config.EMA_ROUNDS
    for round_idx in range(1, config.EMA_ROUNDS + 1):
        # Generating recommendations for the user using the session-based LTR method
        # that incorporates EMA state
        ema_vars.recommended_items = recommend_session_ltr(
            config,
            general_vars,
            popularity_vars,
            ltr_vars,
            ema_vars,
            user_id,
        )

        # Simulating the user's choice based on the affinity between the
        # EMA-updated user state and the item vectors of the recommended items
        chosen_item = simulate_user_choice(general_vars, ema_vars)

        # Updating the EMA user state based on the chosen item and adding it
        # to the set of seen items in the session
        ema_vars.seen_in_session.add(chosen_item)
        ema_vars.session_state = ema_update_user_state(
            config,
            general_vars,
            ema_vars,
            chosen_item,
        )

        # Creating a summary of the current session state (first 5 dimensions)
        # to include in the logs for analysis
        state_summary = np.round(ema_vars.session_state[:5], 4).tolist()

        # Logging the details of this round in the session logs, including the recommended items,
        # the chosen item, and the state vector summary
        session_logs[round_idx - 1] = {
            "round": round_idx,
            "recommended_items": ema_vars.recommended_items,
            "recommended_titles": general_vars.items[
                general_vars.items["item_id"].isin(ema_vars.recommended_items)
            ]["title"].tolist(),
            "chosen_item": chosen_item,
            "chosen_title": general_vars.items.loc[
                general_vars.items["item_id"] == chosen_item, "title"
            ].values[0],
            "state_vector_head": state_summary,
        }

    # Returning a dictionary containing the user ID, base MF method, rho value,
    # number of rounds, and the session logs for this user's EMA session
    return {
        "user_id": int(user_id),
        "mf_method": ema_vars.mf_method_name,
        "rho": config.EMA_RHO,
        "rounds": config.EMA_ROUNDS,
        "logs": session_logs,
    }


def print_session_summary(session_result: dict) -> None:
    """
    Prints a summary of the EMA session results for a single user,
    including the user ID, rho value, and details of each round.
    """

    print(f"User ID: {session_result['user_id']}")
    print(f"rho: {session_result['rho']}")
    print("-" * 80)

    for log in session_result["logs"]:
        print(f"Round {log['round']}")
        print("Recommended item ids:", log["recommended_items"])
        print("Chosen item:", log["chosen_item"], "-", log["chosen_title"])
        print("-" * 80)


def get_session_results(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    pairwise_ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
) -> pd.DataFrame:
    """
    Runs interactive recommendation sessions for each user in ema_vars.session_users
    and collects logs of recommendations, choices, and state updates.
    """

    # Initializing a list to store session results for each user
    session_results = [None] * len(ema_vars.session_users)

    # Running a session for each user and storing the results in the session_results list
    for i, user_id in enumerate(ema_vars.session_users):
        session_result = run_user_session(
            config,
            general_vars,
            popularity_vars,
            pairwise_ltr_vars,
            ema_vars,
            user_id,
        )
        session_results[i] = session_result

    if config.PRINT_CONFIRM:
        print("\nCompleted EMA sessions for all users.")
        print("Number of sessions:", len(session_results))
        if config.ADVANCED_PRINT_MODE:
            print("\nSession results:")
            for session_result in session_results:
                print_session_summary(session_result)
                print("\n")

    return session_results


def compare_rho_values(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
) -> list[dict]:
    '''
    Runs EMA sessions for different rho values specified in the config
    and collects the results for comparison.
    '''
    
    # Initializing a list to store session results for each rho value
    session_results_rho_comparison = []
    original_rho = config.EMA_RHO

    # Running sessions for each rho value in the config and collecting the results
    # in the session_results_rho_comparison list
    for rho_value in config.EMA_RHO_VALUES:
        config.EMA_RHO = rho_value
        for user_id in ema_vars.session_users:
            session_result = run_user_session(
                config,
                general_vars,
                popularity_vars,
                ltr_vars,
                ema_vars,
                user_id,
            )
            session_results_rho_comparison.append(session_result)

    # Restoring the original rho value in the config after the comparisons are done
    config.EMA_RHO = original_rho
    
    if config.PRINT_CONFIRM:
        print("\nCompleted EMA sessions for rho comparison.")
        print("Rho comparison sessions:", len(session_results_rho_comparison))

    return session_results_rho_comparison


def summarize_rho_overlap(session_results: list[dict]) -> pd.DataFrame:
    '''
    Summarizes the average overlap of recommended items between consecutive rounds
    for different rho values. The overlap is calculated using the Jaccard similarity
    between the sets of recommended items in consecutive rounds.
    '''
    
    rows = []

    # Iterating over the session results and calculating the average overlap of
    # recommended items between consecutive rounds for each rho value
    for session in session_results:
        # Extracting the rho value and logs from the session results to
        # calculate the overlap
        rho = session.get("rho")
        logs = session.get("logs", [])

        # Calculating the overlap between the sets of recommended items in consecutive rounds
        for idx in range(1, len(logs)):
            # Getting the sets of recommended items from the current and previous rounds
            prev_items = set(logs[idx - 1].get("recommended_items", []))
            curr_items = set(logs[idx].get("recommended_items", []))
            union_items = prev_items | curr_items

            # Calculating the Jaccard similarity (overlap) between the two
            # sets of recommended items
            if not union_items:
                overlap = 0.0
            else:
                overlap = len(prev_items & curr_items) / len(union_items)

            # Appending the rho value, round number, and calculated overlap to
            # the rows list for summarization
            rows.append(
                {
                    "rho": rho,
                    "round": logs[idx].get("round"),
                    "overlap": overlap,
                }
            )

    # If there are no rows (which can happen if there were no sessions or logs),
    # return an empty DataFrame with the appropriate columns
    if not rows:
        return pd.DataFrame(columns=["rho", "round", "overlap"])

    # Converting the list of rows into a DataFrame and calculating the average overlap
    # for each combination of rho value and round number by grouping the DataFrame
    summary_df = pd.DataFrame(rows).groupby(["rho", "round"], as_index=False)["overlap"].mean()

    return summary_df


def plot_rho_overlap(
    config: Config,
    summary_df: pd.DataFrame,
    method_name: str,
) -> None:
    '''
    Plots the average overlap of recommended items between consecutive rounds
    for different rho values. The plot shows how the choice of rho affects the
    stability of recommendations across rounds, with a higher overlap indicating
    more stable recommendations.
    '''
    
    if summary_df.empty:
        return

    plt.figure(figsize=(8, 5))

    for rho_value, group in summary_df.groupby("rho"):
        group_sorted = group.sort_values("round")
        plt.plot(
            group_sorted["round"],
            group_sorted["overlap"],
            marker="o",
            label=f"rho={rho_value}",
        )

    plt.xlabel("Round")
    plt.ylabel("Avg slate overlap (Jaccard)")
    plt.title(f"EMA rho effect on slate overlap ({method_name.upper()})")
    plt.grid(True)
    plt.legend()

    if config.SAVE_IMAGES:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/ema_rho_overlap_{method_name}.png")
    if config.SHOW_PLOTS:
        plt.show()

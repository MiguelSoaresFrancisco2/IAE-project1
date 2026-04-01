import numpy as np
import pandas as pd

from core.config import Config

from core.structs import EMA_Variables, GeneralVariables, PopularityVariables
from core.utils import get_candidates
from rankers.pairwise_ltr import predict_pairwise_ltr
from core.structs import LTR_Variables


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def get_initial_user_state(
    general_vars: GeneralVariables,
    mf_model: dict,
    user_id: str,
) -> np.ndarray:
    u_idx = general_vars.user_to_index[user_id]
    return normalize_vector(mf_model["P"][u_idx].copy())


def ema_update_user_state(
    config: Config,
    general_vars: GeneralVariables,
    ema_vars: EMA_Variables,
    item_id: int,
) -> np.ndarray:
    i_idx = general_vars.item_to_index[str(item_id)]
    item_vector = ema_vars.mf_model["Q"][i_idx]

    updated_state = (1 - config.EMA_RHO) * ema_vars.session_state + config.EMA_RHO * item_vector
    updated_state = normalize_vector(updated_state)

    return updated_state


def session_adjusted_score(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
    user_id: int,
    item_id: int,
) -> float:
    base_score = predict_pairwise_ltr(
        general_vars,
        popularity_vars,
        ltr_vars,
        user_id,
        item_id,
    )

    i_idx = general_vars.item_to_index[str(item_id)]
    item_vector = normalize_vector(ltr_vars.mf_model["Q"][i_idx])

    session_affinity = float(np.dot(ema_vars.session_state, item_vector))

    return (1 - config.EMA_BETA) * base_score + config.EMA_BETA * session_affinity


def recommend_session_ltr(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
    user_id: int,
) -> list[int]:

    candidates = get_candidates(general_vars, user_id)
    candidates = [item_id for item_id in candidates if item_id not in ema_vars.seen_in_session]

    scored_items = []
    for item_id in candidates:
        score = session_adjusted_score(
            config,
            general_vars,
            popularity_vars,
            ltr_vars,
            ema_vars,
            user_id,
            item_id,
        )
        scored_items.append((item_id, score))

    ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
    return [int(item_id) for item_id, _ in ranked_items[: config.TOP_K]]


def simulate_user_choice(
    general_vars: GeneralVariables,
    ema_vars: EMA_Variables,
) -> int:
    best_item = None
    best_score = -float("inf")

    for item_id in ema_vars.recommended_items:
        i_idx = general_vars.item_to_index[item_id]
        item_vector = normalize_vector(ema_vars.mf_model["Q"][i_idx])
        affinity = float(np.dot(ema_vars.session_state, item_vector))

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
    ema_vars.session_state = get_initial_user_state(general_vars, ema_vars.mf_model, user_id)
    ema_vars.seen_in_session = set()
    session_logs = []

    for round_idx in range(1, config.EMA_ROUNDS + 1):
        recommended = recommend_session_ltr(
            config,
            general_vars,
            popularity_vars,
            ltr_vars,
            ema_vars,
            user_id,
        )

        chosen_item = simulate_user_choice(general_vars, ema_vars)

        session_logs.append(
            {
                "round": round_idx,
                "recommended_items": recommended,
                "recommended_titles": general_vars.items[
                    general_vars.items["item_id"].isin(recommended)
                ]["title"].tolist(),
                "chosen_item": chosen_item,
                "chosen_title": general_vars.items.loc[
                    general_vars.items["item_id"] == chosen_item, "title"
                ].values[0],
            }
        )

        ema_vars.seen_in_session.add(chosen_item)
        ema_vars.session_state = ema_update_user_state(
            config,
            general_vars,
            ema_vars,
            chosen_item,
        )

    return {
        "user_id": int(user_id),
        "rho": config.EMA_RHO,
        "beta": config.EMA_BETA,
        "rounds": config.EMA_ROUNDS,
        "logs": session_logs,
    }


def print_session_summary(session_result: dict) -> None:
    print(f"User ID: {session_result['user_id']}")
    print(f"rho: {session_result['rho']}, beta: {session_result['beta']}")
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
    session_results = []

    for user_id in ema_vars.session_users:
        session_result = run_user_session(
            config,
            general_vars,
            popularity_vars,
            pairwise_ltr_vars,
            ema_vars,
            user_id,
        )
        session_results.append(session_result)

    if config.PRINT_CONFIRM:
        print("Number of sessions:", len(session_results))
        for session_result in session_results:
            print_session_summary(session_result)
            print("\n")

    return session_results


def get_session_summary_df(
    config: Config,
    ema_vars: EMA_Variables,
) -> pd.DataFrame:
    session_rows = []

    for session_result in ema_vars.session_results:
        for log in session_result["logs"]:
            session_rows.append(
                {
                    "user_id": session_result["user_id"],
                    "round": log["round"],
                    "chosen_item": log["chosen_item"],
                    "chosen_title": log["chosen_title"],
                }
            )

    session_summary_df = pd.DataFrame(session_rows)

    if config.PRINT_CONFIRM:
        print(session_summary_df)

    return session_summary_df


def get_session_recommendations(config: Config, ema_vars: EMA_Variables) -> pd.DataFrame:
    session_recommendation_rows = [None] * sum(len(sr["logs"]) for sr in ema_vars.session_results)
    idx = 0

    for session_result in ema_vars.session_results:
        for log in session_result["logs"]:
            session_recommendation_rows[idx] = {
                "user_id": session_result["user_id"],
                "round": log["round"],
                "recommended_items": log["recommended_items"],
                "chosen_item": log["chosen_item"],
                "chosen_title": log["chosen_title"],
            }
            idx += 1

    session_recommendations_df = pd.DataFrame(session_recommendation_rows)

    if config.PRINT_CONFIRM:
        print("Session Recommendations DataFrame:")
        print(session_recommendations_df.head())

    return session_recommendations_df


def compare_rho_values(
    config: Config,
    general_vars: GeneralVariables,
    popularity_vars: PopularityVariables,
    ltr_vars: LTR_Variables,
    ema_vars: EMA_Variables,
) -> None:
    session_results_rho_comparison = []

    for rho_value in [0.1, 0.3]:
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

    print("Sessões rho comparison:", len(session_results_rho_comparison))

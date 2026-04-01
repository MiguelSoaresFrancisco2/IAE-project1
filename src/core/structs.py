import pandas as pd
import numpy as np

from collections import defaultdict


class GeneralVariables:
    items: pd.DataFrame = None
    ratings: pd.DataFrame = None

    train_df: pd.DataFrame = None
    test_df: pd.DataFrame = None

    all_users: list = None
    all_items: set = None

    train_items_by_user: dict = None
    relevant_items_by_user: dict = None

    item_genre_vectors: dict = None

    eligible_users: list = None

    unique_user_ids: list = None
    unique_item_ids: list = None
    user_to_index: dict = None
    item_to_index: dict = None
    index_to_user: dict = None
    index_to_item: dict = None

    n_users: int = None
    n_items: int = None

    train_data: list = None
    test_data: list = None

    user_ratings_train: defaultdict[list] = None
    item_ratings_train: defaultdict[list] = None

    done_methods_names: set = None

    def __init__(self):
        self.done_methods_names = set()


class PopularityVariables:
    item_popularity: dict = None

    results: list = None
    results_df: pd.DataFrame = None


class MF_Variables:
    model: dict = None
    method_name: str

    results: list = None
    results_df: pd.DataFrame = None

    hyperparameters: dict = None

    def __init__(self, method_name: str):
        self.method_name = method_name


class LTR_Variables:
    model: dict = None
    method_name: str

    mf_model: dict = None

    train_data: list[tuple[int, int, int]] = None

    results: list = None
    results_df: pd.DataFrame = None

    hyperparameters: dict = None

    def __init__(self, method_name: str):
        self.method_name = method_name


class MMR_Variables:
    results: list = None
    results_df: pd.DataFrame = None

    hyperparameters: dict = None


class EMA_Variables:
    session_users: list[int] = None
    session_state: np.ndarray = None
    seen_in_session: set = None
    
    mf_model: dict = None
    recommended_items: list[int] = None
    
    session_results: list = None
    session_summary_df: pd.DataFrame = None
    session_recommendations_df: pd.DataFrame = None


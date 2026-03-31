import pandas as pd


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


class PopularityVariables:
    item_popularity: dict = None

    results: list = None
    results_df: pd.DataFrame = None


class MF_SGDVariables:
    model: dict = None
    model_name: str = "mf_sgd"

    results: list = None
    results_df: pd.DataFrame = None

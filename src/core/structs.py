import pandas as pd
import numpy as np

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class IndexMap:
    user_ids: list[int]
    item_ids: list[int]
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]
    index_to_user: dict[int, int]
    index_to_item: dict[int, int]

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


@dataclass
class ResultBundle:
    rows: list[dict]
    df: pd.DataFrame

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "ResultBundle":
        return cls(rows=rows, df=pd.DataFrame(rows))


class GeneralVariables:
    # Main data variables
    items: pd.DataFrame = None
    ratings: pd.DataFrame = None

    all_users: list = None
    all_items: set = None

    item_genre_vectors: dict = None
    eligible_users: list = None

    # Index mapping
    index_map: IndexMap | None = None

    # Train/test splits data and DataFrames
    train_df: pd.DataFrame = None
    test_df: pd.DataFrame = None
    train_data: list = None
    test_data: list = None

    # Other precomputed variables
    train_items_by_user: dict = None
    relevant_items_by_user: dict = None

    user_ratings_train: defaultdict[list] = None
    item_ratings_train: defaultdict[list] = None

    done_methods_names: set = None

    def __init__(self):
        self.done_methods_names = set()


class PopularityVariables:
    item_popularity: dict = None

    results: ResultBundle | None = None


class MF_Variables:
    model: dict = None
    method_name: str

    results: ResultBundle | None = None

    hyperparameters: dict = None

    def __init__(self, method_name: str):
        self.method_name = method_name


class LTR_Variables:
    model: dict = None
    method_name: str

    mf_model: dict = None

    train_data: list[tuple[int, int, int]] = None

    results: ResultBundle | None = None

    hyperparameters: dict = None

    def __init__(self, method_name: str):
        self.method_name = method_name


class MMR_Variables:
    results: ResultBundle | None = None

    hyperparameters: dict = None


class EMA_Variables:
    session_users: list[int] = None
    session_state: np.ndarray = None
    seen_in_session: set = None

    mf_model: dict = None
    mf_method_name: str = None
    recommended_items: list[int] = None

    session_results: list = None

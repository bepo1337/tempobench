import pandas as pd
import numpy as np
from benchmark.utils import all_num_col_names, all_cat_cols_without_ids_and_player_names
from fastdtw import fastdtw
from typing import List, Tuple


def normalized_l1_distance(matrix_A: pd.DataFrame, matrix_B: pd.DataFrame):
    """Normalized Manhattan L1 distance.
    All cell entries need to be positive as otherwise the normalization wont work"""
    matrix_A, matrix_B = matrix_A.values, matrix_B.values
    diff = np.abs(matrix_A - matrix_B)
    max_sum = np.sum(np.maximum(matrix_A, matrix_B))
    return np.sum(diff) / max_sum if max_sum != 0 else 0

def absolute_l1_distance(matrix_A: pd.DataFrame, matrix_B: pd.DataFrame):
    """Absolute Manhattan L1 distance"""
    diff = np.abs(matrix_A - matrix_B)
    return np.sum(diff).sum()


def create_tuples_from_df(a: pd.DataFrame) -> List[Tuple]:
    """Assumes the dataframe is sorted already. Returns a list of the tuples required for fastdtw"""
    list_of_all_relevant_cols = [*all_num_col_names, *all_cat_cols_without_ids_and_player_names]
    time_series_as_tuple = list(a[list_of_all_relevant_cols].itertuples(index=False, name=None))
    return time_series_as_tuple

def weighted_mixed_distance(sequence1, sequence2: List[Tuple], cat_weight=0.3):
    num_cols_idx = len(all_num_col_names)
    len_tuple = len(sequence1)

    num_dist = np.linalg.norm(np.array(sequence1[:num_cols_idx]) - np.array(sequence2[:num_cols_idx])) # is euclid norm
    cat_dist = sum(1 if sequence1[i] != sequence2[i] else 0 for i in range(num_cols_idx, len_tuple)) * cat_weight
    return num_dist + cat_dist


def combined_dtw(a,b: pd.DataFrame):
    """Combined Dynamic Time Warping with custom distance function to incorporate categorical values.
    Assumes a and b are already normalized and encoded"""

    a_tuples = create_tuples_from_df(a)
    b_tuples = create_tuples_from_df(b)
    distance, path = fastdtw(a_tuples, b_tuples, dist=weighted_mixed_distance)
    return distance, path
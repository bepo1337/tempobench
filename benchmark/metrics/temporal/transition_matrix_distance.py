import statistics

import numpy as np
import pandas

from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    VALIDITY_START, VALIDITY_END, normalized_l1_distance, FIRST_NAME, LAST_NAME, PSEUDONYM, POSITION, FOOT, CITIZENSHIP, \
    absolute_l1_distance
from benchmark.metrics import BenchmarkMetric
import pandas as pd


TransitionMatrixDistanceMetricName = "TransitionMatrixDistance"
transition_matrix_skip_columns = [
    PLAYER_ID,
    COACH_ID,
    LEAGUE_ID,
    CLUB_ID,
    SEASON_ID,
    FIRST_NAME,
    LAST_NAME,
    PSEUDONYM,
    POSITION,
    FOOT,
    CITIZENSHIP
]

class TransitionMatrixDistance(BenchmarkMetric):
    """Computes the distance between the 1-lag transition matrices of the real and synthesized data set."""

    def compute(self, X_real, X_syn) -> dict:
        result = {"absolute": {}, "normalized": {}}
        abs_distances = []
        norm_distances = []
        for col_name in X_real.columns.values.tolist():
            # look at categorical and ordinal features
            if (column_to_type[col_name] != ColumnType.CATEGORICAL and column_to_type[col_name] != ColumnType.ORDINAL) or col_name in transition_matrix_skip_columns:
                continue

            abs_distance, normalized_distance = self._compute_transition_matrix_distance(X_real, X_syn, col_name)
            result["absolute"][col_name] = abs_distance
            abs_distances.append(abs_distance)
            result["normalized"][col_name] = normalized_distance
            norm_distances.append(normalized_distance)

        average_abs_distance = statistics.mean(abs_distances)
        average_norm_distance = statistics.mean(norm_distances)
        result["average_abs_distance"] = average_abs_distance
        result["average_norm_distance"] = average_norm_distance
        return result

    def name(self) -> str:
        return TransitionMatrixDistanceMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

    def _compute_transition_matrix_distance(self, real_df, syn_df: pandas.DataFrame, col_name: str) -> (float, float):
        real_df_copy = real_df.copy()
        syn_df_copy = syn_df.copy()

        real_transition_matrix = self._transition_matrix(real_df_copy, col_name)
        syn_transition_matrix = self._transition_matrix(syn_df_copy, col_name)

        # if synth has different categories we need to consider this
        # need both index and columns as index is starting point and column transition end
        all_categories = set(real_transition_matrix.index) | set(real_transition_matrix.columns) | \
                         set(syn_transition_matrix.index) | set(syn_transition_matrix.columns)

        # fill missing values with zeros
        real_transition_matrix = real_transition_matrix.reindex(index=all_categories, columns=all_categories, fill_value=0)
        syn_transition_matrix = syn_transition_matrix.reindex(index=all_categories, columns=all_categories, fill_value=0)
        return self._matrix_diff(real_transition_matrix, syn_transition_matrix)

    def _transition_matrix(self, df: pandas.DataFrame, col_name: str):
        df[VALIDITY_START] = pd.to_datetime(df[VALIDITY_START])  # make Dates comparable to sort them

        df = df.sort_values(by=[PLAYER_ID, VALIDITY_START])

        # Compute transitions within each entty
        # groupby preserves ordering within each group
        df["next_category"] = df.groupby(PLAYER_ID)[col_name].shift(-1)  # write next_category by shifting rows in groups
        df = df.dropna()  # last rows dont have transition, therefore drop them

        # matrix with transition counts
        transition_counts = df.groupby([col_name, "next_category"]).size().unstack(fill_value=0)

        # normalize each row in the matrix to get probabilities
        # axis1 = column wise sum, axis0 = divide row (index matching)
        transition_matrix = transition_counts.div(transition_counts.sum(axis='columns'), axis='index')
        return transition_matrix

    def _matrix_diff(self, a,b: pandas.DataFrame) -> (float, float):
        abs_distance = absolute_l1_distance(a, b)
        # number of possible values in category
        K = a.shape[0]
        # 2 is the maximum distance per row if the transition sets are disjunct
        # for example first matrix: A-->B: 1 and the second matrix does A-->C: 1. Then the difference is 2 which is the maximum.
        # This also applies for league for example if the synthesized data set makes up a value and has 100% probabilty in it.
        normalized_distance = abs_distance/(2*K)
        return abs_distance, normalized_distance
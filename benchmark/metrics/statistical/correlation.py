from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, LAST_TRANSFER_FEE, columns_with_ids, COACH_ID, columns_with_dates, \
    absolute_l1_distance, is_numerical, is_categorical_or_ordinal, all_positions, POSITION, PLAYER_ID, FIRST_NAME, \
    LAST_NAME, PSEUDONYM
from benchmark.utils import MARKET_VALUE, column_to_type, ColumnType
import numpy as np
import pandas as pd
import scipy.stats as stats
import joblib
import os
import json
import tempfile



CorrelationMetricName = "Correlation"
correlation_skip_columns = [
    *columns_with_ids,
    *columns_with_dates,
    FIRST_NAME, LAST_NAME, PSEUDONYM
]

class Correlation(BenchmarkMetric):
    """Computes the correlation of features"""
    def compute(self, X_real, X_syn) -> dict:
        X_real = X_real.copy()
        X_syn = X_syn.copy()
        # drop columns that we wanna skip
        X_real.drop(columns=correlation_skip_columns, inplace=True)
        X_syn.drop(columns=correlation_skip_columns, inplace=True)
        # Null values should also be included in the correlation as its own category
        X_real.fillna("other", inplace=True)
        X_syn.fillna("other", inplace=True)
        correlations = {}
        correlations["dataset"] = self._compute_stats(X_real, X_syn, True)

        for position in all_positions:
            real_copy = X_real.copy()
            syn_copy = X_syn.copy()
            real_copy = real_copy[real_copy[POSITION] == position]
            syn_copy = syn_copy[syn_copy[POSITION] == position]
            if len(real_copy) == 0 or len(syn_copy) == 0:
                correlations[position] = "MISSING: No data for this position; cant compute correlation"
                continue

            correlations[position] = self._compute_stats(real_copy, syn_copy)

        avg_abs_distance, avg_norm_dist = self._calculate_avg_per_matrix_distance(correlations)
        correlations["avg_per_position_matrix_distance"] =avg_abs_distance
        correlations["avg_per_position_norm_matrix_distance"] = avg_norm_dist
        return correlations

    def name(self) -> str:
        return CorrelationMetricName

    def category(self) -> str:
        return Category.STATISTICAL.value

    def _compute_stats(self, X_real, X_syn: pd.DataFrame, store_matrices: bool = False) -> dict:
        # build correlation matrix for real and synth
        real_corr_matrix = self._corr_matrix(X_real)
        synth_corr_matrix = self._corr_matrix(X_syn)

        # calculate the difference between them
        matrix_distance = self._matrix_difference(real_corr_matrix, synth_corr_matrix)
        cols = X_real.columns.tolist()
        column_corr_deviations = self._calc_column_deviation(real_corr_matrix, synth_corr_matrix, cols)
        max_distance = self._calc_max_distance(real_corr_matrix)
        if store_matrices:
            print(f"max distance: {max_distance}")
            df_hash = joblib.hash(X_syn)
            print("storing matrices for correlation with hash: ", df_hash)
            filename_syn = f"temp_corr_{df_hash}_syn.json"
            filename_real = "temp_corr_real.json"
            temp_path_syn = os.path.join(tempfile.gettempdir(), filename_syn)
            temp_path_real = os.path.join(tempfile.gettempdir(), filename_real)
            synth_corr_matrix.to_json(temp_path_syn, orient="records")
            real_corr_matrix.to_json(temp_path_real, orient="records")

        return {
            "corr_matrix_distance": matrix_distance,
            "normalized_distance": matrix_distance/max_distance,
            "column_deviations": column_corr_deviations
        }

    def _corr_matrix(self, df: pd.DataFrame):
        cols = df.columns
        #create nxn matrix with identity set to 1
        corr_matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols[:i]): # we only want to iterate over lower triangle
                a = df[col1]
                b = df[col2]

                a_type = column_to_type[col1]
                b_type = column_to_type[col2]

                corr_val = float(0)
                if is_numerical(a_type) and is_numerical(b_type):
                    corr_val = self._pearson(a, b)

                elif is_numerical(a_type) and is_categorical_or_ordinal(b_type):
                    corr_val = self._eta_squared(b, a)

                elif is_categorical_or_ordinal(a_type) and is_numerical(b_type):
                    corr_val = self._eta_squared(a, b)

                elif is_categorical_or_ordinal(a_type) and is_categorical_or_ordinal(b_type):
                    corr_val = self._cramers_v(a, b)

                # set upper and lower triangle
                corr_matrix.loc[col1, col2] = corr_val
                corr_matrix.loc[col2, col1] = corr_val

        return corr_matrix.astype(float)

    def _matrix_difference(self, real, synth) -> float:
        return absolute_l1_distance(real, synth) # can only use this when we have normalized pearson in [0,1] cuz it assumes positive values only

    def _calc_column_deviation(self, real_corr_matrix, synth_corr_matrix, columns) -> dict:
        real_matrix = np.array(real_corr_matrix)
        synthetic_matrix = np.array(synth_corr_matrix)

        deviations = {}

        # go thru all pairs of features
        for i in range(real_matrix.shape[0]):
            for j in range(real_matrix.shape[1]):
                key = f"{columns[i]},{columns[j]}"

                real_val = real_matrix[i, j]
                synth_val = synthetic_matrix[i, j]

                if real_val == 0: #dont divide by 0
                    deviations[key] = 0.0
                else:
                    deviations[key] = (abs(real_val - synth_val) / real_val)

        return deviations

    def _pearson(self, a,b: pd.Series) -> float:
        #shift pearson
        corr_val = a.corr(b, method="pearson")
        if pd.isnull(corr_val):
            return 0 # one value is constant, therefore we cant have correlation

        normalized_val = (corr_val + 1)/2 #so we are in the range of [0,1] instead of [-1,1] otherwise it would have more weight potentially than the others
        return normalized_val

    def _eta_squared(self, categorical_series, numerical_series: pd.Series) -> float:
        categories = pd.Categorical(categorical_series)
        category_groups = [numerical_series[categories == cat] for cat in categories.categories]

        overall_mean = np.mean(numerical_series)
        numerator = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in category_groups)
        denominator = sum((numerical_series - overall_mean) ** 2)

        return numerator / denominator if denominator > 0 else 0

    def _cramers_v(self, a, b: pd.Series) -> float:
        a_codes = pd.Categorical(a).codes
        b_codes = pd.Categorical(b).codes

        confusion_matrix = pd.crosstab(a_codes, b_codes)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        cramers_v_value = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        return cramers_v_value

    def _calculate_avg_per_matrix_distance(self, correlations) -> (float, float):
        distance_sum = 0
        norm_distance_sum = 0
        positions = 0
        for k,v in correlations.items():
            if k != "dataset":
                distance_sum += v["corr_matrix_distance"]
                norm_distance_sum += v["normalized_distance"]
                positions += 1

        avg_distance = distance_sum/positions
        avg_norm_distance = norm_distance_sum / positions
        return avg_distance, avg_norm_distance

    def _calc_max_distance(self, real_matrix: pd.DataFrame) -> float:
        real_matrix = np.array(real_matrix)
        return np.sum(np.maximum(real_matrix, 1 - real_matrix))
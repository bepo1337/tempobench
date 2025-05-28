import numpy as np
import pandas as pd

from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    columns_with_ids
from benchmark.metrics import BenchmarkMetric
from scipy.stats import entropy

ElementaryStatisticsMetricName = "ElementaryStatistics"

class ElementaryStatistics(BenchmarkMetric):
    """Computes elementary statistical measures such as deviation from mean/median/std deviation for numerical values and
     for categorical values the mode, entropy and the deviation of the entropy is calculated"""

    def compute(self, X_real, X_syn) -> dict:
        dataset_wide_results = self._compute_for_dataframes(X_real, X_syn)
        summary = self._summarize_results(dataset_wide_results)
        result = {"dataset": {"summary": summary, "details": dataset_wide_results}}
        for i in range(2010, 2020):
            X_real_filtered = X_real[X_real[SEASON_ID] == i]
            X_synthetic_filtered = X_syn[X_syn[SEASON_ID] == i]
            season_results = self._compute_for_dataframes(X_real_filtered, X_synthetic_filtered)
            result[i] = season_results

        return result

    def name(self) -> str:
        return ElementaryStatisticsMetricName

    def category(self) -> str:
        return Category.STATISTICAL.value

    def _compute_for_dataframes(self, X_real, X_syn) -> pd.DataFrame:
        result = {}
        for column_name, real_series in X_real.items():
            if column_name in columns_with_ids:
                continue

            column_type = column_to_type[column_name]
            synth_series = X_syn[column_name]
            if column_type == ColumnType.DISCRETE or column_type == ColumnType.CONTINUOUS:
                num_stats = self._numerical_statistics(real_series, synth_series)
                result[column_name] = num_stats
            elif column_type == ColumnType.CATEGORICAL:
                cat_stats = self._categorical_statistics(real_series, synth_series)
                result[column_name] = cat_stats

        return result

    def _numerical_statistics(self, real_series, synth_series) -> dict:
        mean = abs(real_series.mean() - synth_series.mean()) / abs(real_series.mean())
        if real_series.median() == synth_series.median():
            median = 0
        else:
            if real_series.median() == 0:
                median = "inf" #todo inf fine?
            else:
                median = abs(real_series.median() - synth_series.median()) / abs(real_series.median())

        std = abs(real_series.std() - synth_series.std()) / abs(real_series.std())

        def handle_cast_to_int_if_int64(num):
            if isinstance(num, float):
                return num

            if isinstance(num, int):
                return num

            if num.dtype == 'int64':
                return int(num)
            return num

        min_real = handle_cast_to_int_if_int64(real_series.min())
        min_syn = handle_cast_to_int_if_int64(synth_series.min())
        max_real = handle_cast_to_int_if_int64(real_series.max())
        max_syn = handle_cast_to_int_if_int64(synth_series.max())

        return {
            "mean_deviation": mean,
            "median_deviation": median,
            "std_deviation": std,
            "min": {
                "real": min_real,
                "synth": min_syn,
            },
            "max": {
                "real": max_real,
                "synth": max_syn,
            }
        }

    def _categorical_statistics(self, real_series, synth_series: pd.Series) -> dict:
        real_mode = real_series.mode().tolist()
        synth_mode = synth_series.mode().tolist()
        normalized_real_series = real_series.value_counts(normalize=True)
        normalized_synth_series = synth_series.value_counts(normalize=True)

        real_entropy = entropy(normalized_real_series, base=2)
        synth_entropy = entropy(normalized_synth_series, base=2)

        max_real_entropy = np.log2(len(normalized_real_series))

        if len(normalized_synth_series) == 0:
            max_synth_entropy = 0
        else:
            max_synth_entropy = np.log2(len(normalized_synth_series))

        # normalizing the entropy makes the features comarable even though they have different cardinality. so its always in [0,1]
        real_entropy_normalized = real_entropy / max_real_entropy if max_real_entropy > 0 else 0
        synth_entropy_normalized = synth_entropy / max_synth_entropy if max_synth_entropy > 0 else 0

        entropy_dev = None
        if real_entropy_normalized == 0:
            if synth_entropy_normalized == 0:
                entropy_dev = 0
            else:
                entropy_dev = "inf"
        else:
            entropy_dev = abs(real_entropy_normalized - synth_entropy_normalized) / abs(real_entropy_normalized)
        return {"mode": {
            "real": real_mode,
            "synth": synth_mode
        },
            "entropy": {
                "deviation": entropy_dev,
                "real": real_entropy_normalized,
                "synth": synth_entropy_normalized
            }
        }


    def _summarize_results(self, results: dict) -> dict:
        cat_summary = self._summarize_categorical(results)
        num_summary = self._summarize_numerical(results)
        return {
            "categorical": cat_summary,
            "numerical": num_summary
        }

    def _summarize_categorical(self, results: dict):
        mode_matches = 0
        deviation_sum = 0
        cat_values = 0
        for col_name, val in results.items():
            if "mode" in val:
                cat_values += 1
                if val["mode"]["real"] == val["mode"]["synth"]:
                    mode_matches += 1

                deviation_sum += val["entropy"]["deviation"]


        avg_deviation = deviation_sum / cat_values
        return {
            "proportion_mode_matches": mode_matches/cat_values,
            "average_entropy_deviation": avg_deviation
        }

    def _summarize_numerical(self, results: dict):
        std_deviation_sum = 0
        median_deviation_sum = 0
        mean_deviation_sum = 0
        min_deviation_sum = 0
        max_deviation_sum = 0
        num_values = 0
        for col_name, val in results.items():
            if "mean_deviation" in val:
                num_values += 1
                mean_deviation_sum += val["mean_deviation"]
                median_deviation_sum += val["median_deviation"] if val["median_deviation"] != "inf" else 0
                std_deviation_sum += val["std_deviation"]
                real_min = val["min"]["real"]
                syn_min = val["min"]["synth"]
                min_deviation = abs(syn_min - real_min) / real_min if real_min > 0 else 0
                min_deviation_sum += min_deviation
                real_max = val["max"]["real"]
                syn_max = val["max"]["synth"]
                max_deviation = abs(syn_max - real_max) / real_max if real_max > 0 else 0
                max_deviation_sum += max_deviation


        return {
            "avg_mean_deviation": mean_deviation_sum / num_values,
            "avg_median_deviation": median_deviation_sum / num_values,
            "avg_std_deviation": std_deviation_sum/num_values,
            "avg_min_deviation": min_deviation_sum/num_values,
            "avg_max_deviation": max_deviation_sum/num_values
        }
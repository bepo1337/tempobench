import pandas
import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import List

from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, LAST_TRANSFER_FEE, columns_with_ids, performance_data
from benchmark.utils import MARKET_VALUE, column_to_type, ColumnType

JensenShannonMetricName = "JensenShannon"
class JensenShannon(BenchmarkMetric):
    """Computes the Jensen-Shannon divergence. 0 = complete alignment. 1 = complete misalignment"""
    def compute(self, X_real, X_syn) -> dict:
        js_divergence_scores = {}
        for column_name, real_series in X_real.items():
            if column_name in columns_with_ids: # Skip columns that we don't want to calculate divergence for
                continue

            divergence = 0
            try:
                column_type = column_to_type[column_name]
                synth_series = X_syn[column_name]
                if (column_type in [ColumnType.DISCRETE, ColumnType.CATEGORICAL, ColumnType.ORDINAL]
                        and column_name not in performance_data):
                    divergence = self._discrete_js(real_series, synth_series)
                elif column_type == ColumnType.CONTINUOUS or column_name in performance_data:
                    divergence = self._quantile_binning_js(real_series, synth_series)
                else:
                    continue
            except ValueError as e:
                print(f"WARN: couldnt compute divergence for column '{column_name}': {str(e)}")

            js_divergence_scores[column_name] = divergence

        avg = sum(js_divergence_scores.values()) / len(js_divergence_scores)
        highest, lowest = self._highest_and_lowest_scores(js_divergence_scores)
        return {"avg": avg,
                "highest_divergence": highest,
                "lowest_divergence": lowest,
                "features:": js_divergence_scores}

    def name(self) -> str:
        return JensenShannonMetricName

    def category(self) -> str:
        return Category.STATISTICAL.value

    def _discrete_js(self, real_series, synth_series: pandas.Series) -> float:
        # if synth series contains only Null values
        if synth_series.nunique() == 0:
            return 1.0

        real_counts = real_series.value_counts(normalize=True)
        synth_counts = synth_series.value_counts(normalize=True)

        # union of all categories (to align both distributions)
        all_categories = set(real_counts.index).union(set(synth_counts.index))

        # add zero probability for missing categories in each
        real_probabilities = np.array([real_counts.get(cat, 0) for cat in all_categories])
        synth_probabilities = np.array([synth_counts.get(cat, 0) for cat in all_categories])

        # paper from wiki states that using base 2 results in us having a value in [0,1]  https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        return jensenshannon(real_probabilities, synth_probabilities, base=2)


    def _quantile_binning_js(self, real_series: pandas.Series, synth_series: pandas.Series) -> float:
        # use 2 percent quantiles
        quantiles = np.unique(np.percentile(real_series, np.arange(0, 101, 2)), axis=0) # deduplicate for height cuz there are so few values
        #we wanna have values that are smaller or larger than in our data to be reprsented in different bins
        bin_edges = np.concatenate(([-np.inf], quantiles, [np.inf]))

        real_hist, _ = np.histogram(real_series, bins=bin_edges, density=False)
        real_probabilities = real_hist / real_hist.sum()

        synth_hist, _ = np.histogram(synth_series, bins=bin_edges, density=False)
        synth_probabilities = synth_hist / synth_hist.sum()

        return jensenshannon(real_probabilities, synth_probabilities, base=2)

    def _highest_and_lowest_scores(self, result: dict) -> (List[str], List[str]):
        sorted_items = sorted(result.items(), key=lambda item: item[1])
        lowest = [k for k, _ in sorted_items[:3]]
        highest = [k for k, _ in sorted_items[-3:][::-1]]
        return highest, lowest


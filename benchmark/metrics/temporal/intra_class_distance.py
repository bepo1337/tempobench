from typing import List

import pandas as pd

from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    combined_dtw
from benchmark.metrics import BenchmarkMetric
from benchmark.utils.normalize import normalize_and_encode_dfs_in_place
from benchmark.utils.pandas_util import get_players_as_list_of_df

IntraClassDistanceMetricName = "IntraClassDistance"

class IntraClassDistance(BenchmarkMetric):
    """Computes the intra class distance and computes the deviation from the real data set."""

    def compute(self, X_real, X_syn) -> dict:
        real_copy = X_real.copy()
        syn_copy = X_syn.copy()

        normalize_and_encode_dfs_in_place(real_copy, syn_copy)
        n = 200
        real_samples = get_players_as_list_of_df(real_copy, n)
        syn_samples = get_players_as_list_of_df(syn_copy, n)

        real_icd = self._icd(real_samples)
        syn_icd = self._icd(syn_samples)

        deviation = abs(syn_icd - real_icd) / real_icd

        return {
            "real_icd": real_icd,
            "synth_icd": syn_icd,
            "deviation": deviation
        }

    def name(self) -> str:
        return IntraClassDistanceMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

    def _icd(self, samples: List[pd.DataFrame]) -> float:
        sum_distances = 0
        for sample in samples:
            total_sample_dist = 0
            # compute distance between this sample and all others
            for other_sample in samples:
                if sample.equals(other_sample): # dont compute with self
                    continue

                distance, _ = combined_dtw(sample, other_sample)
                sum_distances += distance

        icd = sum_distances / (len(samples)**2)
        return icd

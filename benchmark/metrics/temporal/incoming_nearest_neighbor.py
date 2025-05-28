from typing import List

import pandas as pd

from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    combined_dtw
from benchmark.metrics import BenchmarkMetric
from benchmark.utils.normalize import normalize_and_encode_dfs_in_place
from benchmark.utils.pandas_util import get_players_as_list_of_df

IncomingNearestNeighborMetricName = "IncomingNearestNeighbor"
starting_distance = float(9999999)

class IncomingNearestNeighbor(BenchmarkMetric):
    """Computes the average incoming nearest neighbor distance."""

    def compute(self, X_real, X_syn) -> dict:
        real_copy = X_real.copy()
        syn_copy = X_syn.copy()

        # normalize and encode the dataframes
        normalize_and_encode_dfs_in_place(real_copy, syn_copy)

        # get n samples from synth and compute average distance to whole real sequences
        real_samples = get_players_as_list_of_df(real_copy)
        syn_samples = get_players_as_list_of_df(syn_copy, 50) # only do for subset because of computational time
        average_distance = self._compute_avg_incoming_nearest_neighbor(real_samples, syn_samples)

        return {
            "average_distance": average_distance,
        }

    def name(self) -> str:
        return IncomingNearestNeighborMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

    def _compute_avg_incoming_nearest_neighbor(self, real_samples, synth_samples: List[pd.DataFrame]) -> float:
        total_min_distance = 0
        for synth_sample in synth_samples:
            min_distance = starting_distance
            for real_sample in real_samples:
                distance, _ = combined_dtw(synth_sample, real_sample)
                if distance < min_distance:
                    min_distance = distance

            total_min_distance += min_distance

        average_distance = total_min_distance / len(synth_samples)
        return average_distance

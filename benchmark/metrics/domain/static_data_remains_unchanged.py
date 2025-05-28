import pandas as pd
from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, LAST_NAME, PSEUDONYM, \
    HEIGHT, DATE_OF_BIRTH, FIRST_NAME, CITIZENSHIP, FOOT, POSITION, PLAYER_ID
from benchmark.utils.pandas_util import grouped_by_player_id_sorted_by_validity_start

StaticDataRemainsUnchangedMetricName = "StaticDataRemainsUnchanged"

static_columns = [
    FIRST_NAME,
    LAST_NAME,
    PSEUDONYM,
    HEIGHT,
    DATE_OF_BIRTH,
    FOOT,
    POSITION,
    CITIZENSHIP
]

class StaticDataRemainsUnchanged(BenchmarkMetric):
    """ Computes the proportion of entities where the static data doesn't remain the same over the whole data set.
        0 = no entites violate the constraint. 1 = all entities violate the constraint.
        Details show the columns that are affected the most"""

    def compute(self, X_real, X_syn) -> dict:
        player_ids_violated_set = set()
        for player_id, player_rows in X_syn.groupby(PLAYER_ID):
            for col in static_columns:
                unique_values = player_rows[col].nunique()
                if unique_values > 1:
                    player_ids_violated_set.add(player_id)
                    break

        total_players = X_syn[PLAYER_ID].nunique()
        entity_violation_proportion = len(player_ids_violated_set) / total_players

        return {"entities_violated": entity_violation_proportion}

    def name(self) -> str:
        return StaticDataRemainsUnchangedMetricName

    def category(self) -> str:
        return Category.DOMAIN.value
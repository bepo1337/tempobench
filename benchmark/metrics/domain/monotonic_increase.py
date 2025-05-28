import pandas as pd
from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, LEAGUE_PLAYED_MATCHES, LEAGUE_MINUTES_PLAYED, \
    LEAGUE_GOALS, INTERNATIONAL_PLAYED_MATCHES, INTERNATIONAL_MINUTES_PLAYED, INTERNATIONAL_GOALS, AGE, PLAYER_ID
from benchmark.utils.pandas_util import grouped_by_player_id_sorted_by_validity_start

MonotonicIncreaseMetricName = "MonotonicIncrease"

monotonically_increasing_columns = [
    AGE,
    LEAGUE_PLAYED_MATCHES,
    LEAGUE_MINUTES_PLAYED,
    LEAGUE_GOALS,
    INTERNATIONAL_PLAYED_MATCHES,
    INTERNATIONAL_MINUTES_PLAYED,
    INTERNATIONAL_GOALS
]

class MonotonicIncrease(BenchmarkMetric):
    """Computes the proportion of rows and entities that don't increase monotonically for specified columns.
     rows: 0 = No rows violate the constraint. 1 = All possible rows violate the constraint.
     entities: 0 = No entities violate the constraint. 1 = All entities violate the constraint.
     details: Provides the proportion of rows for each column that violate the constraint. """
    def compute(self, X_real, X_syn) -> dict:
        # init all columns to 0
        column_violations = {key: 0 for key in monotonically_increasing_columns}
        rows_violated = 0
        player_ids_violated_set = set()

        player_sorted_groups = grouped_by_player_id_sorted_by_validity_start(X_syn)
        for rows in player_sorted_groups:
            for i, row in rows.iterrows():
                if i == 0:
                    continue

                prev_row = rows.iloc[i - 1]
                row_violations, row_is_clean = self._calculate_violations(row, prev_row)
                # add possible violations to the total column violations by iterating over all columns
                column_violations = {key: column_violations[key] + row_violations[key] for key in column_violations}
                if not row_is_clean:
                    rows_violated += 1
                    player_ids_violated_set.add(row[PLAYER_ID])


        total_players = X_syn[PLAYER_ID].nunique()
        entity_violation_proportion = len(player_ids_violated_set) / total_players
        possible_row_violations = len(X_syn) - total_players

        # possible row violations can be 0 if no actual sequences are present
        if possible_row_violations == 0:
            rows_violated_proportion = 0
            column_violations_proportions = {key: 0 for key in column_violations}
        else:
            rows_violated_proportion = rows_violated / possible_row_violations
            column_violations_proportions = {key: column_violations[key] / possible_row_violations for key in column_violations}
        return {"entities_violated": entity_violation_proportion,
                "rows_violated": rows_violated_proportion,
                "details": column_violations_proportions}

    def name(self) -> str:
        return MonotonicIncreaseMetricName

    def category(self) -> str:
        return Category.DOMAIN.value

    def _calculate_violations(self, row, prev_row: pd.Series) -> (dict, bool):
        row_violations = {key: 0 for key in monotonically_increasing_columns}
        row_is_clean = True
        for column in monotonically_increasing_columns:
            if row[column] < prev_row[column]:
                row_violations[column] += 1
                row_is_clean = False

        return row_violations, row_is_clean

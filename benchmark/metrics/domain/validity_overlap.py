from datetime import datetime

from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, VALIDITY_START, PLAYER_ID
from benchmark.utils.pandas_util import grouped_by_player_id_sorted_by_validity_start
from benchmark.utils.temporal import date_string_to_datetime

ValidityOverlapMetricName = "ValidityOverlap"

class ValidityOverlap(BenchmarkMetric):
    """Computes the proportion of rows where the validity start is equal or before the previous validity end.
     0 = No rows violate the validity constraint. 1 = All (possible) rows violate the constraint."""
    def compute(self, X_real, X_syn) -> dict:
        violations = 0

        player_sorted_groups = grouped_by_player_id_sorted_by_validity_start(X_syn)
        for rows in player_sorted_groups:
            for i, row in rows.iterrows():
                if i == 0:
                    continue

                row_start = row[VALIDITY_START]
                prev_row = rows.iloc[i - 1]
                prev_row_end = prev_row.validity_end
                if self.start_before_or_equal_to_prev_end(row_start, prev_row_end):
                    violations += 1

        possible_violations = len(X_syn) - X_syn[PLAYER_ID].nunique()
        if possible_violations == 0:
            proportion = "NO_SEQUENCES"
        else:
            proportion = violations / possible_violations
        return {"proportion": proportion}

    def name(self) -> str:
        return ValidityOverlapMetricName

    def category(self) -> str:
        return Category.DOMAIN.value

    def start_before_or_equal_to_prev_end(self, row_start, prev_row_end) -> bool:
        curr_start = date_string_to_datetime(row_start)
        prev_end = date_string_to_datetime(prev_row_end)

        if curr_start <= prev_end:
            return True

        return False

import datetime

from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, VALIDITY_START, VALIDITY_END
from benchmark.utils.temporal import date_string_to_datetime

ValiditiyMaxTimeMetricName = "ValidityMaxTime"
max_days = (datetime.datetime(2020, 6, 30) - datetime.datetime(2010, 7, 1)).days
class ValidityMaxTime(BenchmarkMetric):
    """Computes the proportion of rows that violate the max. 10 year validity contstraint.
     0 = No rows violate the validity constraint. 1 = All rows violate the constraint."""
    def compute(self, X_real, X_syn) -> dict:
        violations = 0
        # for each sample
        for _, row in X_syn.iterrows():
            start = row[VALIDITY_START]
            end = row[VALIDITY_END]
            if self.is_greater_than_ten_years(start, end):
                violations += 1

        proportion = violations / len(X_syn)
        return {"proportion": proportion}

    def name(self) -> str:
        return ValiditiyMaxTimeMetricName

    def category(self) -> str:
        return Category.DOMAIN.value

    def is_greater_than_ten_years(self, date1: str, date2: str) -> bool:
        d1 = date_string_to_datetime(date1)
        d2 = date_string_to_datetime(date2)

        return abs((d2 - d1).days) > max_days
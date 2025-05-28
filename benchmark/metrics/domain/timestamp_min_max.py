import datetime

from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, VALIDITY_START, VALIDITY_END
from benchmark.utils.temporal import date_string_to_datetime

TimestampMinMaxMetricName = "TimestampMinMax"

class TimestampMinMax(BenchmarkMetric):
    """Computes the proportion of rows that are outside the timestamp interval [01.07.2010, 30.06.2020].
     0 = No rows violate the constraint. 1 = All rows violate the constraint."""

    min_timestamp = datetime.datetime(2010, 7, 1)
    max_timestamp = datetime.datetime(2020, 6, 30)

    def compute(self, X_real, X_syn) -> dict:
        violations = 0
        for _, row in X_syn.iterrows():
            start = row[VALIDITY_START]
            end = row[VALIDITY_END]
            if self.timestamp_is_outside_interval(start) or self.timestamp_is_outside_interval(end):
                violations += 1

        proportion = violations / len(X_syn)
        return {"proportion": proportion}

    def name(self) -> str:
        return TimestampMinMaxMetricName

    def category(self) -> str:
        return Category.DOMAIN.value

    def timestamp_is_outside_interval(self, date_str: str) -> bool:
        date = date_string_to_datetime(date_str)
        return self.min_timestamp > date or date > self.max_timestamp
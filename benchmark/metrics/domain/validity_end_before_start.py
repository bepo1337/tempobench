from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, VALIDITY_START, VALIDITY_END
from benchmark.utils.temporal import date_string_to_datetime

ValidityEndBeforeStartMetricName = "ValidityEndBeforeStart"

class ValidityEndBeforeStart(BenchmarkMetric):
    """Computes the proportion of rows where the validity end is before the start.
     0 = No rows violate the validity constraint. 1 = All rows violate the constraint."""
    def compute(self, X_real, X_syn) -> dict:
        violations = 0
        for _, row in X_syn.iterrows():
            start = row[VALIDITY_START]
            end = row[VALIDITY_END]
            if self.end_before_start(start, end):
                violations += 1

        proportion = violations / len(X_syn)
        return {"proportion": proportion}

    def name(self) -> str:
        return ValidityEndBeforeStartMetricName

    def category(self) -> str:
        return Category.DOMAIN.value

    def end_before_start(self, start, end: str) -> bool:
        start = date_string_to_datetime(start)
        end = date_string_to_datetime(end)
        if end < start:
            return True

        return False
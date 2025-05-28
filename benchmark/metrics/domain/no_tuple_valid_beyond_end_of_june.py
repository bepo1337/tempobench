from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, VALIDITY_END, VALIDITY_START

NoTupleValidBeyondEndOfJuneMetricName = "NoTupleValidBeyondEndOfJune"

#Deprecated.This metric is no longer used.
class NoTupleValidBeyondEndOfJune(BenchmarkMetric):
    """ Deprecated. Computes the proportion of entities where the validity end is after the 30.06. This is not possible because on the 01.07 every player gets a new tuple.
        0 = no rows violate the constraint. 1 = all rows violate the constraint."""

    def compute(self, _, X_syn) -> dict:
        violations = 0
        for _, row in X_syn.iterrows():
            validity_start = row[VALIDITY_START]
            validity_end = row[VALIDITY_END]
            month_start = validity_start[5:7]
            month_start_int = int(month_start)
            if month_start_int >= 7:
                continue

            month_end_int = int(validity_end[5:7])
            if month_end_int >= 7:
                violations += 1

        total_rows = X_syn.shape[0]
        return {"proportion": violations / total_rows}

    def name(self) -> str:
        return NoTupleValidBeyondEndOfJuneMetricName

    def category(self) -> str:
        return Category.DOMAIN.value
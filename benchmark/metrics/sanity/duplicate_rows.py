from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category

DuplicateRowsMetricName = "DuplicateRows"

class DuplicateRows(BenchmarkMetric):
    """Computes the proportion of common rows in the synthetic dataset that are in the real data set.
     0 = No common rows. 1 = All rows in the synthetic dataset exist in the real dataset."""
    def compute(self, X_real, X_syn) -> dict:
        common_rows = X_real.merge(X_syn, how="inner")
        num_common_rows = len(common_rows)
        total_rows = X_syn.shape[0]
        return {"proportion": num_common_rows / total_rows}

    def name(self) -> str:
        return DuplicateRowsMetricName

    def category(self) -> str:
        return Category.SANITY.value

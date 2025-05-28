from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category

DataTypeMismatchMetricName = "DataTypeMismatch"

class DataTypeMismatch(BenchmarkMetric):
    """Compares the data types in the real and synthetic data.
     0 = complete alignment in datatypes. 1 = complete mismatch in datatypes"""
    def compute(self, X_real, X_syn) -> dict:
        diff_col_names = []
        different_column_types = 0
        for column in X_real.columns.values.tolist():
            if X_real.dtypes[column] != X_syn[column].dtype:
                different_column_types += 1
                diff_col_names.append(column)

        total_no_of_cols = X_real.shape[1]
        return {"proportion": different_column_types / total_no_of_cols, "columns_violated": diff_col_names}

    def name(self) -> str:
        return DataTypeMismatchMetricName

    def category(self) -> str:
        return Category.SANITY.value
import unittest
import pandas as pd

from benchmark.utils import VALIDITY_START, VALIDITY_END
from .validity_max_time import ValidityMaxTime

class TestValidityMaxTime(unittest.TestCase):

    def setUp(self):
        self.metric = ValidityMaxTime()

    def test_is_greater_than_one_year(self):
        self.assertTrue(self.metric.is_greater_than_ten_years("2023-02-13", "2025-02-14"))
        self.assertFalse(self.metric.is_greater_than_ten_years("2024-02-13", "2025-02-12"))
        self.assertTrue(self.metric.is_greater_than_ten_years("2020-01-01", "2022-01-02"))
        self.assertFalse(self.metric.is_greater_than_ten_years("2023-05-10", "2024-05-09"))

    def test_compute_all_valid(self):
        X_syn = pd.DataFrame({
            VALIDITY_START: ["2024-02-13", "2023-01-01", "2022-06-01"],
            VALIDITY_END: ["2025-02-12", "2023-12-31", "2023-05-30"]
        })
        result = self.metric.compute(None, X_syn)
        self.assertEqual(result["proportion"], 0.0)

    def test_compute_all_invalid(self):
        X_syn = pd.DataFrame({
            VALIDITY_START: ["2020-01-01", "2019-05-10", "2021-02-13"],
            VALIDITY_END: ["2022-01-02", "2021-06-15", "2023-04-14"]
        })
        result = self.metric.compute(None, X_syn)
        self.assertEqual(result["proportion"], 1.0)

    def test_compute_mixed_validity(self):
        X_syn = pd.DataFrame({
            VALIDITY_START: ["2024-02-13", "2020-01-01", "2023-06-01"],
            VALIDITY_END: ["2025-02-12", "2022-01-02", "2024-05-30"]
        })
        result = self.metric.compute(None, X_syn)
        self.assertEqual(result["proportion"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
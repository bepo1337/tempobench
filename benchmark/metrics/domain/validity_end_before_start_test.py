import unittest
import pandas as pd
from datetime import datetime

from benchmark.utils import VALIDITY_START, VALIDITY_END
from benchmark.metrics.domain.validity_end_before_start import ValidityEndBeforeStart

def date_string_to_datetime(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


class TestValidityEndBeforeStart(unittest.TestCase):

    def setUp(self):
        self.metric = ValidityEndBeforeStart()

    def test_no_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-01-01", "2023-02-01", "2023-03-01"],
            VALIDITY_END: ["2023-01-02", "2023-02-05", "2023-03-10"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 0.0)

    def test_all_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-02-10", "2023-03-15", "2023-04-20"],
            VALIDITY_END: ["2023-02-05", "2023-03-10", "2023-04-15"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 1.0)

    def test_partial_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-01-01", "2023-02-10", "2023-03-15"],
            VALIDITY_END: ["2023-01-02", "2023-02-05", "2023-03-10"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 2 / 3)

    def test_edge_case_same_dates(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-01-01"],
            VALIDITY_END: ["2023-01-01"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 0.0)

    def test_end_before_start_function(self):
        self.assertTrue(self.metric.end_before_start("2023-03-05", "2023-03-01"))
        self.assertFalse(self.metric.end_before_start("2023-03-01", "2023-03-05"))
        self.assertFalse(self.metric.end_before_start("2023-03-01", "2023-03-01"))


if __name__ == "__main__":
    unittest.main()
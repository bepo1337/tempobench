import unittest
import pandas as pd

from benchmark.utils import VALIDITY_START, VALIDITY_END
from .no_tuple_valid_beyond_end_of_june import NoTupleValidBeyondEndOfJune


class TestNoTupleValidBeyondEndOfJune(unittest.TestCase):

    def setUp(self):
        self.metric = NoTupleValidBeyondEndOfJune()

    def test_no_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-06-01", "2022-05-15", "2021-04-10"],
            VALIDITY_END: ["2023-06-30", "2022-05-30", "2021-06-29"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 0.0)

    def test_all_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-06-01", "2022-06-15", "2021-05-10"],
            VALIDITY_END: ["2023-07-01", "2022-07-05", "2021-08-10"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 1.0)

    def test_partial_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-06-01", "2022-06-15", "2021-04-10"],
            VALIDITY_END: ["2023-07-01", "2022-06-30", "2021-06-29"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 1/3)

    def test_edge_case_june_30(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2023-06-01"],
            VALIDITY_END: ["2023-06-30"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 0.0)


if __name__ == "__main__":
    unittest.main()
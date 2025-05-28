import unittest
import pandas as pd
from .duplicate_rows import DuplicateRows

class TestCommonRows(unittest.TestCase):

    def setUp(self):
        self.metric = DuplicateRows()

    def test_no_common_rows(self):
        X_real = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"]
        })
        X_syn = pd.DataFrame({
            "A": [4, 5, 6],
            "B": ["d", "e", "f"]
        })
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["proportion"], 0.0)

    def test_all_common_rows(self):
        X_real = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"]
        })
        X_syn = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"]
        })
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["proportion"], 1.0)

    def test_partial_common_rows(self):
        X_real = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"]
        })
        X_syn = pd.DataFrame({
            "A": [2, 3, 4],
            "B": ["b", "c", "d"]
        })
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["proportion"], 2/3)


if __name__ == "__main__":
    unittest.main()
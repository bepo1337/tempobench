import unittest
import pandas as pd
from .datatype_mismatch import DataTypeMismatch

class TestDataTypeMismatch(unittest.TestCase):

    def setUp(self):
        self.metric = DataTypeMismatch()

    def test_no_mismatch(self):
        X_real = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
            "col3": ["a", "b", "c"]
        })
        X_syn = pd.DataFrame({
            "col1": [4, 5, 6],
            "col2": [4.4, 5.5, 6.6],
            "col3": ["d", "e", "f"]
        })
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["proportion"], 0.0)

    def test_complete_mismatch(self):
        X_real = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
            "col3": ["a", "b", "c"]
        })
        X_syn = pd.DataFrame({
            "col1": ["x", "y", "z"],
            "col2": [True, False, True],
            "col3": [100, 200, 300]
        })
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["proportion"], 1.0)

    def test_partial_mismatch(self):
        X_real = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [1.1, 2.2, 3.3],
            "col3": ["a", "b", "c"]
        })
        X_syn = pd.DataFrame({
            "col1": [4.5, 5.6, 6.7],  # Mismatch (int -> float)
            "col2": [1.1, 2.2, 3.3],  # Match
            "col3": [10, 20, 30]  # Mismatch (str -> int)
        })
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["proportion"], 2/3)

if __name__ == "__main__":
    unittest.main()

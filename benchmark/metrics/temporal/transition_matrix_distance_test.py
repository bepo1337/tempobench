import unittest
import pandas as pd

from benchmark import PLAYER_ID
from benchmark.utils import CITIZENSHIP, VALIDITY_START
from .transition_matrix_distance import TransitionMatrixDistance

class TestTransitionMatrixDistance(unittest.TestCase):

    def setUp(self):
        self.metric = TransitionMatrixDistance()

    def test_identical_data(self):
        X_real = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"],
            CITIZENSHIP: ["A", "B", "C", "D"]
        })

        X_syn = X_real.copy() # ident copy

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["average_distance"], 0.0)

    def test_completely_different_data(self):
        X_real = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"],
            CITIZENSHIP: ["A", "B", "C", "D"]
        })

        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2023-01-01", "2023-01-02", "2023-02-01", "2023-02-02"],
            CITIZENSHIP: ["X", "Y", "Z", "W"]
        })

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["average_distance"], 1)

    def test_somewhat_different_data(self):
        X_real = pd.DataFrame({
            PLAYER_ID: [1, 1, 1, 1, 2, 2, 2],
            VALIDITY_START: ["2023-01-01", "2023-01-02", "2023-01-03","2023-01-04","2023-02-01", "2023-02-02","2023-02-03"],
            CITIZENSHIP: ["A", "B" ,"A", "A", "C", "D", "C"]
        })

        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 1, 2, 2, 2],
            VALIDITY_START: ["2023-01-01", "2023-01-02", "2023-01-03","2023-02-01", "2023-02-02","2023-02-03"],
            CITIZENSHIP: ["A", "B", "C", "C", "X", "A"]
        })

        result = self.metric.compute(X_real, X_syn)

        self.assertGreater(result["average_distance"], 0.1)
        self.assertLess(result["average_distance"], 1)

if __name__ == "__main__":
    unittest.main()
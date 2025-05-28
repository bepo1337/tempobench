import unittest
import pandas as pd
from benchmark.utils import PLAYER_ID
from .mean_length_of_sequence import MeanSequenceLength

class TestMeanSequenceLength(unittest.TestCase):

    def setUp(self):
        self.metric = MeanSequenceLength()

    def test_no_deviation(self):
        X_real = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3],
        })
        X_syn = X_real.copy()

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["deviation"], 0.0)
        self.assertEqual(result["real_mean"], 7/3)
        self.assertEqual(result["synth_mean"], 7/3)

    def test_with_deviation(self):
        X_real = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3],
        })
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 2, 2, 3, 3, 3, 3, 3],
        })
        result = self.metric.compute(X_real, X_syn)

        self.assertEqual(result["real_mean"], 7/3)
        self.assertEqual(result["synth_mean"], 8/3)

        expected_deviation = abs(7/3 - 8/3) / (7/3)
        self.assertEqual(result["deviation"], expected_deviation)
if __name__ == "__main__":
    unittest.main()

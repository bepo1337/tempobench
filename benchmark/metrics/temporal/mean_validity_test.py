import unittest
import pandas as pd
import numpy as np
from benchmark.utils import VALIDITY_START, VALIDITY_END
from .mean_validity import MeanValidity

class TestMeanValidity(unittest.TestCase):

    def setUp(self):
        self.metric = MeanValidity()

    def test_no_deviation(self):
        X_real = pd.DataFrame({
            VALIDITY_START: ["2023-01-01", "2023-03-15", "2023-06-10"],
            VALIDITY_END: ["2023-02-01", "2023-04-01", "2023-07-01"]
        })

        X_syn = X_real.copy()
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["deviation"], 0.0)

    def test_with_deviation(self):
        X_real = pd.DataFrame({
            VALIDITY_START: ["2023-01-01", "2023-03-15", "2023-06-10"],
            VALIDITY_END: ["2023-02-01", "2023-04-01", "2023-07-01"]
        })

        X_syn = pd.DataFrame({
            VALIDITY_START: ["2023-01-05", "2023-03-10", "2023-06-05"],
            VALIDITY_END: ["2023-02-10", "2023-04-05", "2023-07-05"]
        })

        result = self.metric.compute(X_real, X_syn)

        exp_real_mean = 69/3
        exp_syn_mean = 92/3
        exp_deviation = abs(exp_real_mean - exp_syn_mean) / abs(exp_real_mean)
        self.assertEqual(result["real_mean"], exp_real_mean)
        self.assertEqual(result["synth_mean"], exp_syn_mean)
        self.assertEqual(result["deviation"], exp_deviation)


if __name__ == "__main__":
    unittest.main()

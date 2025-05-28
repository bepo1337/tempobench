import unittest
import pandas as pd
import numpy as np

from .jensen_shannon import JensenShannon


class TestJensenShannonMetric(unittest.TestCase):

    def setUp(self):
        self.metric = JensenShannon()

    def test_discrete_js_identical(self):
        real_series = pd.Series(["A", "B", "A", "C", "B", "C", "C", "A"])
        synth_series = pd.Series(["A", "B", "A", "C", "B", "C", "C", "A"])
        result = self.metric._discrete_js(real_series, synth_series)
        self.assertEqual(result, 0.0)

    def test_discrete_js_different(self):
        real_series = pd.Series(["A", "B", "A", "C", "B", "C", "C", "A"])
        synth_series = pd.Series(["X", "Y", "X", "Y", "X", "Y", "X", "Y"])
        result = self.metric._discrete_js(real_series, synth_series)
        self.assertAlmostEqual(result, 1)

    def test_discrete_js_missing_category(self):
        real_series = pd.Series(["A", "B", "A", "C", "B", "C", "C", "A"])
        synth_series = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B"])
        result = self.metric._discrete_js(real_series, synth_series)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.8)


    def test_continuous_js_comlete_divergence_no_overlap(self):
        real_series = pd.Series(range(1,101))
        synth_series = pd.Series(range(-200, -1))
        result = self.metric._quantile_binning_js(real_series, synth_series)
        self.assertAlmostEqual(result, 1)


    def test_continuous_js_identical(self):
        real_series = pd.Series(np.random.normal(50, 10, 1000))
        synth_series = real_series.copy()
        result = self.metric._quantile_binning_js(real_series, synth_series)
        self.assertAlmostEqual(result, 0.0)

    def test_continuous_js_different(self):
        real_series = pd.Series(np.random.normal(50, 10, 1000))
        synth_series = pd.Series(np.random.normal(100, 20, 1000))
        result = self.metric._quantile_binning_js(real_series, synth_series)
        self.assertGreater(result, 0.5)


if __name__ == "__main__":
    unittest.main()
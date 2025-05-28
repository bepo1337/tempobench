import unittest
import pandas as pd

from benchmark.utils import MARKET_VALUE, FOOT, SEASON_ID
from benchmark.utils import AGE
from .elementary_statistics import ElementaryStatistics

class TestElementaryStatistics(unittest.TestCase):

    def setUp(self):
        self.metric = ElementaryStatistics()

    def test_numerical_statistics(self):
        X_real = pd.DataFrame({
            SEASON_ID: ["2010", "2010", "2010", "2011", "2011"],
            MARKET_VALUE: [10.0, 20.0, 30.0, 40.0, 50.0],
            AGE: [22, 23, 24, 25, 26]
        })
        X_syn = pd.DataFrame({
            SEASON_ID: ["2010", "2010", "2010", "2011", "2011"],
            MARKET_VALUE: [12.0, 18.0, 33.0, 39.0, 49.0],
            AGE: [21, 23, 26, 24, 27]
        })
        result = self.metric.compute(X_real, X_syn)["dataset"] # Only check for dataset stats for now

        self.assertIn(MARKET_VALUE, result)
        self.assertIn(AGE, result)

        # ensure the statistics contain expected keys
        for column in [MARKET_VALUE, AGE]:
            stats = result[column]
            self.assertIn("mean_deviation", stats)
            self.assertIn("median_deviation", stats)
            self.assertIn("std_deviation", stats)
            self.assertIn("min", stats)
            self.assertIn("max", stats)
            self.assertIn("real", stats["min"])
            self.assertIn("synth", stats["min"])
            self.assertIn("real", stats["max"])
            self.assertIn("synth", stats["max"])

        # check some values
        age_stats = result[AGE]
        self.assertEqual(age_stats["min"]["real"], 22)
        self.assertEqual(age_stats["min"]["synth"], 21)
        self.assertEqual(age_stats["median_deviation"], 0)
        self.assertGreater(age_stats["mean_deviation"], 0)

    def test_categorical_statistics(self):
        X_real = pd.DataFrame({
            SEASON_ID: ["2010", "2010", "2010", "2011", "2011", "2011"],
            FOOT: ["right", "right", "right", "left", "right", "both"]
        })
        X_syn = pd.DataFrame({
            SEASON_ID: ["2010", "2010", "2010", "2011", "2011", "2011"],
            FOOT: ["right", "right", "left", "left", "right", "both"]
        })
        result = self.metric.compute(X_real, X_syn)["dataset"] # Only check for dataset stats for now

        self.assertIn(FOOT, result)

        # ensure categorical statistics contain expected keys
        stats = result[FOOT]
        self.assertIn("mode", stats)
        self.assertIn("real", stats["mode"])
        self.assertIn("synth", stats["mode"])
        self.assertIn("entropy", stats)
        self.assertIn("deviation", stats["entropy"])
        self.assertIn("real", stats["entropy"])
        self.assertIn("synth", stats["entropy"])
        
        modes = stats["mode"]
        self.assertEqual(modes["real"], ["right"])
        self.assertEqual(modes["synth"], ["right"])
        self.assertNotEquals(stats["entropy"]["real"], stats["entropy"]["synth"])

if __name__ == "__main__":
    unittest.main()
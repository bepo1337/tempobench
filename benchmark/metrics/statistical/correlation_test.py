import unittest
import pandas as pd

from benchmark.utils import MARKET_VALUE, FOOT, SEASON_ID, POSITION, LAST_TRANSFER_FEE, CLUB, MARKET_VALUE_CATEGORY
from benchmark.utils import AGE
from .correlation import Correlation

class TestCorrelation(unittest.TestCase):

    def setUp(self):
        self.metric = Correlation()

    # PEARSON
    def test_pearson_no_deviation(self):
        X_real = pd.DataFrame({MARKET_VALUE: [1, 2, 3, 4, 5],
                               LAST_TRANSFER_FEE: [2, 4, 6, 8, 10],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = X_real.copy()

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertEqual(result["corr_matrix_distance"], 0.0)
        deviation = result["column_deviations"][f"{MARKET_VALUE},{LAST_TRANSFER_FEE}"]
        self.assertEqual(deviation, 0)

    def test_pearson_max_deviation(self):
        X_real = pd.DataFrame({MARKET_VALUE: [1, 2, 3, 4, 5], LAST_TRANSFER_FEE: [2, 4, 6, 8, 10],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = pd.DataFrame({MARKET_VALUE: [1, 2, 3, 4, 5], LAST_TRANSFER_FEE: [10, 8, 6, 4, 2],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})

        result = self.metric.compute(X_real, X_syn)
        dataset_result = result["dataset"]
        self.assertGreater(dataset_result["corr_matrix_distance"], 0)
        deviation = dataset_result["column_deviations"][f"{MARKET_VALUE},{LAST_TRANSFER_FEE}"]
        self.assertEqual(deviation, 1)

        goalie_result = result["Goalkeeper"]
        self.assertGreater(goalie_result["corr_matrix_distance"], 0)
        deviation = goalie_result["column_deviations"][f"{MARKET_VALUE},{LAST_TRANSFER_FEE}"]
        self.assertEqual(deviation, 1)

    def test_pearson_some_deviation(self):
        X_real = pd.DataFrame({MARKET_VALUE: [1, 2, 3, 4, 5],
                               LAST_TRANSFER_FEE: [2, 4, 6, 8, 10],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = pd.DataFrame({MARKET_VALUE: [1, 2, 3, 4, 5],
                              LAST_TRANSFER_FEE: [2, 5, 7, 9, 11],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertGreater(result["corr_matrix_distance"], 0)
        self.assertLess(result["corr_matrix_distance"], 0.3)
        deviation = result["column_deviations"][f"{MARKET_VALUE},{LAST_TRANSFER_FEE}"]
        self.assertGreater(deviation, 0)
        self.assertLess(deviation, 1)

    # CORR RATIO
    def test_corr_ratio_no_deviation(self):
        X_real = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"],
                               MARKET_VALUE: [10, 20, 10, 20, 30],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = X_real.copy()

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertEqual(result["corr_matrix_distance"], 0.0)
        deviation = result["column_deviations"][f"{CLUB},{MARKET_VALUE}"]
        self.assertEqual(deviation, 0)

    def test_corr_ratio_max_deviation(self):
        """Completl"""
        X_real = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"], MARKET_VALUE: [10, 20, 10, 20, 30],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"], MARKET_VALUE: [333, 333, 333, 333, 333],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertGreater(result["corr_matrix_distance"], 0.1)
        deviation = result["column_deviations"][f"{CLUB},{MARKET_VALUE}"]
        self.assertEqual(deviation, 1)

    def test_corr_ratio_some_deviation(self):
        X_real = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"], MARKET_VALUE: [10, 20, 10, 20, 30],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = pd.DataFrame({CLUB: ["A", "B", "A", "C", "C"], MARKET_VALUE: [11, 19, 12, 21, 31],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertGreater(result["corr_matrix_distance"], 0)
        deviation = result["column_deviations"][f"{CLUB},{MARKET_VALUE}"]
        self.assertGreater(deviation, 0)
        self.assertLess(deviation, 1)

    # CATEGORICAL VARIABLES (CRAMERS V)
    def test_cramers_v_no_deviation(self):
        X_real = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"],
                               FOOT: ["X", "Y", "X", "Y", "Z"],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = X_real.copy()

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertEqual(result["corr_matrix_distance"], 0.0)
        deviation = result["column_deviations"][f"{CLUB},{FOOT}"]
        self.assertEqual(deviation, 0)

    def test_cramers_v_max_deviation(self):
        X_real = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"], FOOT: ["X", "Y", "X", "Y", "Z"],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = pd.DataFrame({CLUB: ["A", "A", "B", "B", "C", "C"], FOOT: ["X", "Z", "X", "Z", "X", "Z"],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertGreater(result["corr_matrix_distance"], 0)
        deviation = result["column_deviations"][f"{CLUB},{FOOT}"]
        self.assertEqual(deviation, 1)

    def test_cramers_v_some_deviation(self):
        X_real = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"], FOOT: ["X", "Y", "X", "Y", "Z"],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})
        X_syn = pd.DataFrame({CLUB: ["A", "B", "A", "B", "C"], FOOT: ["X", "Y", "X", "Z", "Z"],
                               POSITION: ["Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper", "Goalkeeper"]})

        result = self.metric.compute(X_real, X_syn)["dataset"]
        self.assertGreater(result["corr_matrix_distance"], 0)
        deviation = result["column_deviations"][f"{CLUB},{FOOT}"]
        self.assertGreater(deviation, 0)
        self.assertLess(deviation, 1)

if __name__ == "__main__":
    unittest.main()
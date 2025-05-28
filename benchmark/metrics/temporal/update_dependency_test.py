import unittest
import pandas as pd

from benchmark import PLAYER_ID
from benchmark.metrics.temporal.update_dependency import UpdateDependency
from benchmark.utils import VALIDITY_START, VALIDITY_END, load_real_data, INJURY_CATEGORY, REASON, LEAGUE, \
    LEAGUE_PLAYED_MATCHES, LEAGUE_MINUTES_PLAYED


class TestUpdateDependency(unittest.TestCase):

    def setUp(self):
        self.metric = UpdateDependency()
        self.real = load_real_data()

    def test_real_data_with_itself_full_precision_and_recall(self):
        X_real = self.real
        X_syn = X_real.copy()
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 1)

    def test_no_rules_no_precision_no_recall(self):
        X_real = self.real
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2010-07-01", "2010-07-02", "2010-07-01", "2010-07-02"],
            INJURY_CATEGORY:["completly different", "DIFFERENT", "completly different", "DIFFERENT"],
            LEAGUE: ["SERIE A", None, "SERIE A", "dont_care"]
        })

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)


    def test_two_rules_match(self):
        X_real = self.real
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2010-07-01", "2010-07-02", "2010-07-01", "2010-07-02"],
            LEAGUE_PLAYED_MATCHES:[1, 2, 1, 2],
            LEAGUE_MINUTES_PLAYED:[90, 180, 90, 180],
        })

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["precision"], 2/54)
        self.assertEqual(result["recall"], 1)


if __name__ == "__main__":
    unittest.main()

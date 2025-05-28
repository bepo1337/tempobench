import unittest
import pandas as pd

from benchmark import PLAYER_ID
from benchmark.metrics.temporal.gsp import GeneralizedSequentialPattern
from benchmark.utils import VALIDITY_START, VALIDITY_END, load_real_data, INJURY_CATEGORY, REASON, LEAGUE,CLUB
from .mean_validity import MeanValidity

buli = "Bundesliga"
pl = "Premier League"
injury_and_transfer = "injury,transfer"

class TestGSP(unittest.TestCase):

    def setUp(self):
        self.metric = GeneralizedSequentialPattern()
        self.real = load_real_data()

    def test_real_data_with_itself_100_percent(self):
        X_real = self.real
        X_syn = X_real.copy()
        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result[INJURY_CATEGORY]["precision"], 1)
        self.assertEqual(result[INJURY_CATEGORY]["recall"], 1)
        self.assertEqual(result[LEAGUE]["precision"], 1)
        self.assertEqual(result[LEAGUE]["recall"], 1)

    def test_no_sequences_0_percentage(self):
        X_real = self.real
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            REASON: [injury_and_transfer, injury_and_transfer, injury_and_transfer, injury_and_transfer],
            VALIDITY_START: ["2010-07-01", "2010-07-02", "2010-07-01", "2010-07-02"],
            INJURY_CATEGORY:["completly different", "DIFFERENT", "completly different", "DIFFERENT"],
            LEAGUE: ["SERIE A", None, "SERIE A", "dont_care"],
            CLUB: ["AC MILAN", None, "AC MILAN", "dont_care"]
        })

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result[INJURY_CATEGORY]["precision"], 0)
        self.assertEqual(result[INJURY_CATEGORY]["recall"], 0)
        self.assertEqual(result[LEAGUE]["precision"], 0)
        self.assertEqual(result[LEAGUE]["recall"], 0)
        self.assertEqual(result[CLUB]["precision"], 0)
        self.assertEqual(result[CLUB]["recall"], 0)


    def test_some_matches(self):
        X_real = self.real
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3],
            REASON: [injury_and_transfer, injury_and_transfer, injury_and_transfer, injury_and_transfer,injury_and_transfer,injury_and_transfer, injury_and_transfer],
            VALIDITY_START: ["2010-07-01", "2010-07-02", "2010-07-01", "2010-07-02", "2010-07-01", "2010-07-02", "2010-07-03"],
            INJURY_CATEGORY:["leg", "foot", "completly different", "DIFFERENT", "leg", "foot", "knee"],
            LEAGUE:[buli, None, buli, pl, pl, pl, pl],
            CLUB:["FC Liverpool", None, "FC Liverpool", None, "Some other club", "Club2", "club3"],
        })

        result = self.metric.compute(X_real, X_syn)
        self.assertGreater(result[INJURY_CATEGORY]["precision"], 0)
        self.assertLess(result[INJURY_CATEGORY]["precision"], 1)
        self.assertGreater(result[INJURY_CATEGORY]["recall"], 0)
        self.assertLess(result[INJURY_CATEGORY]["recall"], 1)

        self.assertGreater(result[LEAGUE]["precision"], 0)
        self.assertLess(result[LEAGUE]["precision"], 1)
        self.assertGreater(result[LEAGUE]["recall"], 0)
        self.assertLess(result[LEAGUE]["recall"], 1)

        self.assertGreater(result[CLUB]["precision"], 0)
        self.assertLess(result[CLUB]["precision"], 1)
        self.assertGreater(result[CLUB]["recall"], 0)
        self.assertLess(result[CLUB]["recall"], 1)


if __name__ == "__main__":
    unittest.main()

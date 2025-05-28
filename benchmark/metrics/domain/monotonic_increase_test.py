import unittest
import pandas as pd
from benchmark.utils import PLAYER_ID, LEAGUE_PLAYED_MATCHES, LEAGUE_MINUTES_PLAYED, \
    LEAGUE_GOALS, INTERNATIONAL_PLAYED_MATCHES, INTERNATIONAL_MINUTES_PLAYED, INTERNATIONAL_GOALS, VALIDITY_START, \
    VALIDITY_END, AGE
from .monotonic_increase import MonotonicIncrease


class TestMonotonicIncrease(unittest.TestCase):

    def setUp(self):
        self.metric = MonotonicIncrease()

    def test_no_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2011-04-01", "2011-04-03", "2019-05-20", "2019-05-22"],
            VALIDITY_END: ["2011-04-02", "2011-04-04", "2019-05-21", "2019-05-23"],
            AGE: [30, 31, 25, 25],
            LEAGUE_PLAYED_MATCHES: [1, 2, 3, 4],
            LEAGUE_MINUTES_PLAYED: [10, 20, 30, 40],
            LEAGUE_GOALS: [0, 1, 1, 2],
            INTERNATIONAL_PLAYED_MATCHES: [0, 1, 2, 3],
            INTERNATIONAL_MINUTES_PLAYED: [0, 5, 10, 15],
            INTERNATIONAL_GOALS: [0, 0, 1, 1]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 0.0)
        self.assertEqual(result["rows_violated"], 0.0)

    def test_all_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2011-04-01", "2011-04-03", "2019-05-20", "2019-05-22"],
            VALIDITY_END: ["2011-04-02", "2011-04-04", "2019-05-21", "2019-05-23"],
            AGE: [30, 29, 25, 21],
            LEAGUE_PLAYED_MATCHES: [2, 1, 4, 3],
            LEAGUE_MINUTES_PLAYED: [20, 10, 40, 30],
            LEAGUE_GOALS: [1, 0, 2, 1],
            INTERNATIONAL_PLAYED_MATCHES: [1, 0, 3, 2],
            INTERNATIONAL_MINUTES_PLAYED: [5, 0, 15, 10],
            INTERNATIONAL_GOALS: [0, 0, 1, 0]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 1.0)
        self.assertEqual(result["rows_violated"], 1.0)

    def test_partial_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2011-04-01", "2011-04-03", "2019-05-20", "2019-05-22"],
            VALIDITY_END: ["2011-04-02", "2011-04-04", "2019-05-21", "2019-05-23"],
            AGE: [30, 31, 25, 21],
            LEAGUE_PLAYED_MATCHES: [1, 3, 4, 2],
            LEAGUE_MINUTES_PLAYED: [10, 5, 30, 40],
            LEAGUE_GOALS: [0, 2, 1, 3],
            INTERNATIONAL_PLAYED_MATCHES: [0, 1, 3, 2],
            INTERNATIONAL_MINUTES_PLAYED: [0, 5, 15, 10],
            INTERNATIONAL_GOALS: [1, 0, 1, 0]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 1)
        self.assertEqual(result["rows_violated"], 1)
        details = result["details"]
        self.assertEqual(details[AGE], 1 / 2)
        self.assertEqual(details[LEAGUE_PLAYED_MATCHES], 1 / 2)
        self.assertEqual(details[LEAGUE_MINUTES_PLAYED], 1 / 2)
        self.assertEqual(details[LEAGUE_GOALS], 0)
        self.assertEqual(details[INTERNATIONAL_PLAYED_MATCHES], 1 / 2)
        self.assertEqual(details[INTERNATIONAL_PLAYED_MATCHES], 1 / 2)
        self.assertEqual(details[INTERNATIONAL_GOALS], 1)

    def test_more_partial_violations(self):
        data = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3, 3, 3, 3],
            VALIDITY_START: ["2011-04-01", "2011-04-03", "2019-05-20", "2019-05-22",   "2011-04-01", "2011-04-03", "2011-04-05", "2011-04-07", "2011-04-09", "2011-04-11"],
            VALIDITY_END: ["2011-04-02", "2011-04-04", "2019-05-21", "2019-05-23",     "2011-04-02", "2011-04-04", "2011-04-06", "2011-04-08", "2011-04-10", "2011-04-12"],
            AGE: [30, 31, 25, 21, 20,20,20,20,20,20],
            LEAGUE_PLAYED_MATCHES: [1, 3, 4, 2, 1,1,1,1,1,1],
            LEAGUE_MINUTES_PLAYED: [10, 5, 30, 40,  20,20,20,20,20,20],
            LEAGUE_GOALS: [0, 2, 1, 3,   4, 3, 3, 3, 2, 1],
            INTERNATIONAL_PLAYED_MATCHES: [0, 1, 3, 2,   3, 4, 5 ,6 ,7 ,8],
            INTERNATIONAL_MINUTES_PLAYED: [0, 5, 15, 10,  20, 10, 15, 10, 20, 10],
            INTERNATIONAL_GOALS: [1, 0, 1, 0,  1, 2, 3 ,4 ,5 ,3]
        })

        # 2,4,5,6
        result = self.metric.compute(None, data)
        self.assertEqual(result["entities_violated"], 1)
        self.assertEqual(result["rows_violated"], 6/7)
        details = result["details"]
        self.assertEqual(details[AGE], 1 / 7)
        self.assertEqual(details[LEAGUE_PLAYED_MATCHES], 1 / 7)
        self.assertEqual(details[LEAGUE_MINUTES_PLAYED], 1 / 7)
        self.assertEqual(details[LEAGUE_GOALS], 3/7)
        self.assertEqual(details[INTERNATIONAL_PLAYED_MATCHES], 1 / 7)
        self.assertEqual(details[INTERNATIONAL_MINUTES_PLAYED], 4 / 7)
        self.assertEqual(details[INTERNATIONAL_GOALS], 3/7)


if __name__ == "__main__":
    unittest.main()

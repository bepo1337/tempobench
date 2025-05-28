import unittest
import pandas as pd
import datetime
from .timestamp_min_max import TimestampMinMax
from benchmark.utils import VALIDITY_START, VALIDITY_END

def date_string_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")


class TestTimestampMinMax(unittest.TestCase):

    def setUp(self):
        self.metric = TimestampMinMax()

    def test_no_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2015-01-01", "2018-06-15", "2020-06-30"],
            VALIDITY_END: ["2015-02-01", "2019-03-10", "2020-06-30"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 0.0)

    def test_all_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2005-05-10", "2015-07-15", "2009-12-31"],
            VALIDITY_END: ["2021-06-30", "2020-07-01", "2030-12-25"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 1.0)

    def test_partial_violations(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2011-04-01", "2009-12-31", "2019-05-20"],
            VALIDITY_END: ["2012-06-30", "2025-09-15", "2020-06-30"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 1 / 3)

    def test_edge_case_min_max_dates(self):
        data = pd.DataFrame({
            VALIDITY_START: ["2010-07-01"],
            VALIDITY_END: ["2020-06-30"]
        })
        result = self.metric.compute(None, data)
        self.assertEqual(result["proportion"], 0.0)

    def test_timestamp_is_outside_interval(self):
        self.assertFalse(self.metric.timestamp_is_outside_interval("2015-05-10"))
        self.assertTrue(self.metric.timestamp_is_outside_interval("2009-12-31"))
        self.assertTrue(self.metric.timestamp_is_outside_interval("2021-07-01"))


if __name__ == "__main__":
    unittest.main()

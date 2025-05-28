import json
import pandas as pd
import unittest

from benchmark import ValidityEndBeforeStart, ValiditiyMaxTimeMetricName, ValidityOverlapMetricName
from benchmark.metrics.domain.domain_metrics import all_domain_metrics, all_domain_metrics_dict
from benchmark.metrics.domain.monotonic_increase import MonotonicIncreaseMetricName
from benchmark.metrics.domain.no_tuple_valid_beyond_end_of_june import NoTupleValidBeyondEndOfJuneMetricName
from benchmark.metrics.domain.static_data_remains_unchanged import StaticDataRemainsUnchangedMetricName
from benchmark.metrics.domain.timestamp_min_max import TimestampMinMaxMetricName
from benchmark.metrics.domain.validity_end_before_start import ValidityEndBeforeStartMetricName
from benchmark.utils import load_real_data


class TestAllDomainMetrics(unittest.TestCase):

    def setUp(self):
        self.data = load_real_data()
        self.metrics = all_domain_metrics_dict

    def test_monotonic_increase(self):
        result = self.metrics[MonotonicIncreaseMetricName].compute(None, self.data)
        expected_dict = {"entities_violated": 0,
                         "rows_violated": 0,
                         'details': {'age': 0.0,
                                     'international_goals': 0.0,
                                     'international_minutes_played': 0.0,
                                     'international_played_matches': 0.0,
                                     'league_goals': 0.0,
                                     'league_minutes_played': 0.0,
                                     'league_played_matches': 0.0}}
        self.assertEqual(result, expected_dict)

    # def test_no_tuple_beyond_end_of_june_increase(self):
    #     result = self.metrics[NoTupleValidBeyondEndOfJuneMetricName].compute(None, self.data)
    #     expected_dict = {"proportion": 0}
    #     self.assertEqual(result, expected_dict)


    def test_static_data_remains_unchanged(self):
        result = self.metrics[StaticDataRemainsUnchangedMetricName].compute(None, self.data)
        expected_dict = {"entities_violated": 0,
                "details": "TODO"}
        self.assertEqual(result, expected_dict)

    def test_timestamp_min_max(self):
        result = self.metrics[TimestampMinMaxMetricName].compute(None, self.data)
        expected_dict = {"proportion": 0}
        self.assertEqual(result, expected_dict)


    def test_validity_end_before_start(self):
        result = self.metrics[ValidityEndBeforeStartMetricName].compute(None, self.data)
        expected_dict = {"proportion": 0}
        self.assertEqual(result, expected_dict)

    def test_validity_max_time(self):
        result = self.metrics[ValiditiyMaxTimeMetricName].compute(None, self.data)
        expected_dict = {"proportion": 0}
        self.assertEqual(result, expected_dict)

    def test_validity_overlap(self):
        result = self.metrics[ValidityOverlapMetricName].compute(None, self.data)
        expected_dict = {"proportion": 0}
        self.assertEqual(result, expected_dict)

if __name__ == "__main__":
    unittest.main()

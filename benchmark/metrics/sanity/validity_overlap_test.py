import unittest
from datetime import datetime
import pandas as pd

from benchmark.utils import VALIDITY_START, VALIDITY_END, PLAYER_ID
from .validity_overlap import ValidityOverlap


class TestValidityOverlap(unittest.TestCase):

    def setUp(self):
        self.metric = ValidityOverlap()

    def test_start_before_or_equal_to_prev_end(self):
        self.assertTrue(self.metric.start_before_or_equal_to_prev_end("2024-02-13", "2024-02-13"))
        self.assertTrue(self.metric.start_before_or_equal_to_prev_end("2024-02-12", "2024-02-13"))
        self.assertFalse(self.metric.start_before_or_equal_to_prev_end("2024-02-14", "2024-02-13"))

    def test_compute_no_violations(self):
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2024-02-13", "2025-02-14", "2023-01-01", "2024-01-01"],
            VALIDITY_END: ["2025-02-12", "2026-01-14", "2023-12-31", "2024-12-01"]
        })
        result = self.metric.compute(None, X_syn)
        self.assertEqual(result["proportion"], 0.0)

    def test_compute_all_violations(self):
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2],
            VALIDITY_START: ["2024-02-13", "2024-02-12", "2023-01-01", "2022-12-30"],
            VALIDITY_END: ["2025-02-12", "2025-02-11", "2023-12-31", "2023-12-20"]
        })
        result = self.metric.compute(None, X_syn)
        self.assertEqual(result["proportion"], 1.0)

    def test_compute_mixed_violations(self):
        X_syn = pd.DataFrame({
            PLAYER_ID: [1, 1, 2, 2, 3, 3],
            VALIDITY_START: ["2024-02-13", "2024-02-13", "2023-01-01", "2023-12-31", "2025-05-01", "2025-06-01"],
            VALIDITY_END: ["2025-02-12", "2026-02-10", "2023-12-31", "2024-12-30", "2025-05-31", "2026-04-30"]
        })
        result = self.metric.compute(None, X_syn)
        self.assertEqual(result["proportion"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
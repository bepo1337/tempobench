import unittest
import pandas as pd
from benchmark.utils import PLAYER_ID, all_num_col_names, all_cat_col_without_ids
from benchmark.metrics.temporal.intra_class_distance import IntraClassDistance

class TestAvgIntraClassDistance(unittest.TestCase):

    def setUp(self):
        self.metric = IntraClassDistance()

    def test_no_deviation(self):
        data = {
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3],
        }
        for col in all_num_col_names:
            data[col] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for col in all_cat_col_without_ids:
            data[col] = ['a', 'b', 'a', 'b', 'a', 'b', 'a']

        X_real = pd.DataFrame(data)
        X_syn = X_real.copy()

        result = self.metric.compute(X_real, X_syn)
        self.assertEqual(result["deviation"], 0.0)
        self.assertGreater(result["real_avg_distance"], 0)
        self.assertGreater(result["synth_avg_distance"], 0)

    def test_deviation(self):
        data_real = {
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3],
        }
        data_syn = {
            PLAYER_ID: [1, 1, 2, 2, 3, 3, 3],
        }
        for col in all_num_col_names:
            data_real[col] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            data_syn[col] = [1, 1, 0.5, 0.4, 0.3, 0, 0]
        for col in all_cat_col_without_ids:
            data_real[col] = ['a', 'b', 'a', 'b', 'a', 'b', 'a']
            data_syn[col] = ['b', 'a', 'b', 'a', 'b', 'a', 'b']

        X_real = pd.DataFrame(data_real)
        X_syn = pd.DataFrame(data_syn)

        result = self.metric.compute(X_real, X_syn)
        self.assertGreater(result["deviation"], 0.0)
        self.assertGreater(result["real_avg_distance"], 0)
        self.assertGreater(result["synth_avg_distance"], 0)

if __name__ == "__main__":
    unittest.main()
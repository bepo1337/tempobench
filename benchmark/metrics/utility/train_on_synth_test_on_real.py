from typing import List

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score

from benchmark.metrics.utility.train_model_classification import create_categorical_trained_model
from benchmark.metrics.utility.train_model_classification_mv import create_categorical_trained_model_market_value
from benchmark.metrics.utility.train_model_regression import create_regression_trained_model
from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    combined_dtw, load_real_data_test
from benchmark.metrics import BenchmarkMetric
from benchmark.utils.normalize import normalize_and_encode_dfs_in_place
from benchmark.utils.pandas_util import get_players_as_list_of_df

TrainOnSynthTestOnRealMetricName = "TrainOnSynthTestOnReal"

class TrainOnSynthTestOnReal(BenchmarkMetric):
    """Trains a model on synthetic data and tests it on real data."""

    def compute(self, X_real, X_syn) -> dict:
        # train synth model
        # commented out because it does not work properly yet
        # classification_results = self.classification_results(X_syn.copy())
        regression_results = self._compute_regression_results(X_syn.copy())

        return {
            # "classification": classification_results, #comment in when classification has good results
            "regression": regression_results
        }

    def name(self) -> str:
        return TrainOnSynthTestOnRealMetricName

    def category(self) -> str:
        return Category.UTILITY.value

    def classification_results(self, syn_copy):
        test_set = load_real_data_test()
        try:
            syn_model, categories = create_categorical_trained_model_market_value(syn_copy, test_set, real=False, name="benchmark")
        except Exception as e:
            return {
                "agreement_rate": "error in catboost"
            }

        real_model = self._load_real_classification_model_mv()
        agreement_rate = self._compute_agreement_rate(real_model, syn_model, categories)
        return {
            "agreement_rate": agreement_rate
        }

    def _load_real_classification_model_mv(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(file_dir,"models/real_classification_mv.cbm")
        real_model = CatBoostClassifier(loss_function="MultiClass")
        real_model.load_model(model_path, format="cbm")
        return real_model

    def _compute_agreement_rate(self, real_model, syn_model, categories):
        #todo pass test set and store confusion matrix in temp file in order to visualize it?
        file_dir = os.path.dirname(os.path.abspath(__file__))
        val_path = os.path.join(file_dir,"data/X_test_classification_mv.json")
        X_test = pd.read_json(val_path)
        preds1 = real_model.predict(X_test)
        preds2 = syn_model.predict(X_test)
        agreement = np.mean(preds1 == preds2)
        return agreement

    def _compute_regression_results(self, syn_copy: pd.DataFrame):
        real_model = self._load_real_regression_model()
        test_set = load_real_data_test()
        syn_model, categories = create_regression_trained_model(syn_copy, test_set)
        real_rmse, real_rsquared = self._eval_regression_on_test(real_model)
        syn_rmse, syn_rsquared = self._eval_regression_on_test(syn_model)
        rmse_deviation = abs(syn_rmse - real_rmse) / real_rmse
        rsquared_deviation = abs(syn_rsquared - real_rsquared) / real_rsquared
        return {
            "rmse": {
                "real": real_rmse,
                "syn": syn_rmse,
                "deviation": rmse_deviation
            },
            "r-squared": {
                "real": real_rsquared,
                "syn": syn_rsquared,
                "deviation": rsquared_deviation
            }
        }

    def _load_real_regression_model(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(file_dir,"models/real_with_split_regression.cbm")
        real_model = CatBoostRegressor()
        real_model.load_model(model_path, format="cbm")
        return real_model

    def _eval_regression_on_test(self, model: CatBoostRegressor):
        """ Returns RMSE And r Squared"""
        file_dir = os.path.dirname(os.path.abspath(__file__))
        X_test_path = os.path.join(file_dir,"data/X_test_regression.csv")
        y_test_path = os.path.join(file_dir,"data/y_test_regression.csv")
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)

        X_test["international_competition"] = X_test["international_competition"].fillna("")
        X_test["citizenship"] = X_test["citizenship"].fillna("")

        pred = model.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        return rmse, r2



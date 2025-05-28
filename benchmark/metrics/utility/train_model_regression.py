import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
import os


from benchmark.utils import (MARKET_VALUE, LEAGUE, CITIZENSHIP, POSITION, FOOT, INTERNATIONAL_COMPETITION, CLUB, AGE, SEASON_ID,
                             LAST_TRANSFER_FEE, HEIGHT, LEAGUE_PLAYED_MATCHES, LEAGUE_MINUTES_PLAYED, LEAGUE_GOALS,
                             INTERNATIONAL_PLAYED_MATCHES, INTERNATIONAL_GOALS, INTERNATIONAL_MINUTES_PLAYED)


def create_regression_trained_model(df_train: pd.DataFrame, df_test: pd.DataFrame, real=False, name="real"):
    """Returns model and list of categories used"""
    categorical_features = [
        LEAGUE, CLUB, CITIZENSHIP, POSITION, FOOT, INTERNATIONAL_COMPETITION
    ]
    numerical_features = [AGE, SEASON_ID, LAST_TRANSFER_FEE, HEIGHT, LEAGUE_PLAYED_MATCHES,
                          LEAGUE_MINUTES_PLAYED, LEAGUE_GOALS, INTERNATIONAL_PLAYED_MATCHES,
                          INTERNATIONAL_GOALS, INTERNATIONAL_MINUTES_PLAYED]
    feature_cols = categorical_features + numerical_features
    target_col = [MARKET_VALUE]
    relevant_cols = feature_cols + target_col

    df_train = df_train[relevant_cols].copy()
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    df_test = df_test[relevant_cols].copy()
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    # split
    # catboost cant deal with nan or null explicitly
    X_train[[CLUB, LEAGUE]] = X_train[[CLUB, LEAGUE]].fillna("other")
    X_test[[CLUB, LEAGUE]] = X_test[[CLUB, LEAGUE]].fillna("other")
    train_dataset = cb.Pool(X_train, y_train, cat_features=categorical_features)
    model = cb.CatBoostRegressor(loss_function='RMSE')

    # train model in grid
    grid = {
        'iterations': [200, 500, 1000],
            'learning_rate': [0.03, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9]
        }
    model.grid_search(grid, train_dataset)

    # if real, then save results on test and dont do it live.
    if real:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(file_dir, f"models/{name}_regression.cbm")
            x_validation_path = os.path.join(file_dir, "data/X_test_regression.csv")
            y_validation_path = os.path.join(file_dir, "data/y_test_regression.csv")
            model.save_model(model_path, format="cbm")
            X_test.to_csv(x_validation_path, index=False)
            y_test.to_csv(y_validation_path, index=False)

    return model, categorical_features
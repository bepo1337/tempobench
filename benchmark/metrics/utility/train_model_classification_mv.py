import os
import pandas as pd
import numpy as np
import catboost as cb

def df_to_catboost_data_model_x_y(df: pd.DataFrame):
    df = df.copy()
    mvu = "market value update"

    df_sorted = df.sort_values(by=["player_id", "validity_start"])
    first_rows = df_sorted.groupby("player_id", as_index=False).first()
    market_value_updates = df[df["reason"].str.contains(mvu)]
    result = pd.concat([first_rows, market_value_updates]).drop_duplicates()

    lag_columns = ["validity_start", "age", "market_value", "club", "league", "season_id", "coach",
                   "international_competition"]
    lag_with_delta_columns = ["league_played_matches", "league_goals", "league_minutes_played", "international_goals",
                              "international_minutes_played", "international_played_matches", "market_value",
                              "validity_start"]
    all_shift_cols = set(lag_columns) | set(lag_with_delta_columns)
    all_shift_cols = list(all_shift_cols)
    df_sorted = result.sort_values(by=["player_id", "validity_start"])
    grouped = df_sorted.groupby("player_id")

    # Create lag features for each column and lag
    lagged = grouped[all_shift_cols].shift(1)
    lagged.columns = [f"lag_{col}" for col in lagged.columns]
    df_sorted = pd.concat([df_sorted, lagged], axis=1)


    # create delta cols
    delta_columns = ["league_played_matches", "league_goals", "league_minutes_played", "international_goals",
                     "international_minutes_played", "international_played_matches", "market_value"]
    for col in delta_columns:
        df_sorted[f"delta_{col}"] = df_sorted[col] - df_sorted[f"lag_{col}"]


    #### get feature columns
    relevant_delta_cols = ["delta_" + x for x in delta_columns if x != "market_value"]
    relevant_lag_cols = ["lag_" + x for x in lag_columns if x != "market_value" and x != "validity_start"]
    relevant_attributes_at_time = ["season_id", "last_transfer_fee", "position", "age", "foot", "citizenship", "height",
                                   "club", "league", "coach", "international_competition"]

    df_sorted['target'] = np.select(
        [df_sorted['market_value'] > df_sorted['lag_market_value'],
         df_sorted['market_value'] < df_sorted['lag_market_value']],
        ['up', 'down'],
        default='same'
    )
    df_sorted['validity_start'] = pd.to_datetime(df_sorted['validity_start'])
    df_sorted['lag_validity_start'] = pd.to_datetime(df_sorted['lag_validity_start'])
    mask = df_sorted['validity_start'].notna() & df_sorted['lag_validity_start'].notna()
    df_sorted.loc[mask, 'delta_days'] = (pd.to_datetime(df_sorted.loc[mask, 'lag_validity_start']) - pd.to_datetime(
        df_sorted.loc[mask, 'validity_start'])).dt.days.abs()
    feature_col = relevant_delta_cols + relevant_lag_cols + relevant_attributes_at_time + ["target",
                                                                                           "delta_market_value",
                                                                                           "delta_days"]
    # get relevant rows
    only_relevant_columns = df_sorted[feature_col]
    only_relevant_rows = only_relevant_columns[only_relevant_columns["delta_market_value"].notna()].copy()
    only_relevant_rows.drop(labels=["delta_market_value"],axis=1, inplace=True)
    cast_float_to_int = relevant_delta_cols + ["lag_age", "lag_season_id", "delta_days"]
    only_relevant_rows[cast_float_to_int] = only_relevant_rows[cast_float_to_int].astype(int)

    # create X and y
    cb_df = only_relevant_rows.copy().reset_index(drop=True)
    y = cb_df["target"]
    X = cb_df.drop("target", axis=1)
    return X,y

def create_categorical_trained_model_market_value(df_train: pd.DataFrame, df_test: pd.DataFrame, real=False, name="real"):
    """Returns model and list of categories used"""
    train_df = df_train.copy()
    test_df = df_test.copy()
    X_train, y_train = df_to_catboost_data_model_x_y(train_df)
    X_test, y_test = df_to_catboost_data_model_x_y(test_df)


    categorical_features = ["lag_club", "lag_league", "lag_coach", "lag_international_competition", "club", "league",
                            "coach", "position", "foot", "citizenship", "international_competition"]

    try:
        train_dataset = cb.Pool(X_train, y_train, cat_features=categorical_features)
    except Exception as e:
        print(e)
        raise ValueError("couldnt create CatBoost Pool, see console for more info")
    model = cb.CatBoostClassifier(loss_function='MultiClass')

    # train model in grid
    grid = {
        'iterations': [100], #todo revert
            # 'learning_rate': [0.03, 0.05],
            # 'depth': [5, 10],
            # 'l2_leaf_reg': [4, 5, 6]
            }
    model.grid_search(grid, train_dataset)

    # todo if real, then save results on test and dont do it live.
    if real:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(file_dir, f"models/{name}_classification_mv.cbm")
            x_validation_path = os.path.join(file_dir, "data/X_test_classification_mv.json")
            y_validation_path = os.path.join(file_dir, "data/y_test_classification_mv.json")
            model.save_model(model_path, format="cbm")
            X_test.to_json(x_validation_path, orient="records")
            y_test.to_json(y_validation_path, orient="records")

    return model, categorical_features
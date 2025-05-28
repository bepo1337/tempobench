import pandas as pd
import os
from benchmark.utils import PLAYER_ID, INJURY, VALIDITY_START, LEAGUE_MINUTES_PLAYED, INJURY_CATEGORY, REASON, POSITION, \
    LEAGUE, CLUB, INTERNATIONAL_COMPETITION, SEASON_ID, FOOT, CITIZENSHIP, HEIGHT, AGE
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier

def create_categorical_trained_model(df: pd.DataFrame, real=False, name="real"):
    print("training classification model...")
    X, y, categories = create_x_y_and_categories(df)

    # split into train and val
    data = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_validation, y_train, y_validation = data

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=categories
    )

    validation_pool = Pool(
        data=X_validation,
        label=y_validation,
        cat_features=categories
    )

    model = CatBoostClassifier(
        iterations=50,
        verbose=50,
        loss_function='MultiClassOneVsAll',
    )
    model.fit(train_pool, eval_set=validation_pool)

    # if real then save X_validation and y_validation and the model itself
    if real:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(file_dir, f"models/{name}_classification.cbm")
        x_validation_path = os.path.join(file_dir, "data/X_validation.csv")
        y_validation_path = os.path.join(file_dir, "data/y_validation.csv")
        model.save_model(model_path, format="cbm")
        X_validation.to_csv(x_validation_path, index=False)
        y_validation.to_csv(y_validation_path, index=False)

    return model, categories


def create_x_y_and_categories(df):
    df_injuries = df[df['reason'].str.contains(INJURY) & ~df['reason'].str.contains("injury end")].copy()
    injury_cat_lag_1 = "injury_category_lag1"
    injury_cat_lag_2 = "injury_category_lag2"
    injury_cat_lag_3 = "injury_category_lag3"
    previous_start = "previous_start"
    previous_minutes_played = "previous_minutes_played"
    day_delta = "day_delta"
    # get unique player ids in the injuries dataframe
    player_ids = df_injuries[PLAYER_ID].unique()
    # get all tuples that have the player id from the original df
    df_player = df[df[PLAYER_ID].isin(player_ids)]
    modified_groups = []
    for player_id, group_df in df_player.groupby(PLAYER_ID):
        group_df = group_df.copy()

        # todo hwo are the last first rows handled? are they null?
        group_df[previous_start] = group_df[VALIDITY_START].shift(-1)
        group_df[previous_minutes_played] = group_df[LEAGUE_MINUTES_PLAYED].shift(-1).fillna(0)

        group_df[VALIDITY_START] = pd.to_datetime(group_df[VALIDITY_START])
        group_df[previous_start] = pd.to_datetime(group_df[previous_start])
        group_df[day_delta] = (group_df[VALIDITY_START] - group_df[previous_start]).dt.days

        # filter only for rows that have injuries
        injuries_only = group_df[
            group_df[REASON].str.contains(INJURY) & ~group_df[REASON].str.contains("injury end")].copy()

        # Add lagged columns
        default_val = "no_previous"
        injuries_only[injury_cat_lag_1] = injuries_only[INJURY_CATEGORY].shift(1).fillna(default_val)
        injuries_only[injury_cat_lag_2] = injuries_only[INJURY_CATEGORY].shift(2).fillna(default_val)
        injuries_only[injury_cat_lag_3] = injuries_only[INJURY_CATEGORY].shift(3).fillna(default_val)

        # TODO per group add more derived or lag features
        # minutes_played_per_day_since_last_entry
        # shift previous minutes played and previous start

        # left join on index (only update the rows that have an injury)
        group_df = group_df.merge(
            injuries_only[[injury_cat_lag_1, injury_cat_lag_2, injury_cat_lag_3]],
            how="left",
            left_index=True,
            right_index=True
        )

        modified_groups.append(group_df)
    used_categorical_features = [
        POSITION,
        LEAGUE,
        CLUB,
        INTERNATIONAL_COMPETITION,
        SEASON_ID,
        FOOT,
        CITIZENSHIP,

        injury_cat_lag_1,
        injury_cat_lag_2,
        injury_cat_lag_3,
    ]
    used_numerical_features = [
        HEIGHT,
        AGE,
    ]
    target_features = [INJURY_CATEGORY]
    selected_columns = used_categorical_features + used_numerical_features + target_features
    # concat back into single df
    df_new = pd.concat(modified_groups).sort_index()
    print(df_new.shape)  # this is less because rows thaan in the original df bcause some players dont have injuries
    # filter only for injury rows. Dont wanna include other rows where the injuries are still present
    df_new = df_new[df_new[REASON].str.contains(INJURY) & ~df_new[REASON].str.contains("injury end")].copy()
    df_selected = df_new[selected_columns]
    df_selected.head()
    df_selected.dropna(inplace=True)
    # extract label
    y = df_selected[INJURY_CATEGORY]
    X = df_selected.drop(INJURY_CATEGORY, axis=1)
    return X, y, used_categorical_features
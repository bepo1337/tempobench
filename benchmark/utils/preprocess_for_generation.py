import pandas as pd
import numpy as np
from benchmark.utils import PLAYER_ID, CLUB_ID, COACH_ID, PSEUDONYM, COLUMN_NA, INTERNATIONAL_COMPETITION, \
    DATE_OF_BIRTH, VALIDITY_END, VALIDITY_START

def preprocess_for_gen(df: pd.DataFrame):
    # 1 make all categorical attributes strings
    make_categorical = [PLAYER_ID, CLUB_ID, COACH_ID] #when postprocessing, just turn back into int
    for cat in make_categorical:
        df[cat] = df[cat].astype(str)

    # 2 make all null values to one value such as "N/A"
    df[PSEUDONYM] = df[PSEUDONYM].replace("", COLUMN_NA)
    df[INTERNATIONAL_COMPETITION] = df[INTERNATIONAL_COMPETITION].replace("", COLUMN_NA)
    df.fillna(COLUMN_NA, inplace=True)

    # 3 split dates into 3 columns day, month, year
    split_date_into_parts = [VALIDITY_START, VALIDITY_END, DATE_OF_BIRTH]
    for date_col in split_date_into_parts:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
        df[f"{date_col}_year"] = df[date_col].dt.strftime("%Y")
        df[f"{date_col}_month"] = df[date_col].dt.strftime("%m")
        df[f"{date_col}_day"] = df[date_col].dt.strftime("%d")
        df.drop(columns=[date_col], inplace=True)

    # 4 split reason into binary columns
    reasons = ["regular interval", "new coach", "transfer", "market value update", "injury", "injury end"]
    for reason in reasons:
        col_name = f"reason_{reason.replace(' ', '_')}"
        if reason == "injury":
            df[col_name] = df["reason"].apply(lambda x: int("injury" in x and "injury end" not in x))
        else:
            df[col_name] = df["reason"].apply(lambda x: int(reason in x))

    df.drop("reason", axis=1, inplace=True)

def all_df_preprocess_for_generation(dfs):
    for df in dfs:
        preprocess_for_gen(df)

if __name__ == "__main__":
    df_train = pd.read_json("../data/real_data_train.json")
    df_test = pd.read_json("../data/real_data_test.json")

    dfs = [df_train, df_test]

    all_df_preprocess_for_generation(dfs)
    print("done, saving...")
    df_train.to_json("../data/xxxx.json", orient="records")
    df_test.to_json("../data/yyyy.json", orient="records")
    print("successfully saved")

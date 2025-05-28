import os

import pandas as pd
from benchmark.utils import PLAYER_ID, CLUB_ID, COACH_ID, PSEUDONYM, COLUMN_NA, INTERNATIONAL_COMPETITION, \
    DATE_OF_BIRTH, VALIDITY_END, VALIDITY_START, AGE, LEAGUE_PLAYED_MATCHES, LEAGUE_MINUTES_PLAYED, LEAGUE_GOALS, \
    INTERNATIONAL_MINUTES_PLAYED, INTERNATIONAL_GOALS, INTERNATIONAL_PLAYED_MATCHES, LAST_TRANSFER_FEE, MISSED_MATCHES
import argparse

parser = argparse.ArgumentParser(description="Postprocessing data after generation to fit the format of the ground truth data",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required=True, type=str, help="Input file")
parser.add_argument("-o", "--output", required=False, type=str, help="Output file")

args = parser.parse_args()
input_file = args.input
output_file = args.output
if output_file is None:
    output_file = input_file + "_postproc"


# Load generated samples
file_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_json(f"{file_dir}/../../data/synthesized/input/{input_file}.json")
dates_to_be_post_formatted = [VALIDITY_START, VALIDITY_END, DATE_OF_BIRTH]
#Recombine date columns
def recombine_dates(df):
    for col in dates_to_be_post_formatted:
        year_col = f"{col}_year"
        month_col = f"{col}_month"
        day_col = f"{col}_day"

        df[col] = pd.to_datetime(
            df[year_col].astype(str) + "-" + df[month_col].astype(str).str.zfill(2) + "-" + df[day_col].astype(
                str).str.zfill(2),
            format="%Y-%m-%d",
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        df.drop([year_col, month_col, day_col], axis=1, inplace=True)

#Recombine reason binary columns into a single string
def recombine_reason(df):
    reason_map = ["regular interval", "new coach", "transfer", "market value update", "injury", "injury end"]
    reason_cols = [f"reason_{r.replace(' ', '_')}" for r in reason_map]

    def row_to_reason(row):
        included_reasons = []
        for reason, col in zip(reason_map, reason_cols):
            if row.get(col, 0) == 1:
                included_reasons.append(reason)
        return ", ".join(included_reasons) if included_reasons else "NO_REASON"

    df["reason"] = df.apply(row_to_reason, axis=1)
    df.drop(columns=reason_cols, inplace=True)

# Replace "N/A" with np.nan for nullable fields
def restore_nas(df):
    df.replace(COLUMN_NA, pd.NA, inplace=True)

# Convert PLAYER_ID, CLUB_ID, COACH_ID back to int
def restore_categoricals(df):
    for col in [PLAYER_ID, CLUB_ID, COACH_ID]:
        df[col] = pd.to_numeric(df[col]).astype("Int64")


def set_default_date_for_missing_dates(df: pd.DataFrame, default_date: str = "2021-01-01"):
    for col in dates_to_be_post_formatted:
        df[col] = df[col].fillna(default_date)


def set_empty_strings(df: pd.DataFrame):
    df[PSEUDONYM] = df[PSEUDONYM].replace(COLUMN_NA, "")
    df[INTERNATIONAL_COMPETITION] = df[INTERNATIONAL_COMPETITION].replace(COLUMN_NA, "")


def drop_target_column_tabsyn(df: pd.DataFrame):
    df.drop(columns=['target'], errors='ignore', inplace=True)

def floats_to_ints(df: pd.DataFrame):
    columns = [AGE,
               LEAGUE_PLAYED_MATCHES, LEAGUE_MINUTES_PLAYED, LEAGUE_GOALS,
               INTERNATIONAL_MINUTES_PLAYED, INTERNATIONAL_PLAYED_MATCHES, INTERNATIONAL_GOALS,
               MISSED_MATCHES, LAST_TRANSFER_FEE]

    for col in columns:
        df[col] = df[col].round().astype("Int64")

def postprocess_gen_data(df: pd.DataFrame):
    recombine_dates(df)
    recombine_reason(df)
    restore_categoricals(df)
    set_default_date_for_missing_dates(df)
    set_empty_strings(df)
    restore_nas(df)
    drop_target_column_tabsyn(df)
    floats_to_ints(df)


if __name__ == "__main__":
    postprocess_gen_data(df)
    df.to_json(f"{file_dir}/../../data/synthesized/output/{output_file}.json", orient="records")
    print("done")
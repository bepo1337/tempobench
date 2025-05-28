import json
import argparse
import os
import math

import mysql
import pandas
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import Error
from pandas import Timestamp
from datetime import timedelta
from collections import OrderedDict
from datetime import datetime
import numpy as np

from tqdm import tqdm

from sql_statements import performance_sql, competition_sql, missed_games_sql
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutil import load_data_into_dict_with_player_id

GRUND = "grund"
VERLETZUNG_GRUND = "verletzung"
VERLETZUNG_ENDE_GRUND = "verletzung_ende"
VERLETZUNG_ENDE_DATUM = "verletzung_end"
INJURY_START = "verletzung_start"
INJURY_END = "verletzung_end"
MARKET_VALUE_REASON = "mw update"
TRANSFER = "transfer"
TIMESTAMP = "timestamp"
REGULAR_INTERVAL = "regelmaessiger interval"
NEW_COACH = "neuer trainer"

VALIDITY_START = "validity_start"
VALIDITY_END = "validity_end"

VEREIN_ID = "verein_id"
CLUB_ID = "club_id"
WETTBEWERB_ID = "wettbewerb_id"
SAISON_ID = "saison_id"
VERLETZUNG_COLUMN = "verletzung"
VEREIN = "verein"
LIGA ="liga"
TRAINER_COL = "trainer"
MARKET_VAL_COL = "mw"
GEBURTSDATUM = "geburtsdatum"
AGE = "age"
ABLOESE = "abloese"
MARKET_VALUE_CATEGORY = "market_value_category"
INJURY_CATEGORY = "injury_category"
REASON = "reason"
FOOT = "foot"
INJURY_COLUMN = "injury"
PLAYER_ID = "player_id"
MISSED_MATCHES = "missed_matches"

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data_bis_19_1k.json", nargs="?",
                    help="What file name to import from /data directory (default: data_bis_19_1k.json)")

parser.add_argument("--output", default="test_out_prod.json", nargs="?",
                    help="What file name to output to /data directory (default: test_out_prod.json)")


args, unknown = parser.parse_known_args()
import_input = "../data/" + args.input
output_file = "../data/" + args.output

with open("../data/mapping/injury_to_category.json", "r") as f:
    injury_to_category = json.load(f)

season_to_quantiles = {} # will be filled further below before being used in method
total_unwanted_players_due_to_injury = [] # will be filled furhter below

with open("../data/mapping/country.json", "r") as f:
    country_translations = json.load(f)

with open("../data/mapping/foot.json", "r") as f:
    foot = json.load(f)


with open("../data/mapping/position.json", "r") as f:
    positions = json.load(f)

with open("../data/mapping/reasons.json", "r") as f:
    reasons = json.load(f)

with open("../data/mapping/injury_translations.json", "r") as f:
    injury_translations = json.load(f)

def clean_abloese(row):
    try:
        int_val = int(row[ABLOESE])
        row[ABLOESE] = int_val
    except ValueError:
        row[ABLOESE] = 0

def classify_injury(player_rows: dict):
    for i, row in enumerate(player_rows):
        injury = row[VERLETZUNG_COLUMN]
        if injury is None or injury == "":
            row[INJURY_CATEGORY] = None
            continue

        row[INJURY_CATEGORY] = injury_to_category[injury]


def transform_date(date_str):
    """Transforms a date from 'yyyy-mm-dd' to 'dd.mm.yyyy'."""
    year, month, day = date_str.split("-")
    return f"{year}-{month}-{day}"


def determine_row_priority(row, next_row):
    """First return value is prio row, second is less prioritized row"""
    if TRANSFER in row[GRUND]:
        return row, next_row

    if TRANSFER in next_row[GRUND]:
        return next_row, row

    if NEW_COACH in row[GRUND]:
        return row, next_row

    if NEW_COACH in next_row[GRUND]:
        return next_row, row

    if VERLETZUNG_GRUND in row[GRUND] or VERLETZUNG_ENDE_GRUND in row[GRUND]:
        return row, next_row

    if VERLETZUNG_GRUND in next_row[GRUND] or VERLETZUNG_ENDE_GRUND in next_row[GRUND]:
        return next_row, row

    if MARKET_VALUE_REASON in row[GRUND]:
        return row, next_row

    if MARKET_VALUE_REASON in next_row[GRUND]:
        return next_row, row

    # this case shouldnt happen as two regular intevals shouldnt be at the same time
    return row, next_row


def merge_reasons(prio_row, other_row):
    """Returns the priority row but with merged reasons where the lower priority reason is appended after a ','"""
    prio_reason = prio_row[GRUND]
    low_prio_reason = other_row[GRUND]
    prio_row[GRUND] = f"{prio_reason},{low_prio_reason}"
    return prio_row


def merge_values(prio_row, other_row):
    if other_row[GRUND] == VERLETZUNG_GRUND:
        prio_row[VERLETZUNG_COLUMN] = other_row[VERLETZUNG_GRUND]

    return prio_row


def merge_rows(row, next_row):
    prio_row, other_row = determine_row_priority(row, next_row)
    prio_row = merge_reasons(prio_row, other_row)
    prio_row = merge_values(prio_row, other_row)
    return prio_row


def max_date() -> pandas.Timestamp:
    max_timestamp = Timestamp(year=2020, month=6, day=30)
    return max_timestamp

def is_after_june_2020(timestamp_obj: Timestamp) -> bool:
    max_timestamp = Timestamp(year=2020, month=6, day=30)
    if timestamp_obj > max_timestamp:
        return True

    return False

def add_validity_timestamps(player_rows: list):
    # Add new row for end of injury with timestamp at end of injury and new reason VERLETZUNG_ENDE_GRUND
    for i, row in enumerate(player_rows):
        if row[GRUND] == VERLETZUNG_GRUND:
            injury_end_row = row.copy()
            injury_end_row[GRUND] = VERLETZUNG_ENDE_GRUND
            injury_end_date = Timestamp(injury_end_row[VERLETZUNG_ENDE_DATUM])
            if Timestamp(row[VERLETZUNG_ENDE_DATUM]) == row[TIMESTAMP]: #if injury starts and ends on same day
                injury_end_row[TIMESTAMP] = injury_end_date + pd.Timedelta(days=1)
            else:
                injury_end_row[TIMESTAMP] = injury_end_date
            injury_end_row[VERLETZUNG_COLUMN] = None # remove injury because this row signals that the player is no longer injured

            # we only want to account for injuries that end before the 30.06.2020
            if not is_after_june_2020(injury_end_row[TIMESTAMP]):
                player_rows.append(injury_end_row)


    # Sort list again based on timestamp so injury end is at correct place
    sorted_rows = sorted(player_rows, key=lambda x: x["timestamp"])
    # This loop iterates over the data to set the correct values for the injury_end. It will take the previous row and it can never not
    # have a previous row because at least an injury will be before it
    for z, row in enumerate(sorted_rows):
        if row[GRUND] != VERLETZUNG_ENDE_GRUND:# only need to have the rows that are exactly
            continue

        prev_row = sorted_rows[z-1]
        row[VEREIN_ID] = prev_row[VEREIN_ID]
        row[WETTBEWERB_ID] = prev_row[WETTBEWERB_ID]
        row[SAISON_ID] = prev_row[SAISON_ID]
        row[VEREIN] = prev_row[VEREIN]
        row[LIGA] = prev_row[LIGA]
        row[TRAINER_COL] = prev_row[TRAINER_COL]
        row[MARKET_VAL_COL] = prev_row[MARKET_VAL_COL]

    to_be_returned_rows = []
    added_indexes = []
    max_index = len(sorted_rows) - 1
    # This loop is to deduplicate the rows and merge them if needed
    for i, row in enumerate(sorted_rows):
        if i == max_index:
            if i not in added_indexes:
                to_be_returned_rows.append(row)
                added_indexes.append(i)
            continue

        next_index = i+1
        next_row = sorted_rows[next_index]
        if row[TIMESTAMP] != next_row[TIMESTAMP]:
            if i not in added_indexes: # Only add if its not already added (due to being added from being merged previously)
                to_be_returned_rows.append(row)
                added_indexes.append(i)
            continue

        output_row = merge_rows(row, next_row)
        sorted_rows[next_index] = output_row

        # Case: Two rows merge and the i is not in added_indexes ie it has not been previously merged
        if i not in added_indexes:
            to_be_returned_rows.append(output_row)
            added_indexes.append(next_index)
            continue

        # Case: Three or more rows in a row have the same timestamp. Thus we dont insert a new row but its already updated from the merging of rows above
        added_indexes.append(next_index) # add next index so in next iteratoin it wont be added even if its the last


    # This loop iterates over the remaining rows after they have been merged and adds the validity dates.
    # The elements should already be in the correct order and there should not be rows with the same timestamp
    max_index = len(to_be_returned_rows) -1
    for j, row in enumerate(to_be_returned_rows):
        row[VALIDITY_START] = row[TIMESTAMP]
        if j == max_index:
            row[VALIDITY_END] = max_date()
            continue

        next_idx = j + 1
        next_row = to_be_returned_rows[next_idx]
        next_timestamp = next_row[TIMESTAMP]
        row[VALIDITY_END] = next_timestamp - timedelta(days=1)

    return to_be_returned_rows


def classify_market_value(market_value: int, season_id: int) -> str:
    season = season_to_quantiles[str(season_id)]
    if market_value < season[0]:
        return "VERY LOW"

    if market_value < season[1]:
        return "LOW"

    if market_value < season[2]:
        return "MEDIUM"

    if market_value < season[3]:
        return "HIGH"

    return "VERY HIGH"


def categorize_market_value(player_rows: list):
    for _, row in enumerate(player_rows):
        row[MARKET_VALUE_CATEGORY] = classify_market_value(row[MARKET_VAL_COL], row[SAISON_ID])


def age_at_time(rows):
    for _, row in enumerate(rows):
        dob = row[GEBURTSDATUM]
        dob = Timestamp(dob)
        timestamp = row[TIMESTAMP]
        age = timestamp.year - dob.year

        if (timestamp.month, timestamp.day) < (dob.month, dob.day):
            age -= 1

        row[AGE] = age


def transfer_fee_to_last_transfer_fee(player_rows):
    for i, val in enumerate(player_rows):
        if TRANSFER not in val[GRUND]:
            if i == 0:
                val[ABLOESE] = 0
                continue

            val[ABLOESE] = player_rows[i-1][ABLOESE]
            continue

        clean_abloese(val)


def add_injury_if_previous_row_is_injured(player_rows):
    for i, val in enumerate(player_rows):
        if i == 0:
            continue

        if VERLETZUNG_GRUND in val[GRUND]:
            continue

        if VERLETZUNG_ENDE_GRUND not in val[GRUND]:
            val[VERLETZUNG_COLUMN] = player_rows[i-1][VERLETZUNG_COLUMN]
            val[INJURY_CATEGORY] = player_rows[i-1][INJURY_CATEGORY]


def remove_unneccessary_columns(player_rows):
    for _, val in enumerate(player_rows):
         del val['trans_id']
         del val['trainer_id']
         del val['verletzung_start']
         del val['verletzung_end']
         del val['timestamp']



def remove_player_id_with_overlapping_injuries(players):
    with open("../data/injuries/player_ids_with_overlap.json", "r") as f:
        player_ids_with_overlapping_injuries = json.load(f)

    for playerId in player_ids_with_overlapping_injuries:
        try:
            del players[playerId]
        except:
            print("playerId ", playerId, " doesn't exist")


def remove_injuries_with_same_date(players):
    with open("../data/injuries/player_ids_with_same_injury_dates.json", "r") as f:
        player_ids_with_same_date_injuries = json.load(f)

    for playerId in players:
        if playerId not in player_ids_with_same_date_injuries:
            continue

        player_entries = players[playerId]
        rows_with_injury = []
        # get only rows with injuries
        for row in player_entries:
            if row[GRUND] == VERLETZUNG_GRUND:
                rows_with_injury.append(row)

        # sort by timestamp
        sorted_inj_rows = sorted(rows_with_injury, key=lambda x: x["timestamp"])
        # go thru injuries
        to_be_appended_injuries = []
        for i, val in enumerate(sorted_inj_rows):
            if i == 0:
                to_be_appended_injuries.append(val)
                continue

            prev_val = sorted_inj_rows[i - 1]
            if val["timestamp"] == prev_val["timestamp"] and val["verletzung_end"] == prev_val["verletzung_end"]:
                continue

            to_be_appended_injuries.append(val)

        player_entries = [entr for entr in player_entries if entr["grund"] != "verletzung"]
        player_entries = player_entries + to_be_appended_injuries
        players[playerId] = player_entries
        # add the now date wise deduplicated injury rows back in


def remove_injuries_that_are_subinterval(players):
    with open("../data/injuries/player_ids_with_subinterval_date.json", "r") as f:
        player_ids_with_subinterval_injury = json.load(f)

    for playerId in players:
        if playerId not in player_ids_with_subinterval_injury:
            continue

        player_entries = players[playerId]
        rows_with_injury = []
        for row in player_entries:
            if row["grund"] == "verletzung":
                rows_with_injury.append(row)

        # sort by timestamp
        sorted_inj_rows = sorted(rows_with_injury, key=lambda x: x["timestamp"])
        # go thru injuries
        to_be_appended_injuries = []
        for i, val in enumerate(sorted_inj_rows):
            if i == 0:
                to_be_appended_injuries.append(val)
                continue


            # hier über alle vorherigen iterieren, nicht nur direkt davor
            is_subinterval_injury = False
            cur_injury_start = datetime.strptime(val[INJURY_START], "%Y-%m-%d").date()
            cur_injury_end = datetime.strptime(val[INJURY_END], "%Y-%m-%d").date()

            for j in range(i):
                prev_injury_start = datetime.strptime(sorted_inj_rows[j][INJURY_START], "%Y-%m-%d").date()
                prev_injury_end = datetime.strptime(sorted_inj_rows[j][INJURY_END], "%Y-%m-%d").date()

                if cur_injury_start >= prev_injury_start and cur_injury_end <= prev_injury_end:
                    is_subinterval_injury = True
                    continue

            if not is_subinterval_injury:
                to_be_appended_injuries.append(val)

        player_entries = [entr for entr in player_entries if entr["grund"] != "verletzung"]
        player_entries = player_entries + to_be_appended_injuries
        players[playerId] = player_entries


def write_overlapping_injury_player_id_to_json():
    with open(import_input, "r") as f:
        data = json.load(f)
        df = pandas.DataFrame.from_dict(data)

    df_inj = df[df[INJURY_START].notna()]
    # Group by player id
    grouped = df_inj.groupby(["spieler_id"])
    player_ids_with_overlap = set()
    player_ids_with_same_injury_dates = set()
    player_ids_with_subinterval_date = set()
    for player_id, group in grouped:
        player_id = player_id[0]  # get the actual value, not the tuple
        group = group.sort_values(by=INJURY_START, ascending=True).reset_index(drop=True)

        for idx, row in group.iterrows():
            if idx == 0: continue

            prev_injury_start = datetime.strptime(group.iloc[idx - 1][INJURY_START], "%Y-%m-%d").date()
            prev_injury_end = datetime.strptime(group.iloc[idx - 1][INJURY_END], "%Y-%m-%d").date()
            cur_injury_start = datetime.strptime(row[INJURY_START], "%Y-%m-%d").date()
            cur_injury_end = datetime.strptime(row[INJURY_END], "%Y-%m-%d").date()

            # Current injury is after last one has ended
            if cur_injury_start > prev_injury_end:
                continue

            # It cant be before the previous injury, otherwise it would have been sorted there
            # So it can only have started at the same time or after.
            if cur_injury_start == prev_injury_start:

                # If the end is after the previos end, we hav overlapping injuries
                if cur_injury_end > prev_injury_end:
                    player_ids_with_overlap.add(player_id)
                    continue

                if cur_injury_end < prev_injury_end:
                    player_ids_with_subinterval_date.add(player_id)
                    continue

                # Same exact time interval for injuries
                if cur_injury_end == prev_injury_end:
                    player_ids_with_same_injury_dates.add(player_id)
                    continue

            if cur_injury_start > prev_injury_start:
                # Current injury is in a sub interval of the previous injury
                if cur_injury_end <= prev_injury_end:
                    player_ids_with_subinterval_date.add(player_id)
                    continue
                else:
                    player_ids_with_overlap.add(player_id)

    print(f"player with overlaps: {len(player_ids_with_overlap)}")
    print(f"player with same injury date: {len(player_ids_with_same_injury_dates - player_ids_with_overlap)}")
    print(f"player with subintervals: {len(player_ids_with_subinterval_date - player_ids_with_overlap)}")


    with open("../data/injuries/player_ids_with_overlap.json", "w") as f:
        json.dump(list(player_ids_with_overlap), f, ensure_ascii=False)

    with open("../data/injuries/player_ids_with_same_injury_dates.json", "w") as f:
        json.dump(list(player_ids_with_same_injury_dates), f, ensure_ascii=False)

    with open("../data/injuries/player_ids_with_subinterval_date.json", "w") as f:
        json.dump(list(player_ids_with_subinterval_date), f, ensure_ascii=False)

    global total_unwanted_players_due_to_injury
    total_unwanted_players_due_to_injury = set(list(player_ids_with_overlap)) | set(list(player_ids_with_same_injury_dates)) | set(list(player_ids_with_subinterval_date))

    print("successfully written overlapping injury JSON files")

def remove_rows_without_height(player_id_to_rows):
    for _, rows in player_id_to_rows.items():
        rows[:] = [row for row in rows if row["groesse"] != ""]


def remove_bad_injury_rows(player_rows):
    write_overlapping_injury_player_id_to_json()
    remove_player_id_with_overlapping_injuries(player_rows)
    remove_injuries_with_same_date(player_rows)
    remove_injuries_that_are_subinterval(player_rows)

player_id_to_rows = load_data_into_dict_with_player_id(import_input)
remove_bad_injury_rows(player_id_to_rows) # Overlapping injuries and injuries with same date or injuries that are a subinterval of another injury.
remove_rows_without_height(player_id_to_rows) # Remove rows without height

# create market value quantile live
def create_market_value_quantile():
    with open(import_input, "r") as f:
        data = json.load(f)
        df = pd.DataFrame(data)

    df = df[~df["spieler_id"].isin(total_unwanted_players_due_to_injury)]
    df = df[df["grund"] == "regelmaessiger interval"]

    global season_to_quantiles
    for season in range(2010, 2020):
        season_df = df[df["saison_id"] == season]
        market_value_series = season_df["mw"]
        quantiles = np.percentile(market_value_series, [20, 40, 60, 80]).tolist()
        season_to_quantiles[str(season)] = quantiles

create_market_value_quantile()

def remove_injury_end_outside_buli_and_pl(player_rows):
    for idx in range(len(player_rows) - 1, -1, -1):
        # if injur_end happens after a player transfers outside buli and pl the wettbewr_id will be none
        if VERLETZUNG_ENDE_GRUND in player_rows[idx][GRUND] and player_rows[idx][WETTBEWERB_ID] is None and TRANSFER not in player_rows[idx][GRUND]:
            # if we delete the row, we need to extend the validity of the row before
            prev_row = player_rows[idx-1] #can never be first row as injury_end needs prior injury
            prev_row[VALIDITY_END] = player_rows[idx][VALIDITY_END]
            del player_rows[idx]


for player_id in player_id_to_rows:
    player = player_id_to_rows[player_id]
    player = add_validity_timestamps(player)
    classify_injury(player)
    categorize_market_value(player)
    transfer_fee_to_last_transfer_fee(player)
    age_at_time(player)
    player_id_to_rows[player_id] = player
    add_injury_if_previous_row_is_injured(player)
    remove_unneccessary_columns(player)
    remove_injury_end_outside_buli_and_pl(player)

# merge all rows back together
flattened_list = [item for sublist in player_id_to_rows.values() for item in sublist]

# make timestamps json serializable by transforming them into a plain string from the timestamp object
for key, value in enumerate(flattened_list):
    for i, val in value.items():
        if isinstance(val, Timestamp):
            value[i] = val.strftime("%Y-%m-%d")

# for all rows in processed data
coach_name_to_id = {}
coach_id_counter = 1
def add_coach_id(row):
    # when transfer is to null club, skip this
    if row["verein_id"] is None:
        row["coach_id"] = None
        return

    global coach_id_counter
    if row["trainer"] not in coach_name_to_id:
        coach_name_to_id[row["trainer"]] = coach_id_counter
        coach_id_counter += 1

    coach_id = coach_name_to_id[row["trainer"]]
    row["coach_id"] = coach_id


def rename_columns(row):
    row[REASON] = row.pop("grund")
    row["player_id"] = row.pop("spieler_id")
    club_id_value = row.pop("verein_id")
    row["club_id"] = None if club_id_value is None or math.isnan(club_id_value) else int(club_id_value)
    row["league_id"] = row.pop("wettbewerb_id")
    row["season_id"] = row.pop("saison_id")
    row["injury"] = row.pop("verletzung")
    row["last_transfer_fee"] = row.pop(ABLOESE)
    row["first_name"] = row.pop("vorname")
    row["last_name"] = row.pop("name")
    row["pseudonym"] = row.pop("kuenstlername")
    row["position"] = row.pop("spielerposition")
    row["foot"] = row.pop("fussart")
    row["citizenship"] = row.pop("nationalitaet")
    height_string = row.pop("groesse")
    height_float = float(height_string.replace(",", "."))
    row["height"] = height_float
    row["date_of_birth"] = row.pop("geburtsdatum")
    row["club"] = row.pop("verein")
    row["league"] = row.pop("liga")
    row["coach"] = row.pop("trainer")
    row["market_value"] = row.pop("mw")
    row["league_played_matches"] = row.pop("games")


def map_foot(strong_foot) -> str:
    if strong_foot is None:
        return "-"

    return foot[strong_foot]


def translate_reason(original_reason) -> str:
    parts = original_reason.split(",")
    for i, part in enumerate(parts):
        parts[i] = reasons[part]

    return ",".join(parts)


def translate_injury(original_injury) -> str:
    if original_injury is None:
        return None

    return injury_translations[original_injury]


def translate_values(row):
    if row["citizenship"] is not None:
        row["citizenship"] = country_translations[row["citizenship"]]
    else:
        row["citizenship"] = "unknown"
    if row["position"] is not None:
        row["position"] = positions[row["position"]]
    else:
        row["position"] = "unknown"
    row[FOOT] = map_foot(row[FOOT])
    row[REASON] = translate_reason(row[REASON])
    row[INJURY_COLUMN] = translate_injury(row[INJURY_COLUMN])


def cast_floats_to_int(player):
    club_id_val = player["club_id"]
    player["club_id"] = None if club_id_val is None or math.isnan(club_id_val) else int(club_id_val)
    age_val = player["age"]
    player["age"] = None if math.isnan(age_val) else int(age_val)


for key, value in enumerate(flattened_list):
    add_coach_id(value)
    rename_columns(value)
    translate_values(value)
    cast_floats_to_int(value)



###  Add performance data below this
load_dotenv()
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")


def set_performance_data(cursor, row):
    player_id = row["player_id"]
    timestamp = row["validity_start"]
    params = (player_id, timestamp, player_id, timestamp, player_id, timestamp, player_id, timestamp, player_id, timestamp,
        player_id, timestamp)
    cursor.execute(performance_sql, params)
    result = cursor.fetchone()
    league_goals = result[0]
    league_minutes_played = result[1]
    league_games = result[2]
    international_goals = result[3]
    international_minutes_played = result[4]
    international_games = result[5]
    updated_keys = {
        "league_goals": league_goals,
        "league_minutes_played": int(league_minutes_played),
        "league_played_matches": league_games,
        "international_goals": international_goals,
        "international_minutes_played": int(international_minutes_played),
        "international_played_matches": international_games
    }
    row.update(updated_keys)


def set_international_competition(cursor, row):
    club_id = row["club_id"]
    params = (club_id, club_id, row["season_id"])
    cursor.execute(competition_sql, params)
    result = cursor.fetchone()
    updated_key = {}
    int_key_comp = "international_competition"
    if result is not None:
        international_comp = result[0]
        updated_key[int_key_comp] = international_comp
    else:
        updated_key[int_key_comp] = ""

    row.update(updated_key)


def set_injury_data(cursor, row):
    if row[INJURY_COLUMN] is None:
        row.update({MISSED_MATCHES: 0})
        return

    club_id = row[CLUB_ID]
    start_timestamp = row[VALIDITY_START]
    end_timestamp = row[VALIDITY_END]
    params = (club_id, club_id, start_timestamp, end_timestamp)
    cursor.execute(missed_games_sql, params)
    results = cursor.fetchone()
    missed_matches = results[0]
    row.update({MISSED_MATCHES: missed_matches})


def fetch_additional_data(cursor):
    for row in tqdm(flattened_list, desc="Processing rows", unit="row"):
        set_performance_data(cursor, row)
        set_international_competition(cursor, row)
        set_injury_data(cursor, row)


try:
    connection = mysql.connector.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name
    )

    if connection.is_connected():
        print("Successfully connected to the database")
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE();")
        current_database = cursor.fetchone()
        print("Connected to database:", current_database[0])
        fetch_additional_data(cursor)

except Error as e:
    print("Error while connecting to MySQL:", e)

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("MySQL connection closed")


sort_order = [
    "player_id",
    "reason",
    "validity_start",
    "validity_end",
    "first_name",
    "last_name",
    "pseudonym",
    "height",
    "date_of_birth",
    "age",
    "foot",
    "position",
    "citizenship",
    "injury",
    "injury_category",
    "missed_matches",
    "market_value",
    "market_value_category",
    "last_transfer_fee",
    "club",
    "club_id",
    "season_id",
    "league",
    "league_id",
    "international_competition",
    "coach",
    "coach_id",
    "league_played_matches",
    "league_minutes_played",
    "league_goals",
    "international_played_matches",
    "international_minutes_played",
    "international_goals"
]


ordered_data = [OrderedDict((key, item[key]) for key in sort_order) for item in flattened_list]
# df = df.astype(object)
# df = df.mask(df.applymap(lambda x: isinstance(x, float) and np.isnan(x)), None)
# rows_with_nan = [row for _, row in df.iterrows()
#                  if any(isinstance(val, float) and np.isnan(val) for val in row)]
# df["club_id"] = df["club_id"].fillna(-1.0)
print("Writing to JSON...")
with open(output_file, "w") as file:
    json.dump(flattened_list, file, ensure_ascii=False)
### TODO THIS FILE IS NOT REQUIRED ANY MORE. ITS FUNCTIONALITY MOVED TO preprocess.py


import json
import os
import argparse
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from tqdm import tqdm

from sql_statements import performance_sql, competition_sql, transfer_sql, missed_games_sql
from datetime import timedelta


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="test_prod.json", nargs="?",
                    help="What file name to import from /data directory (default: test_prod.json)")

parser.add_argument("--output", default="test_out_prod.json", nargs="?",
                    help="What file name to output to /data directory (default: test_out_prod.json)")


args, unknown = parser.parse_known_args()
import_input = "../data/" + args.input
output_file = "../data/" + args.output

with open(import_input, 'r') as file:
    raw_list = json.load(file)

load_dotenv()
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")


def set_performance_data(cursor, row):
    player_id = row["spieler_id"]
    timestamp = row["timestamp"]
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
        "league_games": league_games,
        "international_goals": international_goals,
        "international_minutes_played": int(international_minutes_played),
        "international_games": international_games
    }
    row.update(updated_keys)


def get_season_id(timestamp) -> int:
    # TODO we dont need it anymore as its in the original data set already
    year_str = timestamp[:4]
    year = int(year_str)
    month_str = timestamp[5:7]
    month = int(month_str)
    if month > 6:
        return year
    else:
        return year - 1


def set_international_competition(cursor, row):
    club_id = row["verein_id"]
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
    if row["grund"] != "verletzung":
        return

    club_id = row["verein_id"]
    injury_start = row["verletzung_start"]
    injury_end = row["verletzung_end"]
    player_id = row["spieler_id"]
    params = (player_id, club_id, injury_start, injury_end)

    cursor.execute(transfer_sql, params)
    results = cursor.fetchone()
    if results is not None:
        club_to = results[0]
        transfer_date = results[1]
        first_team_end_interval = transfer_date - timedelta(days=1)
        one_day_before_transfer = first_team_end_interval.strftime("%Y-%m-%d")

        first_team_missed_matches_params = (club_id, club_id, injury_start, one_day_before_transfer)
        cursor.execute(missed_games_sql, first_team_missed_matches_params)
        results = cursor.fetchone()
        missed_games_first_team = results[0]

        second_team_missed_matches_params = (club_to, club_to, transfer_date, injury_end)
        cursor.execute(missed_games_sql, second_team_missed_matches_params)
        results = cursor.fetchone()
        missed_games_second_team = results[0]

        row.update({"missed_games": missed_games_first_team + missed_games_second_team})
        return

    params = (club_id, club_id, injury_start, injury_end)
    cursor.execute(missed_games_sql, params)
    results = cursor.fetchone()
    missed_games = results[0]
    row.update({"missed_games": missed_games})


def fetch_additional_data(cursor):
    for row in tqdm(raw_list, desc="Processing rows", unit="row"):
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


print("Writing to JSON...")

# write back json
with open(output_file, "w") as outfile:
    json.dump(raw_list, outfile, ensure_ascii=False)

print("Finished writing to JSON")
import argparse
import json
import os
import sys
from datetime import datetime

from scipy.stats import kstwo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myutil import load_data_into_dict_with_player_id

INJURY_START = "verletzung_start"
INJURY_END = "verletzung_end"


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data_bis_19.json", nargs="?",
                    help="What file name to import from /data directory (default: data_bis_19.json)")

args, unknown = parser.parse_known_args()
import_input = "../data/" + args.input



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
        #TODO REMOVE
        player_ids_with_subinterval_injury.append(35247)

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

def remove_bad_injury_rows(player_rows):
    remove_player_id_with_overlapping_injuries(player_rows)
    remove_injuries_with_same_date(player_rows)
    remove_injuries_that_are_subinterval(player_rows)


player_id_to_rows = load_data_into_dict_with_player_id(import_input)
remove_bad_injury_rows(player_id_to_rows)  #


with open("vierinha.json", "w") as f:
    json.dump(player_id_to_rows, f, ensure_ascii=False)
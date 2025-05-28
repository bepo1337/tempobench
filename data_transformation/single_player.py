import json
import argparse
import pandas as pd
import sys
sys.path.append("../.")

parser = argparse.ArgumentParser()
parser.add_argument("--id", default=0, nargs="?", type=int,
                    help="ID of player to fetch (default:0)")

parser.add_argument("--input", default="raw", nargs="?",
                    help="What data to import (raw/processed) (default:raw)")

parser.add_argument("--output", default="<PLAYER_ID>_<input>", nargs="?",
                    help="What file name to output to /data directory (default: <PLAYER_ID>_<input>.json)")


args, unknown = parser.parse_known_args()
if args.input == "raw":
    ending = "20_03_raw.json"
else:
    ending = "real_data.json"

id = args.id
import_input = "../data/" + ending
output_file = "../data/" + str(id) + "_" + args.input + ".json"


with open(import_input, "r") as file:
    data = json.load(file)
    df = pd.DataFrame.from_dict(data)

player_id_column_name = "spieler_id" if args.input == "raw" else "player_id"
player_rows = df[df[player_id_column_name] == id]

with open(output_file, "w") as f:
    player_rows = player_rows.to_dict(orient='records')
    json.dump(player_rows, f, ensure_ascii=False)
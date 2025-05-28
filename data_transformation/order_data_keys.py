# TODO is now also in preprocess and not used in standalone
import json
from collections import OrderedDict

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
    "market_value",
    "market_value_category",
    "transfer_fee",
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


with open("../data/data_processed.json", "r") as f:
    data = json.load(f)

for i, val in enumerate(data):
    val["league_id"] = val.pop("competition_id")

ordered_data = [OrderedDict((key, item[key]) for key in sort_order) for item in data]

with open("../data/data_processed.json", "w") as f:
    json.dump(ordered_data, f, ensure_ascii=False)





import json

import numpy
import pandas as pd

with open("../data/data_bis_19.json", "r") as f:
    data = json.load(f)
    df = pd.DataFrame(data)


with open("../data/injuries/player_ids_with_overlap.json", "r") as f:
    player_ids_unwanted = json.load(f)

# filter for only rows with regelmeasseiger interval and not in player_ids_unwanted

df = df[~df["spieler_id"].isin(player_ids_unwanted)]
df = df[df["grund"] == "regelmaessiger interval"]


season_to_quantiles = {}

for season in range (2010, 2020):
    season_df = df[df["saison_id"] == season]
    market_value_series = season_df["mw"]
    quantiles = numpy.percentile(market_value_series, [20, 40, 60, 80]).tolist()
    season_to_quantiles[season] = quantiles

with open('../data/mapping/season_to_quantiles.json', 'w') as outfile:
    json.dump(season_to_quantiles, outfile)
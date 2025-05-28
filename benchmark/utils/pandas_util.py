import pandas
import pandas as pd
from typing import List
import random
import numpy as np

from benchmark.utils import VALIDITY_START, PLAYER_ID


def grouped_by_player_id_sorted_by_validity_start(df: pd.DataFrame) -> List[pd.DataFrame]:
    player_rows_sorted = []
    for player_id, group in df.groupby(PLAYER_ID):
        sorted_group = group.sort_values(by=VALIDITY_START, ascending=True).reset_index(drop=True)
        player_rows_sorted.append(sorted_group)

    return player_rows_sorted


def get_players_as_list_of_df(df: pd.DataFrame, n=None) -> List[pd.DataFrame]:
    if n is None:
        n = len(df[PLAYER_ID].unique())

    #guarantee order
    unique_ids = df[PLAYER_ID].unique().tolist()
    random.seed(42)
    sampled_ids = random.sample(unique_ids, n)

    grouped = df.groupby(PLAYER_ID)
    return [grouped.get_group(pid) for pid in sampled_ids]

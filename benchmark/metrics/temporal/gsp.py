import json
import os
import tempfile
from typing import List

import pandas as pd
from gsppy.gsp import GSP
import joblib
from benchmark.metrics import BenchmarkMetric
from benchmark.utils import Category, REASON, INJURY, VALIDITY_START, PLAYER_ID, INJURY_CATEGORY, LEAGUE, CLUB

gsp_supp_thresholds = {
    INJURY_CATEGORY: 0.05,
    LEAGUE: 0.03,
    CLUB: 0.01
}

GeneralizedSequentialPatternMetricName = "GeneralizedSequentialPattern"
class GeneralizedSequentialPattern(BenchmarkMetric):
    """Computes the precision, recall and F1 score for GSP mining sequences between the real and synthetic data set."""

    def compute(self, X_real, X_syn) -> dict:
        possible_sequences = len(X_syn) - X_syn[PLAYER_ID].nunique()
        if possible_sequences == 0:
            return {"error": "NO_SEQUENCES"}

        return_dict = {
            INJURY_CATEGORY: self._injury_category_match(X_real, X_syn),
            LEAGUE: self._league_match(X_real, X_syn),
            CLUB: self._club_match(X_real, X_syn)
        }

        self._write_to_tmp_file(X_syn, return_dict)
        return return_dict

    def name(self) -> str:
        return GeneralizedSequentialPatternMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

    def _possible_matches_for_each_length(self, X_real_result) -> dict:
        possible_matches_list = {}
        for i, sequence_length in enumerate(X_real_result):
            # skip sequences with length 1
            if i == 0:
                continue

            list_of_entries = []
            for key, _ in sequence_length.items():
                list_of_entries.append(key)
            possible_matches_list[i + 1] = list_of_entries
        return possible_matches_list

    def _possible_total_matches(self, list_of_possible_matches) -> int:
        possible_matches_count = 0
        for _, possible_matches_for_len in list_of_possible_matches.items():
            possible_matches_count += len(possible_matches_for_len)
        return possible_matches_count

    def _compare_synth_to_real_sequences(self, X_real_result, X_syn_result) -> float:
        list_of_possible_matches_per_length = self._possible_matches_for_each_length(X_real_result)
        possible_matches = self._possible_total_matches(list_of_possible_matches_per_length)
        total_matches = 0
        for i, sequences_found in enumerate(X_syn_result):
            # skip 1 length sequences
            if i == 0: continue


            sequence_length = i + 1
            # skip lengths that are not in the real dataset
            if not sequence_length in list_of_possible_matches_per_length:
                continue

            real_list_for_length = list_of_possible_matches_per_length[sequence_length]
            for sequence in sequences_found:
                if sequence in real_list_for_length:
                    total_matches += 1


        # also calculate length of sequences for syntehtic data for precision
        total_syn = self._possible_total_matches(self._possible_matches_for_each_length(X_syn_result))
        recall = total_matches / possible_matches
        precision = total_matches / total_syn if total_syn else 0
        f1_score = (2*recall*precision) / (recall+precision) if recall > 0 or precision > 0 else 0
        return {"precision": precision, "recall": recall, "f1": f1_score}

    def _injury_category_match(self, X_real, X_syn) -> float:
        try:
            syn_result = self._injury_category_gsp_result(X_syn)
        except ValueError as e:
            error_msg = f"couldn't process: {e}"
            return {
                "error": error_msg,
                "f1": 0,
                "precision": 0,
                "recall": 0
            }

        real_result = self._injury_category_gsp_result(X_real)
        return self._compare_synth_to_real_sequences(real_result, syn_result)

    def _injury_category_gsp_result(self, dataset: pd.DataFrame):
        injury_cat_sequences =  self._create_injury_cat_sequences(dataset)
        supp = gsp_supp_thresholds[INJURY_CATEGORY]
        result = GSP(injury_cat_sequences).search(supp)
        return result

    def _create_injury_cat_sequences(self, dataset: pd.DataFrame) -> List:
        dataset = dataset.copy()
        dataset = dataset[dataset[REASON].str.contains(INJURY) & ~dataset[REASON].str.contains("injury end")]
        grouped_dataset = self._get_player_groups(dataset)
        sequences = [group[INJURY_CATEGORY].tolist() for _, group in grouped_dataset]
        return sequences

    def _league_match(self, X_real, X_syn):
        real_result = self._transfer_league_gsp_result(X_real)
        syn_result = self._transfer_league_gsp_result(X_syn)
        return self._compare_synth_to_real_sequences(real_result, syn_result)

    def _transfer_league_gsp_result(self, dataset: pd.DataFrame):
        league_sequences = self._create_transfer_sequences(dataset, LEAGUE)
        supp = gsp_supp_thresholds[LEAGUE]
        result = GSP(league_sequences).search(supp)
        # print(result)
        return result

    def _club_match(self, X_real, X_syn):
        real_result = self._transfer_club_gsp_result(X_real)
        syn_result = self._transfer_club_gsp_result(X_syn)
        return self._compare_synth_to_real_sequences(real_result, syn_result)

    def _transfer_club_gsp_result(self, dataset: pd.DataFrame):
        league_sequences = self._create_transfer_sequences(dataset, CLUB)
        supp = gsp_supp_thresholds[CLUB]
        result = GSP(league_sequences).search(supp)
        # print(result)
        return result

    def _get_player_groups(self, dataset):
        dataset.sort_values(by=[PLAYER_ID, VALIDITY_START], inplace=True)
        dataset[VALIDITY_START] = pd.to_datetime(dataset[VALIDITY_START])
        sorted_dataset = dataset.sort_values(by=[PLAYER_ID, VALIDITY_START])
        grouped_dataset = sorted_dataset.groupby(PLAYER_ID)
        return grouped_dataset


    def _create_transfer_sequences(self, dataset: pd.DataFrame, column: str) -> List:
        dataset = dataset.copy()
        # Want the first entry for each player and then all transfers
        first_rows = dataset.drop_duplicates(subset=PLAYER_ID, keep='first')
        transfer_rows = dataset[dataset[REASON].str.contains('transfer', case=False, na=False)]
        # drop duplicates in case the first entry is a transfer
        result = pd.concat([first_rows, transfer_rows]).drop_duplicates()
        grouped_dataset = self._get_player_groups(result)
        sequences = [group[column].tolist() for _, group in grouped_dataset]
        return sequences


    def _write_to_tmp_file(self, X_syn, return_dict):
        df_hash = joblib.hash(X_syn)
        filename = f"temp_{df_hash}.json"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, 'w') as f:
            json.dump(return_dict, f)
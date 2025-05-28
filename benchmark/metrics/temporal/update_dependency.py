from collections import defaultdict
import multiprocessing
import time
from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    VALIDITY_START, VALIDITY_END
from benchmark.metrics import BenchmarkMetric
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

UpdateDependencyMetricName = "UpdateDependency"
update_dep_ignore_columns = {PLAYER_ID, VALIDITY_START, VALIDITY_END} # set literal, faster than list for lookup
supp_threshold = 0.2
confidence_threshold = 0.7
antecedents = "antecedents"
consequents = "consequents"

def mine_rules_worker(df, queue):
    try:
        metric = UpdateDependency()
        result = metric._mine_rules(df)
        queue.put(result)
    except Exception as e:
        queue.put(e)


def mine_with_timeout(df, timeout_sec):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=mine_rules_worker, args=(df, queue))
    start = time.time()
    p.start()
    p.join(timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return "timeout"

    result = queue.get()
    if isinstance(result, Exception):
        raise result

    end = time.time()
    print(f"Process took {end - start:.2f} seconds")
    return result

class UpdateDependency(BenchmarkMetric):
    """Computes the precision, recall and F1 score of update dependencies present in the synthetic data set relative to the real data set."""

    def compute(self, X_real, X_syn) -> dict:
        possible_sequences = len(X_syn) - X_syn[PLAYER_ID].nunique()
        if possible_sequences == 0:
            return {"precision": 0, "recall": 0, "error": "NO_SEQUENCES"}


        syn_rules = mine_with_timeout(X_syn, timeout_sec=60)
        if syn_rules == "timeout":
            return {"precision": 0, "recall": 0, "error": "timeout; mining of synthetic rules took too long"}
        real_rules = self._mine_rules(X_real)
        real_ante_to_conseq_map = self._antecedent_to_consequent_map(real_rules)
        precision, recall = self._precision_recall(real_ante_to_conseq_map, syn_rules)
        return {
            "precision": precision,
            "recall": recall
        }

    def name(self) -> str:
        return UpdateDependencyMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

    def _mine_rules(self, dataset: pd.DataFrame):
        start = time.time()

        df = dataset.copy()
        df[VALIDITY_START] = pd.to_datetime(df[VALIDITY_START])
        change_sets = []

        #filter for players that actually have a sequence
        filtered = df.groupby('player_id').filter(lambda x: len(x) >= 2)
        for _, group in filtered.groupby('player_id'):
            group = group.sort_values(VALIDITY_START)
            prev_row = None
            for _, row in group.iterrows():
                if prev_row is not None:
                    changed_cols = [col for col in df.columns if
                                    col not in update_dep_ignore_columns and row[col] != prev_row[col]]
                    if changed_cols:
                        change_sets.append(changed_cols)
                prev_row = row

        encoder = TransactionEncoder()
        encoded = encoder.fit(change_sets).transform(change_sets)
        df_encoded = pd.DataFrame(encoded, columns=encoder.columns_)
        frequent_itemsets = apriori(df_encoded, min_support=supp_threshold, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold)
        end = time.time()
        print(f"Process took {end - start:.2f} seconds")
        return rules

    def _antecedent_to_consequent_map(self, rule_df: pd.DataFrame) -> dict:
        rule_df = rule_df[[antecedents, consequents]]
        possible_rule_matches = defaultdict(set)
        for i, row in rule_df.iterrows():
            ante = row[antecedents]
            conseq = row[consequents]
            possible_rule_matches[ante].add(conseq)

        return possible_rule_matches


    def _precision_recall(self, real_map, syn_df):
        total_real = sum(len(v) for v in real_map.values())
        matched = 0
        total_syn = len(syn_df)
        for _, row in syn_df.iterrows():
            real_consequents = real_map.get(row[antecedents])
            if real_consequents and row[consequents] in real_consequents:
                matched += 1

        recall = matched / total_real
        precision = matched / total_syn if total_syn else 0
        return recall, precision
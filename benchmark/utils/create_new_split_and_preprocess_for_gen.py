import os

from benchmark.utils.preprocess_for_generation import all_df_preprocess_for_generation
from .split_data import create_new_split
import pandas as pd


# execute: "python3 -m benchmark.utils.create_new_split_and_preprocess_for_gen" from root dir
if __name__ == "__main__":
    train_df, test_df = create_new_split()
    dfs = [train_df, test_df]
    all_df_preprocess_for_generation(dfs)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    train_df.to_json(f"{file_dir}/../data/real_data_train_pre.json", orient="records")
    test_df.to_json(f"{file_dir}/../data/real_data_test_pre.json", orient="records")



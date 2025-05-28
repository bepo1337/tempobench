from .loaders import load_real_data
import numpy as np
import json
import os
# run with  "python3 -m benchmark.utils.split_data" from root directory
def create_new_split():
    real_data = load_real_data()

    player_ids = real_data["player_id"].unique()
    seed = 41
    np.random.seed(seed)
    np.random.shuffle(player_ids)
    split_index = int(0.8 * len(player_ids))
    train_player_ids = player_ids[:split_index]
    test_player_ids = player_ids[split_index:]
    train_df = real_data[real_data["player_id"].isin(train_player_ids)]
    test_df = real_data[real_data["player_id"].isin(test_player_ids)]

    file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{file_dir}/../data/real_data_train.json", "w") as f:
        out = train_df.to_json(orient="records")
        f.write(out)

    with open(f"{file_dir}/../data/real_data_test.json", "w") as f:
        out = test_df.to_json(orient="records")
        f.write(out)

    return train_df, test_df


if __name__ == "__main__":
    create_new_split()
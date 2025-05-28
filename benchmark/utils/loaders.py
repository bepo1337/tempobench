import pandas as pd
import json
import os
import importlib.util
import inspect
import pkgutil
import pathlib
from benchmark.metrics import BenchmarkMetric

def load_real_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "../data/real_data.json")
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)

def load_real_data_train() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "../data/real_data_train.json")
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_real_data_test() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "../data/real_data_test.json")
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)
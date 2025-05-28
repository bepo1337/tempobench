import pandas as pd
from benchmark.utils import all_num_col_names, all_cat_col_without_ids

def normalize_num_columns(X_real, X_syn: pd.DataFrame):
    """Normalizes all numerical columns"""
    for col in all_num_col_names:
        min_val = min(X_real[col].min(), X_syn[col].min())
        max_val = max(X_real[col].max(), X_syn[col].max())
        X_real[col] = (X_real[col] - min_val) / (max_val - min_val)
        X_syn[col] = (X_syn[col] - min_val) / (max_val - min_val)

def encode_cat_columns(X_real, X_syn: pd.DataFrame):
    """Encodes categorical attributes into numerical labels. Uses -1 as NULL value"""
    for col in all_cat_col_without_ids:
        unique_vals = list(set(X_real[col].dropna().unique()).union(set(X_syn[col].dropna().unique())))

        if not unique_vals: # shouldnt happen i think
            continue

        cat_map = {val: i for i, val in enumerate(unique_vals)}

        X_real[col] = X_real[col].map(cat_map).fillna(-1).astype(int)
        X_syn[col] = X_syn[col].map(cat_map).fillna(-1).astype(int)

def normalize_and_encode_dfs_in_place(X_real, X_syn: pd.DataFrame):
    """Normalize two DataFrames in place"""
    normalize_num_columns(X_real, X_syn)
    encode_cat_columns(X_real, X_syn)
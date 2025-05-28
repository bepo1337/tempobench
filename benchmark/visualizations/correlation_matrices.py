import os
import json
import hashlib
import tempfile
import pandas
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from benchmark.visualizations.visualization import BenchmarkVisualization
from benchmark.utils import Category, columns_with_ids, columns_with_dates, FIRST_NAME, LAST_NAME, PSEUDONYM

correlation_skip_columns = [
    *columns_with_ids,
    *columns_with_dates,
    FIRST_NAME, LAST_NAME, PSEUDONYM
]

CorrelationVisualizationName = "CorrelationVisualization"
class CorrelationVisualization(BenchmarkVisualization):

    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")
        X_real = X_real.copy()
        X_syn = X_syn.copy()
        # drop columns that we wanna skip
        X_real.drop(columns=correlation_skip_columns, inplace=True)
        X_syn.drop(columns=correlation_skip_columns, inplace=True)
        # Null values should also be included in the correlation as its own category
        X_real.fillna("other", inplace=True)
        X_syn.fillna("other", inplace=True)
        filename_real = "temp_corr_real.json"
        df_hash = joblib.hash(X_syn)
        filename_syn = f"temp_corr_{df_hash}_syn.json"
        temp_path_syn = os.path.join(tempfile.gettempdir(), filename_syn)
        temp_path_real = os.path.join(tempfile.gettempdir(), filename_real)
        if not os.path.exists(temp_path_syn):
            print(f"No temporary file found for hash: {df_hash}. Correlation not included in calculation?")
            return

        syn_corr_matrix = pd.read_json(temp_path_syn)
        real_corr_matrix = pd.read_json(temp_path_real)

        self.visualize_and_store_correlation_matrix(real_corr_matrix, "Real", path)
        self.visualize_and_store_correlation_matrix(syn_corr_matrix, generator_name, path)

    def visualize_and_store_correlation_matrix(self, corr_matrix, title, path):
        fig, ax = plt.subplots(figsize=(6, 5))
        corr_matrix.columns = corr_matrix.index = [str(i) for i in range(len(corr_matrix))]
        sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', cbar=True, annot=False)
        ax.set_title(title)
        plt.tight_layout()
        col_name = ""
        if title == "Real":
            col_name = "Real"

        self.store_matplotlib_png(path, col_name, fig)

    def category(self) -> str:
        return Category.STATISTICAL

    def name(self) -> str:
        return CorrelationVisualizationName

    def description(self) -> str:
        return "Visualizes correlation matrices"
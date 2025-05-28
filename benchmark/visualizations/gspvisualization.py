import os
import json
import hashlib
import tempfile
import pandas
import joblib
import matplotlib.pyplot as plt
import numpy as np

from benchmark.visualizations.visualization import BenchmarkVisualization
from benchmark.utils import Category

GSPVisualizationName = "GSPVisualization"
class GSPVisualization(BenchmarkVisualization):

    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")

        df_hash = joblib.hash(X_syn)
        filename = f"temp_{df_hash}.json"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(temp_path):
            print(f"No temporary file found for hash: {df_hash}. Maybe GSP was not included in the calculation of the metrics?")
            return

        with open(temp_path, 'r') as f:
            metrics = json.load(f)

        categories = list(metrics.keys())
        f1_scores = [metrics[cat]['f1'] for cat in categories]
        precision_scores = [metrics[cat]['precision'] for cat in categories]
        recall_scores = [metrics[cat]['recall'] for cat in categories]

        x = np.arange(len(categories))
        width = 0.25

        fig, ax = plt.subplots()
        ax.bar(x - width, f1_scores, width=width, label='F1 Score')
        ax.bar(x, precision_scores, width=width, label='Precision')
        ax.bar(x + width, recall_scores, width=width, label='Recall')

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=15)
        ax.set_ylabel('Score', fontsize=15)
        ax.set_title(generator_name)
        ax.annotate("GSP evaluation metrics by category",
                    xy=(0.5, -0.15), xycoords='axes fraction', ha='center', va='top', fontsize=15)
        ax.set_ylim(0, 1.05)
        ax.legend()

        plt.tight_layout()
        self.store_matplotlib_png(path, "", fig)

    def category(self) -> str:
        return Category.TEMPORAL

    def name(self) -> str:
        return GSPVisualizationName

    def description(self) -> str:
        return "Visualizes precision, recall, and F1 scores for GSP"
import pandas
import numpy as np
import pandas as pd
from benchmark.utils import REASON, VALIDITY_START, Category, INJURY
from benchmark.visualizations.visualization import BenchmarkVisualization
import matplotlib.pyplot as plt

InjuriesPerMonthVisualizationName = "InjuriesPerMonth"
month = "month"
class InjuriesPerMonthVisualization(BenchmarkVisualization):


    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")
        real_group = self.get_injury_per_month_group(X_real)
        syn_group = self.get_injury_per_month_group(X_syn)

        # want to have relative share in case we dont have exact same data set size
        real_values = np.array(list(real_group.values), dtype=float)
        syn_values = np.array(list(syn_group.values), dtype=float)

        real_rel = real_values / real_values.sum()
        syn_rel = syn_values / syn_values.sum()

        x_labels = list(real_group.keys())
        x = np.arange(len(x_labels))

        width = 0.4  # width  of each bar

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(x - width / 2, real_rel, width=width, label='Real Data', color='blue')
        ax.bar(x + width / 2, syn_rel, width=width, label='Synthetic Data', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=15)
        ax.set_ylabel('Relative share', fontsize=16)
        ax.set_title(generator_name, fontsize=16)
        ax.annotate("Injuries per month", xy=(0.5, -0.15),
                    xycoords='axes fraction', ha='center', va='top', fontsize=16)
        ax.set_ylim(0, 0.4)
        ax.legend(fontsize=16)

        plt.tight_layout()
        self.store_matplotlib_png(path, "", fig)


    def category(self) -> str:
        return Category.TEMPORAL

    def name(self) -> str:
        return InjuriesPerMonthVisualizationName

    def description(self) -> str:
        return "Plots the relative share of months where injuries occurred"

    def get_injury_per_month_group(self, df: pd.DataFrame):
        df_injuries = df[df[REASON].str.contains("injury") & ~df[REASON].str.contains("injury end")].copy()
        df_injuries[month] = df_injuries[VALIDITY_START].str[5:7]
        month_counts = df_injuries[month].value_counts().reindex([f"{i:02d}" for i in range(1, 13)], fill_value=0)
        return month_counts
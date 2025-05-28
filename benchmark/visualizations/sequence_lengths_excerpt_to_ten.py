from collections import defaultdict

import pandas
import numpy as np
import pandas as pd
from benchmark.utils import REASON, VALIDITY_START, Category, PLAYER_ID
from benchmark.visualizations.visualization import BenchmarkVisualization
import matplotlib.pyplot as plt

SequenceLengthsToTenVisualizationName = "SequenceLengthsToTen"
class SequenceLengthsToTenVisualization(BenchmarkVisualization):

    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")
        real_lengths = X_real.groupby(PLAYER_ID).size().value_counts()
        syn_lengths = X_syn.groupby(PLAYER_ID).size().value_counts()

        real_binned = self._bin_counts(real_lengths)
        syn_binned = self._bin_counts(syn_lengths)

        # sort them by the sequence length
        all_bins = sorted(set(real_binned.keys()), key=lambda x: (int(x.split('+')[0]) if '+' in x else int(x)))

        real_values = np.array([real_binned.get(bin_label) for bin_label in all_bins], dtype=float)
        # possibly the syn values dont have values for the bins, assing 0 to them in that case
        syn_values = np.array([syn_binned.get(bin_label, 0) for bin_label in all_bins], dtype=float)

        real_rel = real_values / real_values.sum()
        syn_rel = syn_values / syn_values.sum()

        x = np.arange(len(all_bins))
        width = 0.4

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width / 2, real_rel, width=width, label='Real Data', color='blue')
        ax.bar(x + width / 2, syn_rel, width=width, label='Synthetic Data', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(all_bins, fontsize=14)
        ax.set_ylabel('Relative share', fontsize=16)
        ax.set_title(generator_name, fontsize=16)
        ax.annotate("Relative share of sequence length", xy=(0.5, -0.15), xycoords='axes fraction',
                    ha='center', va='top', fontsize=16)

        ax.set_ylim(0, 0.65)

        ax.legend(fontsize=16)

        plt.tight_layout()
        self.store_matplotlib_png(path, "", fig)


    def category(self) -> str:
        return Category.TEMPORAL

    def name(self) -> str:
        return SequenceLengthsToTenVisualizationName

    def description(self) -> str:
        return "Plots the relative share of sequence lengths for the smallest lengths to 10"

    def _bin_counts(self, value_counts):
        max_bin = 10
        binned = defaultdict(int)
        for length, count in value_counts.items():
            if length >= max_bin:
                binned[f"{max_bin}+"] += count
            else:
                binned[str(length)] += count
        return binned
import pandas
import numpy as np
import pandas as pd
from matplotlib import ticker

from benchmark.utils import REASON, VALIDITY_START, Category
from benchmark.visualizations.visualization import BenchmarkVisualization, color_real, color_syn
import matplotlib.pyplot as plt

TransferFeePerSeasonVisualizationName = "TransferFeePerSeason"
month = "month"
transfer = "transfer"
class TransferFeePerSeasonVisualization(BenchmarkVisualization):

    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")

        real_grouped = self._get_transfer_fees_per_season(X_real, "real")
        synthetic_grouped = self._get_transfer_fees_per_season(X_syn, "synthetic")

        combined = pd.concat([real_grouped, synthetic_grouped])

        colors = {"real": color_real, "synthetic": color_syn}
        markers = {"Bundesliga": "x", "Premier League": "o", None: "-"}

        fig = plt.figure(figsize=(10, 6))

        for (league, origin), data in combined.groupby(["league", "origin"]):
            plt.plot(
                data["season_id"],
                data["last_transfer_fee"] / 1e6, #divide to show million euros instead of raw values
                marker=markers.get(league, "o"),
                color=colors[origin],
                linestyle="-",
                label=f"{league} ({origin})"
            )

        plt.ylim(0, 20)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=14)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='y')

        plt.xlabel("Season", fontsize=15)
        plt.ylabel("Average transfer fee in million €", fontsize=18)
        ax.set_title(generator_name, fontsize=16)
        ax.annotate("Average transfer fee per season by league and data set", xy=(0.5, -0.15), xycoords='axes fraction',
                    ha='center', va='top', fontsize=18)
        plt.legend(title="League + Dataset", fontsize=18)
        plt.grid(True)

        self.store_matplotlib_png(path, "", fig)


    def category(self) -> str:
        return Category.TEMPORAL

    def name(self) -> str:
        return TransferFeePerSeasonVisualizationName

    def description(self) -> str:
        return "Plots the average tranasfer fee per season per league over all seasons"

    def _get_transfer_fees_per_season(self,df: pd.DataFrame, origin: str):
        df_transfers = df[(df["last_transfer_fee"] > 0) & (df["reason"].str.contains("transfer"))].copy()
        df_transfers["origin"] = origin
        grouped = df_transfers.groupby(["season_id", "league", "origin"])["last_transfer_fee"].mean().reset_index()
        return grouped
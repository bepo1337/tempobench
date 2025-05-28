import pandas
import numpy as np
import pandas as pd
from matplotlib import ticker

from benchmark.utils import REASON, VALIDITY_START, Category
from benchmark.visualizations.visualization import BenchmarkVisualization, color_real, color_syn
import matplotlib.pyplot as plt

MarketValuePerSeasonVisualizationName = "MarketValuePerSeason"
class MarketValuePerSeasonVisualization(BenchmarkVisualization):

    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")

        real_grouped = self._get_market_value_per_season(X_real, "real")
        synthetic_grouped = self._get_market_value_per_season(X_syn, "synthetic")

        combined = pd.concat([real_grouped, synthetic_grouped])

        colors = {"real": color_real, "synthetic": color_syn}
        markers = {"Bundesliga": "x", "Premier League": "o", None: "-"}

        fig = plt.figure(figsize=(11, 6))

        for (league, origin), data in combined.groupby(["league", "origin"]):
            plt.plot(
                data["season_id"],
                data["market_value"],
                marker=markers.get(league, "o"),
                color=colors[origin],
                linestyle="-",
                label=f"{league} ({origin})"
            )

        plt.ylim(0, 17)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=14)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='y')

        plt.xlabel("Season", fontsize=15)
        plt.ylabel("Average market value", fontsize=18)
        ax.set_title(generator_name, fontsize=16)
        ax.annotate("Average market value per season by league and dataset",
                    xy=(0.5, -0.15), xycoords='axes fraction', ha='center', va='top', fontsize=16)
        plt.legend(title="League + Dataset", fontsize=18)
        plt.grid(True)

        self.store_matplotlib_png(path, "", fig)


    def category(self) -> str:
        return Category.TEMPORAL

    def name(self) -> str:
        return MarketValuePerSeasonVisualizationName

    def description(self) -> str:
        return "Plots the average market value per season per league over all seasons"

    def _get_market_value_per_season(self, df: pd.DataFrame, origin: str):
        df_transfers = df[df["reason"].str.contains("regular interval")].copy()
        df_transfers["origin"] = origin
        grouped = df_transfers.groupby(["season_id", "league", "origin"])["market_value"].mean().reset_index()
        return grouped
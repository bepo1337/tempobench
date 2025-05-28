import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from benchmark.utils import Category, MARKET_VALUE, LEAGUE_GOALS, AGE, LAST_TRANSFER_FEE, HEIGHT, LEAGUE_PLAYED_MATCHES, \
    LEAGUE_MINUTES_PLAYED, INTERNATIONAL_GOALS, INTERNATIONAL_MINUTES_PLAYED, INTERNATIONAL_PLAYED_MATCHES, \
    MISSED_MATCHES, POSITION, LEAGUE, INTERNATIONAL_COMPETITION, INJURY_CATEGORY, MARKET_VALUE_CATEGORY, FOOT
from benchmark.visualizations.visualization import BenchmarkVisualization, color_real, color_syn

COL_PLOT_DISTRIBUTION = [
    AGE,
    LAST_TRANSFER_FEE,
    HEIGHT,
    LEAGUE_PLAYED_MATCHES,
    LEAGUE_GOALS,
    LEAGUE_MINUTES_PLAYED,
    INTERNATIONAL_GOALS,
    INTERNATIONAL_MINUTES_PLAYED,
    INTERNATIONAL_PLAYED_MATCHES,
    MISSED_MATCHES,
    MARKET_VALUE,
]

COL_PLOT_CATEGORICAL = [
    POSITION,
    LEAGUE,
    INTERNATIONAL_COMPETITION,
    INJURY_CATEGORY,
    MARKET_VALUE_CATEGORY,
    FOOT,
]

COL_INCLUDE_NULL = [
    LEAGUE
]

MarginalDistributionVisualizationName = "MarginalDistribution"

class MarginalDistributionVisualization(BenchmarkVisualization):
    def create(self, X_real: pd.DataFrame, X_syn: pd.DataFrame, path: str, generator_name: str):
        print(f"creating {self.name()}")

        self.create_numerical_plots(X_real, X_syn, path, generator_name)
        self.create_categorical_plots(X_real, X_syn, path, generator_name)


    def category(self) -> str:
        return Category.STATISTICAL

    def name(self) -> str:
        return MarginalDistributionVisualizationName

    def description(self) -> str:
        return (
            "Plots kernel density estimates for numeric features and histograms for subset of categorical features, "
            "comparing real vs. synthetic distributions."
        )

    def create_numerical_plots(self, X_real: pd.DataFrame, X_syn: pd.DataFrame, path: str, generator_name: str):
        for col in COL_PLOT_DISTRIBUTION:
            real_values = X_real[col]
            syn_values = X_syn[col]

            combined_vals = pd.concat([real_values, syn_values])
            cutoff = combined_vals.quantile(0.975)

            # sampled at 200 points between 0 and cutoff
            x_grid = np.linspace(0, cutoff, 200)
            real_kde = gaussian_kde(real_values)
            syn_kde = gaussian_kde(syn_values)
            y_real = real_kde(x_grid)
            y_syn = syn_kde(x_grid)

            fig, ax = plt.subplots(figsize=(8, 4))

            # fill area under curve to see differences easier
            ax.fill_between(x_grid, y_syn, alpha=0.3, color=color_syn, label='Synthetic', zorder=1)
            ax.fill_between(x_grid, y_real, alpha=0.15, color=color_real, label='Real', zorder=2)
            ax.plot(x_grid, y_syn, color=color_syn, zorder=1)
            ax.plot(x_grid, y_real, color=color_real, zorder=2)


            # get proportion of rows that are above cutoff for each dataset
            over_real = (real_values > cutoff).mean()
            over_syn = (syn_values > cutoff).mean()
            # ax.text(0.99, 0.95, f"Real >97.5%: {over_real:.1%}\nSynth >97.5%: {over_syn:.1%}", transform = ax.transAxes,
            #     ha = 'right', va = 'top', fontsize = 9,
            #     bbox = dict(boxstyle='round,pad=0.3', alpha=0.2))

            ax.set_xlabel(col, fontsize=16)
            ax.set_ylabel('Density', fontsize=16)
            ax.set_title(f"{generator_name}", fontsize=16)
            ax.legend(title="Data set", fontsize=16)
            ax.grid(True)
            ax.set_xlim(0, cutoff)
            ax.set_ylim(bottom=0)

            self.store_matplotlib_png(path, f"{col}", fig)
            plt.close(fig)

    def create_categorical_plots(self, X_real: pd.DataFrame, X_syn: pd.DataFrame, path: str, generator_name: str):
        for col in COL_PLOT_CATEGORICAL:
            # compute normalized counts (proportions) per category
            if col in COL_INCLUDE_NULL:
                real_col = X_real[col].fillna('null')
                syn_col = X_syn[col].fillna('null')
            else:
                real_col = X_real[col]
                syn_col = X_syn[col]
            real_counts = real_col.value_counts(normalize=True).sort_index()
            syn_counts = syn_col.value_counts(normalize=True).sort_index()

            # ensure both share the same categories
            all_cats = real_counts.index.union(syn_counts.index)
            real_counts = real_counts.reindex(all_cats, fill_value=0)
            syn_counts = syn_counts.reindex(all_cats, fill_value=0)

            # x locations
            x = np.arange(len(all_cats))
            width = 0.35

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x - width / 2, real_counts.values, width,
                   alpha=0.8, color=color_real, label='Real')
            ax.bar(x + width / 2, syn_counts.values, width,
                   alpha=0.8, color=color_syn, label='Synthetic')

            ax.set_xlabel(col, fontsize=16)
            ax.set_ylabel('Proportion', fontsize=16)
            ax.set_title(generator_name, fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(all_cats)
            ax.legend(fontsize=16)
            ax.grid(axis='y')

            self.store_matplotlib_png(path, f"{col}", fig)
            plt.close(fig)
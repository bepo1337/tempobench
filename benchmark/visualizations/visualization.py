from abc import ABCMeta, abstractmethod
import pandas
import plotly
import os
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

color_real = "blue"
color_syn = "orange"

class BenchmarkVisualization(metaclass=ABCMeta):
    @abstractmethod
    def create(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame, path: str, generator_name: str):
        """
        Create the visualization and save it.
        :param X_real: The original dataset
        :param X_syn: The synthetic dataset
        :param path: The path to save the visualization
        """
        ...

    @abstractmethod
    def category(self) -> str:
        """Name of category (will be a subfolder)"""

    @abstractmethod
    def name(self) -> str:
        """Name of the visualization."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Description of what the visualization represents."""
        ...

    def store_plotly_png(self, path: str, col_name: str, fig: plotly.graph_objects.Figure):
        if not os.path.exists(path):
            os.mkdir(path)

        path_category = f"./{path}/{self.category()}"
        if not os.path.exists(path_category):
            os.mkdir(path_category)

        fig.write_image(f"./{path}/{self.category()}/{self.name()}_{col_name}.png")

    def store_plotly_html(self, path: str, col_name: str, fig: plotly.graph_objects.Figure):
        if not os.path.exists(path):
            os.mkdir(path)

        path_category = f"./{path}/{self.category()}"
        if not os.path.exists(path_category):
            os.mkdir(path_category)

        fig.write_html(f"./{path}/{self.category()}/{self.name()}_{col_name}.html")

    def store_plotly_all(self, path: str, col_name: str, fig: plotly.graph_objects.Figure):
        self.store_plotly_png(path, col_name, fig)
        self.store_plotly_html(path, col_name, fig)

    def store_matplotlib_png(self, path: str, col_name: str, fig: Figure):
        if not os.path.exists(path):
            os.mkdir(path)

        path_category = f"./{path}/{self.category()}"
        if not os.path.exists(path_category):
            os.mkdir(path_category)

        if col_name != "":
            save_path = f"{path_category}/{self.name()}_{col_name}.png"
        else:
            save_path = f"{path_category}/{self.name()}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


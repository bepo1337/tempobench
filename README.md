# TempoBench: Framework for evaluating synthetic temporal tabular data
## Disclaimer 🎓
This project is a master thesis at the [University of Hamburg](https://www.uni-hamburg.de/). It is a prototypical implementation of the benchmarking framework TempoBench presented in the context of this master thesis.
## Install 
To install this application, Python 3.10+ needs to be installed and a virtual environment setup or just install the dependencies globally with: \
`pip install -r requirements.txt` \
For the evaluation for the model in the thesis, Python 3.10 was used.
## Configuration 🔌
A possible configuration can be found in the `benchmarkt_client.py`.
The following can be configured:

| Parameter                  | Type                                                     | Description                                                                    | Required |
|----------------------------|----------------------------------------------------------|--------------------------------------------------------------------------------|----------|
| `X_syn`                    | `pd.DataFrame`                                            | Synthetic dataset to evaluate.                                                 | Yes      |
| `generator_name`           | `str`                                                     | Name of the generator used to create `X_syn`.                                  | Yes      |
| `metrics`                  | `list[Union[str, BenchmarkMetric]]`                       | Metrics to evaluate; can be metric names or instances.                         | No       |
| `metric_categories`        | `list[Category]`                                          | Categories of metrics to include.                      | No       |
| `visualizations`           | `list[Union[str, BenchmarkVisualization]]`                | Visualizations to render; accepts names or instances.                          | No       |
| `visualization_categories` | `list[Category]`                                       | Categories of visualizations to render.                                        | No       |
| `visualization_path`       | `str`                                                     | Path where visualizations will be stored. Default: `"workspace"`.              | No       |
| `output_path`              | `str`                                                     | Path to output JSON with benchmark results. Default: `"benchmark_result.json"`.| No       |

## How to run
Execute the following command after editing the template in `benchmark_client.py` with the desired configuration: \
`python benchmarkt_client.py` or `python3 benchmark_client.py`

## Structure of the repository
The structure of the repository is as follows:

| Name                   | Description                                                          |
|------------------------|----------------------------------------------------------------------|
| `benchmark`            | Code for the TempoBench prototype and all implemented metrics.       |
| `data`                 | Data required for preprocessing the raw Transfermarkt data set.      |
| `data_transformation`  | Preprocessing of the Transfermarkt data set.                         |
| `descriptive_use_in_thesis` | Notebooks and visualizations for descriptive analysis in the thesis. |
| `diagrams`             | Diagrams used in the thesis.                                         |
| `results_for_thesis`   | Benchmarking results that are referred to in the thesis.             |
## Implemented Metrics
The implemented metrics are implemented. More information can be found in the thesis document.

| Category           | Metric                                                                 |
|--------------------|------------------------------------------------------------------------|
| Sanity             | (SAN1) Duplicate rows                                                  |
|                    | (SAN2) Datatype mismatch                                               |
| Domain             | (D1) Manual inspection                                                 |
|                    | (D2) Monotonic increase                                                |
|                    | (D3) Static data remains unchanged                                     |
|                    | (D4) Timestamp min. and max.                                           |
|                    | (D5) Tuple validity overlap                                            |
|                    | (D6) Validity end before start                                         |
|                    | (D7) Maximum validity time                                             |
| Statistical        | (S1) Elementary statistics                                             |
|                    | (S2) Univariate distribution divergence (with visualization V1)       |
|                    | (S4) Column correlation (with visualization V2)                       |
| Temporal           | (T1) Incoming Nearest Neighbor Distance                                |
|                    | (T2) Mean sequence length (with visualization V3)                     |
|                    | (T3) Mean validity                                                     |
|                    | (T4) Intra-class distance                                              |
|                    | (T5) GSP                                                               |
|                    | (T6) Update dependency                                                 |
|                    | (T7) Transition matrix distance                                        |
| Downstream utility | (U1) TSTR-Regression                                                   |
| Visualization      | (V5) Events (Samples) per Month                                        |
|                    | (V6) Injuries per Month                                                |
|                    | (V7) Market Value per Season                                           |
|       | (V8) Transfer Fee per Season                                           |
|                    | (V9) Transfers per Season                                              |
## Add own metric
Own metrics can be implemented using the following interfaces and an instance passed in the configuration of TempoBench. \
```
class BenchmarkMetric(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, X_real: pandas.DataFrame, X_syn: pandas.DataFrame) -> dict:
        """
        Compute the metric.
        :param X_real: The original dataset
        :param X_syn: The synthetic dataset
        :return: Dict that contains all values from this metricd
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        ...

    @abstractmethod
    def category(self) -> str:
        """Category of the metric (see thesis later for what kind there are and validate?)"""
        ...
```
## Add own visualization
Own visualizations can be implemented using the following interfaces and an instance passed in the configuration of TempoBench.

```
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
```
## Repositories for training the generative models
The following repositories were used when training the three generative models used in the thesis:
- TVAE: https://github.com/bepo1337/synthcity
- TabSyn: https://github.com/bepo1337/tabsyn
- PAR: https://github.com/bepo1337/DeepEcho
## Author
* **Benjamin Pöhlmann** - [bepo1337](https://github.com/bepo1337)
import json
from typing import Union
import time
import pandas as pd

from benchmark.metrics import BenchmarkMetric
from benchmark.metrics.domain.monotonic_increase import MonotonicIncrease, MonotonicIncreaseMetricName
from benchmark.metrics.domain.static_data_remains_unchanged import StaticDataRemainsUnchanged,   StaticDataRemainsUnchangedMetricName
from benchmark.metrics.domain.timestamp_min_max import TimestampMinMaxMetricName, TimestampMinMax
from benchmark.metrics.domain.validity_end_before_start import ValidityEndBeforeStart
from benchmark.metrics.domain.validity_max_time import ValiditiyMaxTimeMetricName, ValidityMaxTime
from benchmark.metrics.domain.validity_overlap import ValidityOverlapMetricName, ValidityOverlap
from benchmark.metrics.statistical import *
import benchmark.utils as utils
from benchmark.metrics.statistical.correlation import Correlation, CorrelationMetricName
from benchmark.metrics.statistical.elementary_statistics import ElementaryStatisticsMetricName, ElementaryStatistics
from benchmark.metrics.statistical.jensen_shannon import JensenShannonMetricName, JensenShannon
from benchmark.metrics.sanity.duplicate_rows import DuplicateRowsMetricName, DuplicateRows
from benchmark.metrics.sanity.datatype_mismatch import DataTypeMismatchMetricName, DataTypeMismatch
from benchmark.metrics.temporal.gsp import GeneralizedSequentialPatternMetricName, GeneralizedSequentialPattern
from benchmark.metrics.temporal.incoming_nearest_neighbor import IncomingNearestNeighborMetricName, \
    IncomingNearestNeighbor
from benchmark.metrics.temporal.intra_class_distance import IntraClassDistanceMetricName, IntraClassDistance
from benchmark.metrics.temporal.mean_length_of_sequence import MeanSequenceLengthMetricName, MeanSequenceLength
from benchmark.metrics.temporal.mean_validity import MeanValidity, MeanValidityMetricName
from benchmark.metrics.temporal.transition_matrix_distance import TransitionMatrixDistanceMetricName, \
    TransitionMatrixDistance
from benchmark.metrics.temporal.update_dependency import UpdateDependencyMetricName, UpdateDependency
from benchmark.metrics.utility.train_on_synth_test_on_real import TrainOnSynthTestOnRealMetricName, \
    TrainOnSynthTestOnReal
from benchmark.utils import Category, REASON
from benchmark.visualizations import BenchmarkVisualization
from benchmark.visualizations.correlation_matrices import CorrelationVisualizationName, CorrelationVisualization
from benchmark.visualizations.events_per_month import EventsPerMonthVisualizationName, EventsPerMonthVisualization
from benchmark.visualizations.gspvisualization import GSPVisualization, GSPVisualizationName
from benchmark.visualizations.injury_per_month import InjuriesPerMonthVisualizationName, InjuriesPerMonthVisualization
from benchmark.visualizations.market_value_per_season import MarketValuePerSeasonVisualizationName, \
    MarketValuePerSeasonVisualization
from benchmark.visualizations.marginal_dis_visualization import MarginalDistributionVisualization, \
    MarginalDistributionVisualization, MarginalDistributionVisualizationName
from benchmark.visualizations.sequence_lengths import SequenceLengthsVisualizationName, SequenceLengthsVisualization
from benchmark.visualizations.sequence_lengths_excerpt_to_ten import SequenceLengthsToTenVisualizationName, \
    SequenceLengthsToTenVisualization
from benchmark.visualizations.transfer_fee_per_season import TransferFeePerSeasonVisualizationName, \
    TransferFeePerSeasonVisualization
from benchmark.visualizations.transfer_per_month import TransfersPerMonthVisualizationName, TransferPerMonthVisualization

builtin_metrics = {
    # Domain
    MonotonicIncreaseMetricName: MonotonicIncrease(),
    StaticDataRemainsUnchangedMetricName: StaticDataRemainsUnchanged(),
    TimestampMinMaxMetricName: TimestampMinMax(),
    ValidityEndBeforeStart: ValidityEndBeforeStart(),
    ValiditiyMaxTimeMetricName: ValidityMaxTime(),
    ValidityOverlapMetricName: ValidityOverlap(),
    #Sanity
    DataTypeMismatchMetricName: DataTypeMismatch(),
    DuplicateRowsMetricName: DuplicateRows(),
    #Statistical Fidelity
    CorrelationMetricName: Correlation(),
    ElementaryStatisticsMetricName: ElementaryStatistics(),
    JensenShannonMetricName: JensenShannon(),
    #Temporal Fidelity
    UpdateDependencyMetricName: UpdateDependency(),
    GeneralizedSequentialPatternMetricName: GeneralizedSequentialPattern(),
    IntraClassDistanceMetricName: IntraClassDistance(),
    MeanSequenceLengthMetricName: MeanSequenceLength(),
    MeanValidityMetricName: MeanValidity(),
    TransitionMatrixDistanceMetricName: TransitionMatrixDistance(),
    IncomingNearestNeighborMetricName: IncomingNearestNeighbor(),
    #Downstream Utility
    TrainOnSynthTestOnRealMetricName: TrainOnSynthTestOnReal(),
}

builtin_visualizations = {
    #Fidlity
    MarginalDistributionVisualizationName: MarginalDistributionVisualization(),
    CorrelationVisualizationName: CorrelationVisualization(),
    #Temporal
    TransfersPerMonthVisualizationName: TransferPerMonthVisualization(),
    InjuriesPerMonthVisualizationName: InjuriesPerMonthVisualization(),
    EventsPerMonthVisualizationName: EventsPerMonthVisualization(),
    SequenceLengthsVisualizationName: SequenceLengthsVisualization(),
    SequenceLengthsToTenVisualizationName: SequenceLengthsToTenVisualization(),
    TransferFeePerSeasonVisualizationName: TransferFeePerSeasonVisualization(),
    MarketValuePerSeasonVisualizationName: MarketValuePerSeasonVisualization(),
    GSPVisualizationName: GSPVisualization(),
}

valid_categories = [Category.SANITY, Category.DOMAIN, Category.STATISTICAL, Category.TEMPORAL, Category.UTILITY]


_REASON_ORDER = [
    "injury",
    "injury end",
    "new coach",
    "transfer",
    "regular interval",
    "market value update",
]
_REASON_INDEX = {reason: idx for idx, reason in enumerate(_REASON_ORDER)}

class TempoBench:
    def __init__(self, X_syn: pd.DataFrame,
                 generator_name: str,
                 metrics: list[Union[str, BenchmarkMetric]] = None,
                 metric_categories: list[Category] = None,
                 visualizations: list[Union[str, BenchmarkVisualization]] = [],
                 visualization_categories: list[Category] = None,
                 visualization_path: str = "workspace",
                 output_path: str = "benchmark_result.json"):
        """
        Initialize the TempoBench class.
        :param X_syn: pandas.DataFrame containing the synthesized data
        :param metrics: List of custom metrics from the client and strings from the implemented metrics
        """
        if not isinstance(X_syn, pd.DataFrame):
            raise ValueError("Synthesized dataset must be a Pandas DataFrame")

        self.X_real = utils.load_real_data_train()
        self.X_syn = X_syn[self.X_real.columns] # use same column order because the index matters in some metrics
        self.generator_name = generator_name
        self.metrics = self._validate_metrics(metrics)
        self.metric_categories = self._validate_metric_categories(metric_categories)
        self.visualizations = self._validate_visualizations(visualizations)
        self.visualization_categories = self._validate_visualization_categories(visualization_categories)
        self.visualization_path = visualization_path + "/" + generator_name
        self.output_path = output_path

        self._merge_metrics_and_categories()
        self._merge_visualizations_and_categories()
        self._sort_reason_strings()
        self._validate_if_any_metric_or_vis_passed()
        print("TempoBench initialized")


    def _validate_metrics(self, metrics: list[Union[str, BenchmarkMetric]]) -> list[BenchmarkMetric]:
        valid_metrics = []
        if metrics is None:
            return []

        for metric in metrics:
            if isinstance(metric, str):
                valid_metrics.append(self._load_metric(metric))
            elif isinstance(metric, BenchmarkMetric):
                valid_metrics.append(metric)
            else:
                raise TypeError("Metric must be either a string or a BenchmarkMetric")

        return valid_metrics

    def _load_metric(self, metric: str) -> BenchmarkMetric:
        if metric in builtin_metrics:
            return builtin_metrics[metric]

        raise ValueError(f"Metric '{metric}' is not in builtin metrics'")

    def _validate_visualizations(self, visualizations):
        if visualizations is None:
            return None

        valid_visualizations = []
        for vis in visualizations:
            if isinstance(vis, str):
                valid_visualizations.append(self._load_visualization(vis))
            elif isinstance(vis, BenchmarkVisualization):
                valid_visualizations.append(vis)
            else:
                raise TypeError("Visualization must be either a string or a BenchmarkVisualization")

        return valid_visualizations


    def _validate_metric_categories(self, metric_categories):
        if metric_categories is None or len(metric_categories) == 0:
            return None
        else:
            for cat in metric_categories:
                if cat not in valid_categories:
                    raise ValueError(f"Metric category '{cat}' is not a valid metric category")

            return metric_categories

    def _merge_metrics_and_categories(self):
        if self.metric_categories is None:
            return self.metrics

        for category in self.metric_categories:
            for metric in builtin_metrics.values():
                if metric.category() == category:
                    self.metrics.append(metric)


    def _validate_visualization_categories(self, visualization_categories):
        if visualization_categories is None or len(visualization_categories) == 0:
            return None
        else:
            for cat in visualization_categories:
                if cat not in valid_categories:
                    raise ValueError(f"Visualization category '{cat}' is not a valid visualization category")

            return visualization_categories


    def _merge_visualizations_and_categories(self):
        if self.visualization_categories is None:
            return self.visualizations

        for category in self.visualization_categories:
            for viz in builtin_visualizations.values():
                if viz.category() == category:
                    self.visualizations.append(viz)


    def _load_visualization(self, vis):
        if vis in builtin_visualizations:
            return builtin_visualizations[vis]

        raise ValueError(f"Visualization '{vis}' is not in builtin visualizations'")


    def _validate_if_any_metric_or_vis_passed(self):
        if len(self.metrics) == 0  and len(self.visualizations) == 0:
            raise ValueError("No metrics or visualization passed to the tool. What to benchmark? Please pass in a metric or visualization to continue.")


    def _sort_reason_strings(self):
        self.X_real = sort_reason(self.X_real)
        self.X_syn = sort_reason(self.X_syn)


    def run(self):
        print("running...")
        result = {}
        for metric in self.metrics:
            if metric.category() not in result:
                result[metric.category()] = {}

            start = time.time()
            print(f"{metric.category()}/{metric.name()} computing...")
            result[metric.category()][metric.name()] = metric.compute(X_real=self.X_real, X_syn=self.X_syn)
            duration = time.time() - start
            print(f"{metric.category()}/{metric.name()} finished, execution time: {duration:.2f} seconds")

        if self.visualizations is not None:
            print("computing visualizations...")
            for visualization in self.visualizations:
                visualization.create(self.X_real, self.X_syn, self.visualization_path, self.generator_name)

            print("finished computing visualizations")

        print("finished calculating metrics")
        with open(self.output_path, "w") as f:
            f.write(json.dumps(result, indent=4))
        print(f"sucessfully written to {self.output_path}")


def sort_reason(df: pd.DataFrame) -> pd.DataFrame:
    def _reorder_reason_string(s: str) -> str:
        # split and trim
        tokens = [tok.strip() for tok in s.split(",") if tok.strip()]

        # partition into known vs unknown
        known = [tok for tok in tokens if tok in _REASON_INDEX]
        unknown = [tok for tok in tokens if tok not in _REASON_INDEX]  # if synthetic generator has unknown reasons
        known.sort(key=lambda tok: _REASON_INDEX[tok])
        return ", ".join(known + unknown)


    df = df.copy()
    df[REASON] = df[REASON].apply(_reorder_reason_string)
    return df

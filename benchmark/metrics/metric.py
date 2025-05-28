from abc import ABCMeta, abstractmethod

import pandas

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

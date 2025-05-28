from benchmark import BenchmarkMetric
from benchmark.metrics.domain.monotonic_increase import MonotonicIncrease
from benchmark.metrics.domain.no_tuple_valid_beyond_end_of_june import NoTupleValidBeyondEndOfJune
from benchmark.metrics.domain.static_data_remains_unchanged import StaticDataRemainsUnchanged
from benchmark.metrics.domain.timestamp_min_max import TimestampMinMax
from benchmark.metrics.domain.validity_end_before_start import ValidityEndBeforeStart
from benchmark.metrics.domain.validity_max_time import ValidityMaxTime
from benchmark.metrics.domain.validity_overlap import ValidityOverlap

all_domain_metrics = [
    MonotonicIncrease(),
    NoTupleValidBeyondEndOfJune(),
    StaticDataRemainsUnchanged(),
    TimestampMinMax(),
    ValidityEndBeforeStart(),
    ValidityMaxTime(),
    ValidityOverlap(),
]
all_domain_metrics_dict = {metric.name(): metric for metric in all_domain_metrics}

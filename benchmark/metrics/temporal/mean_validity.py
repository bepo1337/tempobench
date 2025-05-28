from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID, \
    VALIDITY_START, VALIDITY_END
from benchmark.metrics import BenchmarkMetric
import pandas as pd


MeanValidityMetricName = "MeanValidity"

class MeanValidity(BenchmarkMetric):
    """Computes the mean tuple validity and the deviation from the real data set."""

    def compute(self, X_real, X_syn) -> dict:
        real_mean_validity = self._mean_validity(X_real)
        synth_mean_validity = self._mean_validity(X_syn)
        deviation = abs(real_mean_validity - synth_mean_validity) / real_mean_validity

        return {
            "real_mean": real_mean_validity,
            "synth_mean": synth_mean_validity,
            "deviation": deviation
        }

    def name(self) -> str:
        return MeanValidityMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

    def _mean_validity(self, df: pd.DataFrame) -> int:
        df_copy = df.copy()
        df_copy[VALIDITY_START] = pd.to_datetime(df_copy[VALIDITY_START])
        df_copy[VALIDITY_END] = pd.to_datetime(df_copy[VALIDITY_END])
        days_difference = (abs(df_copy[VALIDITY_END] - df_copy[VALIDITY_START])).dt.days
        mean_day_diff = days_difference.mean()
        return mean_day_diff

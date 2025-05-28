from benchmark.utils import Category, column_to_type, ColumnType, COACH_ID, PLAYER_ID, LEAGUE_ID, CLUB_ID, SEASON_ID
from benchmark.metrics import BenchmarkMetric

MeanSequenceLengthMetricName = "MeanSequenceLength"

class MeanSequenceLength(BenchmarkMetric):
    """Computes the mean length of the sequences and the deviation from the real data set."""

    def compute(self, X_real, X_syn) -> dict:
        real_mean = X_real.groupby(PLAYER_ID).size().mean()
        synth_mean = X_syn.groupby(PLAYER_ID).size().mean()

        deviaton = abs(synth_mean - real_mean) / real_mean

        return {
            "real_mean": real_mean,
            "synth_mean": synth_mean,
            "deviation": deviaton
        }

    def name(self) -> str:
        return MeanSequenceLengthMetricName

    def category(self) -> str:
        return Category.TEMPORAL.value

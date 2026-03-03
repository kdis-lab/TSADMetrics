from ....base.Metric import Metric
import numpy as np

class AverageDetectionCount(Metric):
    """
    Calculate average detection count for anomaly detection in time series.

    This metric computes, for each ground-truth anomalous segment, the percentage of points within that segment 
    that are predicted as anomalous. It then averages these percentages across all true anomaly events, 
    providing an estimate of detection coverage per event.

    Reference:
        Implementation based on:
            https://ceur-ws.org/Vol-1226/paper31.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"adc"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    """

    name = "adc"
    binary_prediction = True
    param_schema = {}  

    def __init__(self, **kwargs):
        super().__init__(name="adc", **kwargs)

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        """
        Calculate the average detection count.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The average detection count score.
        """
        starts, ends = self._segment_bounds(y_true)
        if starts.size == 0:
            return 1.0 if np.count_nonzero(y_pred) == 0 else 0.0

        pred_prefix_sum = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(y_pred, dtype=np.int64))
        )
        detected_counts = pred_prefix_sum[ends + 1] - pred_prefix_sum[starts]
        segment_lengths = ends - starts + 1
        return float(np.mean(detected_counts / segment_lengths))

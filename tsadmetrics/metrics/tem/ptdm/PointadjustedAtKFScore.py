from ....base.Metric import Metric
import numpy as np

class PointadjustedAtKFScore(Metric):
    """
    Calculate the Point-adjusted at K% F-score for anomaly detection in time series.

    This metric extends the standard Point-adjusted F-Score by introducing a minimum
    coverage threshold K for each anomalous segment. For every ground-truth anomalous
    segment, if at least K% of its points are predicted as anomalous, the entire segment
    is marked as detected (all points are marked positive in the adjusted prediction).
    Otherwise, the segment remains unchanged, preserving only the correctly detected
    points for point-level evaluation.

    In this way, partial detections below the threshold contribute proportionally at
    the point level, but not at the segment level.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://ojs.aaai.org/index.php/AAAI/article/view/20680

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pakf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (float):
            The minimum percentage of the anomaly that must be detected to consider the anomaly as detected.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """

    name = "pakf"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 0.5,
            "type": float
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="pakf", **kwargs)

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        """
        Calculate the point-adjusted at K% F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The point-adjusted at k F-score, which is the harmonic mean of precision and recall, adjusted by the beta value.
        """
        adjusted_prediction = y_pred.copy()
        starts, ends = self._segment_bounds(y_true)
        k = float(self.params["k"])

        pred_prefix_sum = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(y_pred, dtype=np.int64))
        )
        for start, end in zip(starts, ends):
            segment_length = int(end - start + 1)
            correct_points = int(pred_prefix_sum[end + 1] - pred_prefix_sum[start])
            if correct_points > 0 and (correct_points / segment_length) >= k:
                adjusted_prediction[start : end + 1] = 1

        tp = int(np.count_nonzero(np.logical_and(adjusted_prediction, y_true)))
        true_positive_total = int(np.count_nonzero(y_true))
        pred_positive_total = int(np.count_nonzero(adjusted_prediction))
        fn = true_positive_total - tp
        fp = pred_positive_total - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0.0 or recall == 0.0:
            return 0.0

        beta = float(self.params["beta"])
        beta2 = beta * beta
        return ((1.0 + beta2) * precision * recall) / (beta2 * precision + recall)

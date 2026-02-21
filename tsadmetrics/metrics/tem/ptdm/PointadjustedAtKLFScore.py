from ....base.Metric import Metric
import numpy as np

class PointadjustedAtKLFScore(Metric):
    """
    Calculates the point-adjusted at K% F-score with a tolerance window `l` for anomaly detection in time series. 
    It extends the standard F-score by applying a temporal adjustment: if at least K% of the points within an 
    anomalous segment are predicted as anomalous, the entire segment is considered correctly detected. 
    Additionally, a tolerance window of size `l` is applied around each predicted positive point, so that points 
    within ±l positions of a true anomaly are also counted as detected, making the metric more robust to small 
    temporal misalignments.


    Reference:
        For more information, see the original paper:
            https://www.mdpi.com/1424-8220/24/16/5310

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"paklf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (float):
            The minimum percentage of the anomaly that must be detected to consider the anomaly as detected.
        l (int):
            The tolerance window (in time steps) applied around each predicted positive point.
            Points within ±l distance of a true anomaly are treated as correctly detected.
            Default is 0, meaning no tolerance.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """

    name = "paklf"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 0.5,
            "type": float
        },
        "l": {
            "default": 1,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="paklf", **kwargs)

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        """
        Calculate the point-adjusted at K% F-score with a tolerance window l.

        Parameters:
            y_true (np.array):
                Ground truth binary labels for the time series.
            y_pred (np.array):
                Predicted binary labels for the time series.

        Returns:
            float: The adjusted F-score considering k% detection and tolerance l.
        """
        adjusted_prediction = y_pred.copy()
        starts, ends = self._segment_bounds(y_true)
        k = float(self.params["k"])
        l = int(self.params["l"])

        pred_prefix_sum = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(y_pred, dtype=np.int64))
        )

        for start, end in zip(starts, ends):
            segment_length = int(end - start + 1)
            correct_points = int(pred_prefix_sum[end + 1] - pred_prefix_sum[start])

            if correct_points == 0 or (correct_points / segment_length) < k:
                continue

            if l < 0:
                continue

            seg_pred_positions = np.flatnonzero(y_pred[start : end + 1]) + int(start)
            if seg_pred_positions.size == 0:
                continue

            segment_mask = np.zeros(segment_length, dtype=np.bool_)
            for pos in seg_pred_positions:
                left = max(int(start), int(pos) - l)
                right = min(int(end), int(pos) + l)
                if left <= right:
                    segment_mask[left - int(start) : right - int(start) + 1] = True

            adjusted_prediction[start : end + 1] = np.logical_or(
                adjusted_prediction[start : end + 1], segment_mask
            )

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

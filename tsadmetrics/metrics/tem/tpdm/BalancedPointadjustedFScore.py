from ....base.Metric import Metric
import numpy as np
from math import ceil

class BalancedPointadjustedFScore(Metric):
    """
    Balanced point-adjusted F-score for anomaly detection in time series.
    This metric modifies the standard F-score by applying a temporal adjustment: for each ground-truth
    anomalous segment, if at least one point is predicted as anomalous, the entire segment is considered
    correctly detected. Additionally, for each false positive point at time t, all points in the
    range [t - floor(w/2), t + floor(w/2)] are set to 1, which can generate additional false positives.
    The adjusted predictions are then compared to the ground-truth labels using the standard
    point-wise F-score formula.


    Reference:
        For more information, see the original paper:
            https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10890568
        

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"bpaf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        beta (float):
            Weight factor for recall in the F-score (default = 1.0).
        w (int):
            Temporal window size for expanding each true positive point.
    """

    name = "bpaf"
    binary_prediction = True
    param_schema = {
        "beta": {"default": 1.0, "type": float},
        "w": {"default": 1, "type": int}
    }

    def __init__(self, **kwargs):
        super().__init__(name="bpaf", **kwargs)

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        y_true_bool = np.asarray(y_true, dtype=np.bool_)
        adjusted_prediction = np.asarray(y_pred, dtype=np.bool_).copy()
        w = self.params['w']
        half_w = ceil(w / 2)
        n = int(y_true_bool.size)

        starts, ends = self._segment_bounds(y_true_bool)
        for start, end in zip(starts, ends):
            adjusted_prediction[start:end + 1] = bool(np.any(adjusted_prediction[start:end + 1]))

        fp_indices = np.flatnonzero(adjusted_prediction & ~y_true_bool)
        if fp_indices.size > 0:
            window_size = (2 * half_w + 1) if half_w >= 0 else 0
            if half_w < 0 or (fp_indices.size * window_size) <= n:
                for t in fp_indices:
                    start = max(0, int(t) - half_w)
                    end = min(n - 1, int(t) + half_w)
                    if start <= end:
                        adjusted_prediction[start:end + 1] = True
            else:
                left = np.maximum(0, fp_indices - half_w)
                right = np.minimum(n - 1, fp_indices + half_w)
                valid = left <= right
                if np.any(valid):
                    diff = np.zeros(n + 1, dtype=np.int32)
                    np.add.at(diff, left[valid], 1)
                    np.add.at(diff, right[valid] + 1, -1)
                    expanded = np.cumsum(diff[:-1], dtype=np.int64) > 0
                    adjusted_prediction[expanded] = True

        tp = int(np.sum(adjusted_prediction & y_true_bool, dtype=np.int64))
        total_pred = int(np.sum(adjusted_prediction, dtype=np.int64))
        total_true = int(np.sum(y_true_bool, dtype=np.int64))
        fp = total_pred - tp
        fn = total_true - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0

        beta = self.params['beta']
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

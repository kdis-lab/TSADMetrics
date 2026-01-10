from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise
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

    def _compute(self, y_true, y_pred):
        adjusted_prediction = y_pred.copy()
        w = self.params['w']
        half_w = ceil(w / 2)

        for start, end in full_series_to_segmentwise(y_true):
            if np.any(adjusted_prediction[start:end + 1]):
                adjusted_prediction[start:end + 1] = 1
            else:
                adjusted_prediction[start:end + 1] = 0

        fp_indices = np.where((adjusted_prediction == 1) & (y_true == 0))[0]
        for t in fp_indices:
            start = max(0, t - half_w)
            end = min(len(y_true) - 1, t + half_w)
            adjusted_prediction[start:end + 1] = 1
        print(adjusted_prediction)
        tp = np.sum(adjusted_prediction * y_true)
        fp = np.sum(adjusted_prediction * (1 - y_true))
        fn = np.sum((1 - adjusted_prediction) * y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0

        beta = self.params['beta']
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

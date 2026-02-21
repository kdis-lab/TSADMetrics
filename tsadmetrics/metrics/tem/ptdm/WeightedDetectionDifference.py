from ....base.Metric import Metric
import numpy as np
from ....utils.functions_counting_metrics import counting_method

class WeightedDetectionDifference(Metric):
    """
    Calculate weighted detection difference for anomaly detection in time series.

    For each true anomaly segment, each point in the segment is assigned a weight based on a
    Gaussian function centered at the segment’s midpoint: points closer to the center receive higher
    weights, which decay with distance according to the standard deviation sigma. These weights form
    the basis for scoring both correct detections and false alarms.

    WS (Weighted Sum) is defined as the sum of Gaussian weights for all predicted anomaly points that
    fall within any true anomaly segment (extended by delta time steps at the ends).
    WF (False Alarm Weight) is the sum of Gaussian weights for all predicted anomaly points that do
    not overlap any true anomaly segment (within the same extension).

    The final score is:

        .. math::
            \\text{WDD} = \\text{WS} - \\text{WF} \\cdot \\text{FA}

    Where:

    - WS: 
        Sum of Gaussian weights for all predicted anomaly points that fall 
        within any true anomaly segment (extended by delta time steps at the ends).
    - WF:
        Sum of Gaussian weights for all predicted anomaly points that do not 
        overlap any true anomaly segment (within the same extension).
    - FA (False Anomaly):
        Number of predicted anomaly segments that do not overlap any true anomaly segment
        even within a k-step tolerance window around true points.

    Reference:
        For more information, see the original paper:
            https://acta.sapientia.ro/content/docs/evaluation-metrics-for-anomaly-detection.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"wdd"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (int):
            The maximum number of time steps within which an anomaly must be predicted to be considered detected.
    """
    name = "wdd"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 5,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="wdd", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the weighted detection difference.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The weighted detection difference.
        """
        if np.count_nonzero(y_pred) == 0:
            return 0.0

        def gaussian(dt, tmax):
            if dt < tmax:
                return 1 - dt / tmax
            else:
                return -1

        tmax = int(len(y_true))
        ones_indices = np.where(y_true == 1)[0]
        y_modified = y_true.astype(float).copy()

        zero_mask = y_true == 0
        if ones_indices.size == 0:
            y_modified[zero_mask] = -1.0
        else:
            n = y_true.size
            dist = np.full(n, tmax, dtype=np.int64)
            last_one = -1
            for i in range(n):
                if y_true[i] == 1:
                    last_one = i
                    dist[i] = 0
                elif last_one != -1:
                    dist[i] = i - last_one

            last_one = -1
            for i in range(n - 1, -1, -1):
                if y_true[i] == 1:
                    last_one = i
                elif last_one != -1:
                    right_dist = last_one - i
                    if right_dist < dist[i]:
                        dist[i] = right_dist

            zero_dist = dist[zero_mask]
            y_modified[zero_mask] = np.where(
                zero_dist < tmax,
                1.0 - (zero_dist / tmax),
                -1.0,
            )

        pred_mask = y_pred == 1
        wf = float(np.sum(y_modified[pred_mask]))
        ws = float(np.sum(y_modified[np.logical_not(pred_mask)]))

        _, _, _, fa = counting_method(y_true, y_pred, int(self.params["k"]))

        return ws - wf * fa

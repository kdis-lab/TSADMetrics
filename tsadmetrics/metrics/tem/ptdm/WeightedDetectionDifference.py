from ....base.Metric import Metric
import numpy as np

class WeightedDetectionDifference(Metric):
    """
    Calculate weighted detection difference for anomaly detection in time series.

    For each true anomaly segment, each point in the segment is assigned a Gaussian weight centered at
    the segment midpoint, so detections close to the center receive higher reward than detections near
    the edges. Predicted anomaly points that do not belong to any true anomaly segment are penalized
    according to their temporal distance to the nearest true anomaly: the farther they are, the larger
    their false-alarm penalty, capped by the tolerance parameter k.

    WS (Weighted Sum) is defined as the sum of Gaussian weights for all predicted anomaly points that
    fall inside true anomaly segments. WF (False Alarm Weight) is defined as the average penalty weight
    assigned to false positive predicted points, and FA is the number of those false positive points.

    The final score is:

        .. math::
            \\text{WDD} = \\text{WS} - \\text{WF} \\cdot \\text{FA}

    Where:

    - WS: 
        Sum of Gaussian weights for all predicted anomaly points that fall 
        within any true anomaly segment (extended by delta time steps at the ends).
    - WF:
        Average penalty weight of predicted anomaly points that do not overlap any true anomaly segment.
    - FA (False Anomaly):
        Number of predicted anomaly points that do not overlap any true anomaly segment.

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

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _true_positive_weights(self, y_true):
        n = int(y_true.size)
        weights = np.zeros(n, dtype=float)
        starts, ends = self._segment_bounds(y_true)
        for start, end in zip(starts, ends):
            center = (float(start) + float(end)) / 2.0
            sigma = max((float(end - start + 1)) / 2.0, 1.0)
            indices = np.arange(start, end + 1, dtype=float)
            weights[start : end + 1] = np.exp(-0.5 * ((indices - center) / sigma) ** 2)
        return weights

    @staticmethod
    def _nearest_true_distances(y_true):
        n = int(y_true.size)
        if np.count_nonzero(y_true) == 0:
            return np.full(n, n, dtype=np.int64)

        dist = np.full(n, n, dtype=np.int64)
        last_true = -1
        for idx in range(n):
            if y_true[idx] == 1:
                last_true = idx
                dist[idx] = 0
            elif last_true != -1:
                dist[idx] = idx - last_true

        last_true = -1
        for idx in range(n - 1, -1, -1):
            if y_true[idx] == 1:
                last_true = idx
            elif last_true != -1:
                right_dist = last_true - idx
                if right_dist < dist[idx]:
                    dist[idx] = right_dist

        return dist

    def _false_positive_penalties(self, y_true):
        n = int(y_true.size)
        if np.count_nonzero(y_true) == 0:
            return np.ones(n, dtype=float)

        penalties = np.zeros(n, dtype=float)
        false_mask = y_true == 0
        distances = self._nearest_true_distances(y_true)
        tolerance = max(int(self.params["k"]), 1)
        penalties[false_mask] = np.minimum(1.0, distances[false_mask] / tolerance)
        return penalties

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
        y_true_bool = np.asarray(y_true, dtype=np.bool_)
        y_pred_bool = np.asarray(y_pred, dtype=np.bool_)

        if not np.any(y_pred_bool):
            return 0.0

        true_weights = self._true_positive_weights(y_true_bool)
        false_penalties = self._false_positive_penalties(y_true_bool)

        true_positive_mask = np.logical_and(y_pred_bool, y_true_bool)
        false_positive_mask = np.logical_and(y_pred_bool, np.logical_not(y_true_bool))

        ws = float(np.sum(true_weights[true_positive_mask]))
        fa = int(np.count_nonzero(false_positive_mask))
        wf = float(np.mean(false_penalties[false_positive_mask])) if fa > 0 else 0.0

        return ws - wf * fa

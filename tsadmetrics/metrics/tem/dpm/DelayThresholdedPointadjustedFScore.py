from ....base.Metric import Metric
import numpy as np


class DelayThresholdedPointadjustedFScore(Metric):
    """
    Calculate delay thresholded point-adjusted F-score for anomaly detection in time series.

    This metric is based on the standard F-score, but applies a temporal adjustment 
    to the predictions before computing it. Specifically, for each ground-truth anomalous segment, 
    if at least one point within the first k time steps of the segment is predicted as anomalous, 
    all points in the segment are marked as correctly detected. The adjusted predictions are then 
    compared to the ground-truth labels using the standard point-wise F-score formulation.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://doi.org/10.1145/3292500.3330680

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"dtpaf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (int):
            Maximum number of time steps from the start of an anomaly segment within which a prediction must occur 
            for the segment to be considered detected.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """
    name = "dtpaf"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 1,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="dtpaf", **kwargs)

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        """
        Calculate the delay thresholded point-adjusted F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The computed delay thresholded point-adjusted F-score.
        """
        k = int(self.params["k"])

        starts, ends = self._segment_bounds(y_true)
        y_true_bool = np.asarray(y_true, dtype=np.bool_)
        tp = 0
        if k == 1:
            for start, end in zip(starts, ends):
                if y_pred[start] == 1:
                    tp += int(end - start + 1)
        elif k > 1:
            pred_prefix_sum = np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(y_pred, dtype=np.int64))
            )
            for start, end in zip(starts, ends):
                window_end = min(start + k - 1, end)
                if (pred_prefix_sum[window_end + 1] - pred_prefix_sum[start]) > 0:
                    tp += int(end - start + 1)

        true_positives_total = int(np.count_nonzero(y_true_bool))
        fn = true_positives_total - tp
        pred_positives_total = int(np.count_nonzero(y_pred))
        pred_on_true = int(np.count_nonzero(np.logical_and(y_pred, y_true_bool)))
        fp = pred_positives_total - pred_on_true

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0.0 or recall == 0.0:
            return 0.0

        beta = float(self.params["beta"])
        beta2 = beta * beta
        return ((1.0 + beta2) * precision * recall) / (beta2 * precision + recall)

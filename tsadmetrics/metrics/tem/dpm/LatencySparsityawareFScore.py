from ....base.Metric import Metric
import numpy as np
from ....utils.functions_latency_sparsity_aware import calc_twseq

class LatencySparsityawareFScore(Metric):
    """
    Calculate latency and sparsity aware F-score for anomaly detection in time series.
    
    This metric is based on the standard F-score, but applies a temporal adjustment 
    to the predictions before computing it. Specifically, for each ground-truth anomalous segment, 
    all points in the segment are marked as correctly detected only after the first true positive 
    is predicted within that segment. This encourages early detection by delaying credit for correct 
    predictions until the anomaly is initially detected. Additionally, to reduce the impact of 
    scattered false positives, predictions are subsampled using a sparsity factor n, so that 
    only one prediction is considered every n time steps. The adjusted predictions are then used 
    to _compute the standard point-wise F-score.

    Reference:
        Implementation based on:
            https://dl.acm.org/doi/10.1145/3447548.3467174
        For more information, see the original paper:
            https://doi.org/10.1145/3447548.3467174

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"lsaf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        ni (int):
            The batch size used in the implementation to handle latency and sparsity.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """
    name = "lsaf"
    binary_prediction = True
    param_schema = {
        "ni": {
            "default": 1,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="lsaf", **kwargs)

    @staticmethod
    def _segment_bounds(series_bool):
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        """
        Calculate the latency and sparsity aware F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The latency and sparsity aware F-score, which is the harmonic mean 
            of precision and recall, adjusted by the beta value.
        """

        if np.count_nonzero(y_pred) == 0:
            return 0.0

        tw = int(self.params["ni"])
        if tw == 1:
            actual = np.asarray(y_true, dtype=np.bool_)
            predict = np.asarray(y_pred, dtype=np.bool_)
            adjusted = predict.copy()
            starts, ends = self._segment_bounds(actual)

            for start, end in zip(starts, ends):
                seg_pred = predict[start : end + 1]
                if np.any(seg_pred):
                    first_idx = int(start + np.flatnonzero(seg_pred)[0])
                    adjusted[first_idx : end + 1] = True
                else:
                    adjusted[start : end + 1] = False

            tp = float(np.count_nonzero(np.logical_and(adjusted, actual)))
            fp = float(np.count_nonzero(np.logical_and(adjusted, np.logical_not(actual))))
            fn = float(np.count_nonzero(np.logical_and(np.logical_not(adjusted), actual)))
            precision = tp / (tp + fp + 0.00001)
            recall = tp / (tp + fn + 0.00001)
        else:
            _, precision, recall, _, _, _, _, _ = calc_twseq(
                y_pred,
                y_true,
                normal=0,
                threshold=0.5,
                tw=tw,
            )

        if precision == 0 or recall == 0:
            return 0.0

        beta = float(self.params["beta"])
        beta2 = beta * beta
        return ((1.0 + beta2) * precision * recall) / (beta2 * precision + recall)

from ....base.Metric import Metric
import numpy as np

class TimeTolerantFScore(Metric):
    """
    Calculate time tolerant F-score for anomaly detection in time series.
    This metric is based on the standard F-score, but applies a temporal adjustment 
    to the predictions before computing it. Specifically, a predicted anomalous point is considered 
    a true positive if it lies within a temporal window of size :math:`{\\tau}` around any ground-truth anomalous point. 
    This allows for small temporal deviations in the predictions to be tolerated. The adjusted predictions are then used 
    to _compute the standard point-wise F-Score.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://arxiv.org/abs/2008.05788

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"ttf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        t (int):
            The time tolerance parameter.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """
    name = "ttf"
    binary_prediction = True
    param_schema = {
        "t": {
            "default": 5,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="ttf", **kwargs)

    @staticmethod
    def _dilate_binary_mask(mask, radius):
        n = int(mask.size)
        if n == 0 or radius < 0:
            return np.zeros(n, dtype=np.bool_)

        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return np.zeros(n, dtype=np.bool_)

        left = np.maximum(0, indices - radius)
        right = np.minimum(n - 1, indices + radius)

        diff = np.zeros(n + 1, dtype=np.int32)
        np.add.at(diff, left, 1)
        np.add.at(diff, right + 1, -1)
        return np.cumsum(diff[:-1], dtype=np.int64) > 0

    def _compute(self, y_true, y_pred):
        """
        Calculate the time tolerant F-score (optimized version).
        """
        t = int(self.params["t"])
        beta = float(self.params["beta"])

        true_anomalies = np.asarray(y_true, dtype=np.bool_)
        predictions = np.asarray(y_pred, dtype=np.bool_)

        pred_dilated = self._dilate_binary_mask(predictions, t)
        true_dilated = self._dilate_binary_mask(true_anomalies, t)

        tp_recall = int(np.sum(true_anomalies & pred_dilated, dtype=np.int64))
        fn_recall = int(np.sum(true_anomalies & ~pred_dilated, dtype=np.int64))
        recall = tp_recall / (tp_recall + fn_recall) if (tp_recall + fn_recall) > 0 else 0.0

        tp_precision = int(np.sum(predictions & true_dilated, dtype=np.int64))
        fp_precision = int(np.sum(predictions & ~true_dilated, dtype=np.int64))
        precision = tp_precision / (tp_precision + fp_precision) if (tp_precision + fp_precision) > 0 else 0.0

        if precision == 0 and recall == 0:
            return 0.0

        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

from ...base.Metric import Metric
import numpy as np

class PointwiseFScore(Metric):
    """
    Point-wise F-score for anomaly detection in time series.

    This metric computes the classical F-score without considering temporal context,
    treating each time-series point independently. It balances precision and recall
    according to the configurable parameter `beta`.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pwf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
    """

    name = "pwf"
    binary_prediction = True
    param_schema = {
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the PointwiseFScore metric.

        Parameters:
            **kwargs:
                Additional keyword arguments passed to the base `Metric` class.
                These may include configuration parameters such as `beta`.
        """
        super().__init__(name="pwf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Compute the point-wise F-score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series.
                Values must be 0 (normal) or 1 (anomaly).
            y_pred (np.ndarray):
                Predicted binary labels for the time series.
                Values must be 0 (normal) or 1 (anomaly).

        Returns:
            float:
                The computed point-wise F-score.
                Returns 0 if either precision or recall is 0.
        """
        tp = np.sum(y_pred * y_true)
        fp = np.sum(y_pred * (1 - y_true))
        fn = np.sum((1 - y_pred) * y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0
        
        beta = self.params['beta']
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

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
        tp = int(np.count_nonzero(np.logical_and(y_true, y_pred)))
        if tp == 0:
            return 0.0

        true_positives_total = int(np.count_nonzero(y_true))
        pred_positives_total = int(np.count_nonzero(y_pred))
        fn = true_positives_total - tp
        fp = pred_positives_total - tp

        beta = float(self.params["beta"])
        beta2 = beta * beta
        numerator = (1.0 + beta2) * tp
        denominator = numerator + beta2 * fn + fp
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

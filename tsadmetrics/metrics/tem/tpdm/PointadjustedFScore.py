from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise

class PointadjustedFScore(Metric):
    """
    Point-adjusted F-score for anomaly detection in time series.

    This metric modifies the standard F-score by applying a temporal adjustment
    to the predictions:
    
    - For each ground-truth anomalous segment, if at least one point is predicted
      as anomalous, all points in that segment are considered correctly detected.
    - The adjusted predictions are then compared to the ground-truth labels using
      the standard point-wise F-score formula.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://doi.org/10.1145/3178876.3185996
        

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"paf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        beta (float):
            Weight factor for recall in the F-score (default = 1.0).
            
    """

    name = "paf"
    binary_prediction = True
    param_schema = {
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the PointadjustedFScore metric.

        Parameters:
            **kwargs:
                Optional keyword arguments passed to the base `Metric` class.
                Supported parameter:
                    - beta (float): Weight factor for recall in the F-score.
        """
        super().__init__(name="paf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Compute the point-adjusted F-score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series (0 = normal, 1 = anomaly).
            y_pred (np.ndarray):
                Predicted binary labels for the time series (0 = normal, 1 = anomaly).

        Returns:
            float:
                The computed point-adjusted F-score. Returns 0 if either precision or
                recall is 0.
        """
        adjusted_prediction = y_pred.copy()

        for start, end in full_series_to_segmentwise(y_true):
            if np.any(adjusted_prediction[start:end + 1]):
                adjusted_prediction[start:end + 1] = 1
            else:
                adjusted_prediction[start:end + 1] = 0

        tp = np.sum(adjusted_prediction * y_true)
        fp = np.sum(adjusted_prediction * (1 - y_true))
        fn = np.sum((1 - adjusted_prediction) * y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0

        beta = self.params['beta']
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_pointwise

class TemporalDistance(Metric):
    """
    Calculate temporal distance for anomaly detection in time series.

    This metric computes the sum of the distances from each labelled anomaly point to
    the closest predicted anomaly point, and from each predicted anomaly point to the
    closest labelled anomaly point. 

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://sciendo.com/article/10.2478/ausi-2019-0008

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"td"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        distance (int):
            The distance type parameter for the temporal distance calculation.
            - 0: Euclidean distance
            - 1: Squared Euclidean distance
    """
    name = "td"
    binary_prediction = True
    param_schema = {
        "distance": {
            "default": 0,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="td", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the temporal distance.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The temporal distance.
        """

        def _dist(a, b):
            dist = 0
            for pt in a:
                if len(b) > 0:
                    dist += min(abs(b - pt))
                else:
                    dist += len(y_true)
            return dist

        y_true_pw = np.array(full_series_to_pointwise(y_true))
        y_pred_pw = np.array(full_series_to_pointwise(y_pred))

        distance = self.params['distance']
        if distance == 0:
            return _dist(y_true_pw, y_pred_pw) + _dist(y_pred_pw, y_true_pw)
        elif distance == 1:
            return _dist(y_true_pw, y_pred_pw)**2 + _dist(y_pred_pw, y_true_pw)**2
        else:
            raise ValueError(f"Distance {distance} not supported")

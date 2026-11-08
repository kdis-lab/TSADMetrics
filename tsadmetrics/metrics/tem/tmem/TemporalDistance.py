from ....base.Metric import Metric
import numpy as np

class TemporalDistance(Metric):
    """
    Calculate temporal distance for anomaly detection in time series.

    This metric computes two directional sums of one-dimensional absolute Euclidean
    distances between anomalous points and their nearest counterpart. The selected
    distance mode combines those sums either directly or by squaring each directional
    sum.

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
        distance (str):
            The distance formulation used for the calculation. Accepted values are
            ``"absolute"`` for absolute Euclidean distance and ``"squared"`` for
            squared Euclidean distance applied to each directional sum. The default
            is ``"absolute"``.
    """
    name = "td"
    binary_prediction = True
    param_schema = {
        "distance": {
            "default": "absolute",
            "type": str
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="td", **kwargs)
        if self.params["distance"] not in {"absolute", "squared"}:
            raise ValueError(
                "Parameter 'distance' must be either 'absolute' or 'squared'."
            )

    @staticmethod
    def _sum_min_dist(a, b, series_len):
        if a.size == 0:
            return 0
        if b.size == 0:
            return int(series_len) * int(a.size)

        total = 0
        j = 0
        b_size = int(b.size)
        for pt in a:
            while j + 1 < b_size and abs(int(b[j + 1]) - int(pt)) <= abs(int(b[j]) - int(pt)):
                j += 1
            total += abs(int(b[j]) - int(pt))
        return total

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
        y_true_pw = np.flatnonzero(y_true)
        y_pred_pw = np.flatnonzero(y_pred)

        distance = self.params['distance']
        d_true_to_pred = self._sum_min_dist(y_true_pw, y_pred_pw, len(y_true))
        d_pred_to_true = self._sum_min_dist(y_pred_pw, y_true_pw, len(y_true))
        if distance == "absolute":
            return d_true_to_pred + d_pred_to_true
        elif distance == "squared":
            return d_true_to_pred**2 + d_pred_to_true**2
        else:
            raise ValueError(f"Distance {distance} not supported")

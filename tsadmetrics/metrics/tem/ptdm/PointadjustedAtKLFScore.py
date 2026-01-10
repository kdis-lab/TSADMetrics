from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise, full_series_to_pointwise, pointwise_to_full_series

class PointadjustedAtKLFScore(Metric):
    """
    Calculates the point-adjusted at K% F-score with a tolerance window `l` for anomaly detection in time series. 
    It extends the standard F-score by applying a temporal adjustment: if at least K% of the points within an 
    anomalous segment are predicted as anomalous, the entire segment is considered correctly detected. 
    Additionally, a tolerance window of size `l` is applied around each predicted positive point, so that points 
    within ±l positions of a true anomaly are also counted as detected, making the metric more robust to small 
    temporal misalignments.


    Reference:
        For more information, see the original paper:
            https://www.mdpi.com/1424-8220/24/16/5310

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"paklf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (float):
            The minimum percentage of the anomaly that must be detected to consider the anomaly as detected.
        l (int):
            The tolerance window (in time steps) applied around each predicted positive point.
            Points within ±l distance of a true anomaly are treated as correctly detected.
            Default is 0, meaning no tolerance.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """

    name = "paklf"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 0.5,
            "type": float
        },
        "l": {
            "default": 1,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="paklf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the point-adjusted at K% F-score with a tolerance window l.

        Parameters:
            y_true (np.array):
                Ground truth binary labels for the time series.
            y_pred (np.array):
                Predicted binary labels for the time series.

        Returns:
            float: The adjusted F-score considering k% detection and tolerance l.
        """

        adjusted_prediction = full_series_to_pointwise(y_pred).tolist()
        l = self.params['l']

        for start, end in full_series_to_segmentwise(y_true):
            correct_points = 0
            for i in range(start, end + 1):
                if i in adjusted_prediction:
                    correct_points += 1

            if correct_points / (end + 1 - start) >= self.params['k']:
                for i in range(start, end + 1):
                    if y_true[i] == 1 and y_pred[i]==1:
                        for j in range(max(start, i - l), min(end, i + l) + 1):
                            adjusted_prediction.append(j)


        adjusted_prediction = pointwise_to_full_series(np.sort(np.unique(adjusted_prediction)), len(y_true))
        tp = np.sum(adjusted_prediction * y_true)
        fp = np.sum(adjusted_prediction * (1 - y_true))
        fn = np.sum((1 - adjusted_prediction) * y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0

        beta = self.params['beta']
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)


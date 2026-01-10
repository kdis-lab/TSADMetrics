from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise


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

        adjusted_prediction = y_pred.copy()
        k = self.params["k"]

        for start, end in full_series_to_segmentwise(y_true):
            anomaly_adjusted = False
            for i in range(start, min(start + k, end + 1)):
                if adjusted_prediction[i] == 1:
                    adjusted_prediction[start:end + 1] = 1
                    anomaly_adjusted = True
                    break
            if not anomaly_adjusted:
                adjusted_prediction[start:end + 1] = 0

        tp = np.sum(adjusted_prediction * y_true)
        fp = np.sum(adjusted_prediction * (1 - y_true))
        fn = np.sum((1 - adjusted_prediction) * y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0

        beta = self.params["beta"]
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

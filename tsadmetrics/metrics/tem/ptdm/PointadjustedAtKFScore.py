from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise, full_series_to_pointwise, pointwise_to_full_series

class PointadjustedAtKFScore(Metric):
    """
    Calculate the Point-adjusted at K% F-score for anomaly detection in time series.

    This metric extends the standard Point-adjusted F-Score by introducing a minimum
    coverage threshold K for each anomalous segment. For every ground-truth anomalous
    segment, if at least K% of its points are predicted as anomalous, the entire segment
    is marked as detected (all points are marked positive in the adjusted prediction).
    Otherwise, the segment remains unchanged, preserving only the correctly detected
    points for point-level evaluation.

    In this way, partial detections below the threshold contribute proportionally at
    the point level, but not at the segment level.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://ojs.aaai.org/index.php/AAAI/article/view/20680

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pakf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (float):
            The minimum percentage of the anomaly that must be detected to consider the anomaly as detected.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """

    name = "pakf"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 0.5,
            "type": float
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="pakf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the point-adjusted at K% F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The point-adjusted at k F-score, which is the harmonic mean of precision and recall, adjusted by the beta value.
        """

        adjusted_prediction = full_series_to_pointwise(y_pred).tolist()
        for start, end in full_series_to_segmentwise(y_true):
            correct_points = 0
            for i in range(start, end + 1):
                if i in adjusted_prediction:
                    correct_points += 1
                    if correct_points / (end + 1 - start) >= self.params['k']:
                        for j in range(start, end + 1):
                            adjusted_prediction.append(j)
                        break

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

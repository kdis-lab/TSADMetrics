from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise

class CompositeFScore(Metric):
    """
    Composite F-score for anomaly detection in time series.

    This metric combines aspects of the point-wise F-score and the segment-wise
    F-score. It is defined as the harmonic mean of point-wise precision and
    segment-wise recall. Using point-wise precision ensures that false positives
    are properly penalized, a limitation often found in purely segment-wise
    metrics.

    Reference:
        For more information, see the original paper:
            https://doi.org/10.1109/TNNLS.2021.3105827

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"cf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    """

    name = "cf"
    binary_prediction = True
    param_schema = {
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the CompositeFScore metric.

        Parameters:
            **kwargs:
                Optional keyword arguments passed to the base `Metric` class.
                Supported parameter:
                    - beta (float): Weight factor for recall in the F-score.
        """
        super().__init__(name="cf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Compute the composite F-score.

        The score is computed as:
            F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

        where:
            - precision is computed point-wise.
            - recall is computed segment-wise, meaning a segment is counted as
              correctly detected if any point within it is predicted as anomalous.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series (0 = normal, 1 = anomaly).
            y_pred (np.ndarray):
                Predicted binary labels for the time series.

        Returns:
            float:
                The composite F-score. Returns 0 if either precision or recall is 0.
        """
        tp = np.sum(y_pred * y_true)
        fp = np.sum(y_pred * (1 - y_true))

        tp_sw = 0
        fn_sw = 0
        for gt_anomaly in full_series_to_segmentwise(y_true):
            found = False
            for i_index in range(gt_anomaly[0], gt_anomaly[1] + 1):
                if y_pred[i_index] == 1:
                    tp_sw += 1
                    found = True
                    break
            if not found:
                fn_sw += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp_sw / (tp_sw + fn_sw) if (tp_sw + fn_sw) > 0 else 0

        if precision == 0 or recall == 0:
            return 0

        beta = self.params['beta']
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

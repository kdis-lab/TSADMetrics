from ...base.Metric import Metric
import numpy as np
from ...utils.functions_auc import precision_recall_curve

class PointwiseAucPr(Metric):
    """
    Point-wise Area Under the Precision-Recall Curve (AUC-PR) for anomaly detection.

    This metric computes the standard Area Under the Precision-Recall Curve (AUC-PR)
    in a **point-wise manner**. Each time-series data point is treated independently
    when calculating precision and recall, making this suitable for anomaly detection tasks
    where anomalies are labeled at the individual point level.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pw_auc_pr"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.
    """
    
    name = "pw_auc_pr"
    binary_prediction = False
    def __init__(self, **kwargs):
        """
        Initialize the PointwiseAucPr metric.

        Parameters:
            **kwargs:
                Additional keyword arguments passed to the base `Metric` class.
                These may include configuration parameters or overrides.
        """
        super().__init__(name="pw_auc_pr", **kwargs)

    def _compute(self, y_true, y_anomaly_scores):
        """
        Compute the point-wise AUC-PR score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series.
                Values must be 0 (normal) or 1 (anomaly).
            y_anomaly_scores (np.ndarray):
                Continuous anomaly scores assigned to each point in the series.

        Returns:
            float:
                The computed point-wise AUC-PR score.
        """

        precision, recall, _ = precision_recall_curve(y_true, y_anomaly_scores)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

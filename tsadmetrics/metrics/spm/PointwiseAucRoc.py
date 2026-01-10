from ...base.Metric import Metric
import numpy as np
from ...utils.functions_auc import roc_curve, auc

class PointwiseAucRoc(Metric):
    """
    Point-wise Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for anomaly detection.

    This metric computes the standard Area Under the ROC Curve (AUC-ROC)
    in a **point-wise manner**. Each time-series data point is treated independently
    when calculating true positives, false positives, and false negatives.
    It is widely used to evaluate the ability of anomaly scoring functions
    to distinguish between normal and anomalous points.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pw_auc_roc"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.
    """

    name = "pw_auc_roc"
    binary_prediction = False

    def __init__(self, **kwargs):
        """
        Initialize the PointwiseAucRoc metric.

        Parameters:
            **kwargs:
                Additional keyword arguments passed to the base `Metric` class.
                These may include configuration parameters or overrides.
        """
        super().__init__(name="pw_auc_roc", **kwargs)

    def _compute(self, y_true, y_anomaly_scores):
        """
        Compute the point-wise AUC-ROC score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series.
                Values must be 0 (normal) or 1 (anomaly).
            y_anomaly_scores (np.ndarray):
                Continuous anomaly scores assigned to each point in the series.

        Returns:
            float:
                The computed point-wise AUC-ROC score.
        """
        fpr, tpr, _ = roc_curve(y_true, y_anomaly_scores)
        return auc(fpr, tpr)

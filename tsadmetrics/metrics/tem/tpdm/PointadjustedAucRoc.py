from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise
from ....utils.functions_auc import auc

class PointadjustedAucRoc(Metric):
    """
    Point-adjusted Area Under the ROC Curve (AUC-ROC) for anomaly detection in time series.

    Unlike standard point-wise AUC-ROC, this metric applies **point-adjusted evaluation**:
    
    - Each anomalous segment in `y_true` is considered correctly detected if **at least one
      point** within that segment is predicted as anomalous.
    - Once a segment is detected, all its points are marked as detected in the adjusted
      predictions.
    - Adjusted predictions are then used to compute true positive rate (TPR) and false
      positive rate (FPR) at multiple thresholds to construct the ROC curve.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pa_auc_pr"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.

    Raises:
        ValueError:
            If `y_true` and `y_anomaly_scores` have mismatched lengths.
        TypeError:
            If inputs are not array-like.
    """

    name = "pa_auc_roc"
    binary_prediction = False
    param_schema = {}

    def __init__(self, **kwargs):
        """
        Initialize the PointadjustedAucRoc metric.

        Parameters:
            **kwargs:
                Optional keyword arguments passed to the base `Metric` class.
        """
        super().__init__(name="pa_auc_roc", **kwargs)

    def _compute_point_adjusted(self, y_true, y_pred):
        """
        Apply point-adjustment and compute TPR and FPR.

        For each ground-truth anomalous segment, if any point is predicted as
        anomalous, the entire segment is marked as detected.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels (0 = normal, 1 = anomaly).
            y_pred (np.ndarray):
                Binary predictions (0 = normal, 1 = anomaly).

        Returns:
            tuple[float, float]:
                - tpr (float): True positive rate.
                - fpr (float): False positive rate.
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
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + (len(y_true) - np.sum(y_true) - fp)) if (fp + (len(y_true) - np.sum(y_true) - fp)) > 0 else 0.0
        return tpr, fpr

    def _compute(self, y_true, y_anomaly_scores):
        """
        Compute the point-adjusted AUC-ROC score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series.
            y_anomaly_scores (np.ndarray):
                Continuous anomaly scores assigned to each point.

        Returns:
            float:
                The point-adjusted AUC-ROC score.
        """
        unique_thresholds = np.unique(y_anomaly_scores)
        unique_thresholds = np.sort(unique_thresholds)[::-1]  # descending

        tprs, fprs = [], []

        for threshold in unique_thresholds:
            y_pred_binary = (y_anomaly_scores >= threshold).astype(int)
            tpr, fpr = self._compute_point_adjusted(y_true, y_pred_binary)
            tprs.append(tpr)
            fprs.append(fpr)

        # Add endpoints for ROC curve
        tprs = [0.0] + tprs + [1.0]
        fprs = [0.0] + fprs + [1.0]

        # Sort by FPR to ensure monotonic increasing for AUC calculation
        sorted_indices = np.argsort(fprs)
        fprs_sorted = np.array(fprs)[sorted_indices]
        tprs_sorted = np.array(tprs)[sorted_indices]

        return auc(fprs_sorted, tprs_sorted)

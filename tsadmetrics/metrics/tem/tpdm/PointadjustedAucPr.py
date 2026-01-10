from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise
from ....utils.functions_auc import auc

class PointadjustedAucPr(Metric):
    """
    Point-adjusted Area Under the Precision-Recall Curve (AUC-PR) for anomaly detection.

    Unlike the standard point-wise AUC-PR, this variant uses a point-adjusted evaluation:
    
    - Each anomalous segment in `y_true` is considered correctly detected if **at least
      one point** within that segment is predicted as anomalous.
    - Once a segment is detected, all its points are marked as detected in the adjusted
      prediction.
    
    This adjustment accounts for the fact that detecting any part of an anomalous
    segment is often sufficient in practice.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pa_auc_pr"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.
    """

    name = "pa_auc_pr"
    binary_prediction = False
    param_schema = {}

    def __init__(self, **kwargs):
        """
        Initialize the PointadjustedAucPr metric.

        Parameters:
            **kwargs:
                Optional keyword arguments passed to the base `Metric` class.
        """
        super().__init__(name="pa_auc_pr", **kwargs)

    def _compute_point_adjusted(self, y_true, y_pred):
        """
        Apply point-adjustment to predictions and compute precision/recall.

        For each ground-truth anomalous segment, if any point is predicted as
        anomalous, the entire segment is marked as detected.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels (0 = normal, 1 = anomaly).
            y_pred (np.ndarray):
                Binary predictions (0 = normal, 1 = anomaly).

        Returns:
            tuple[float, float]:
                - precision (float): Adjusted precision score.
                - recall (float): Adjusted recall score.
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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision, recall

    def _compute(self, y_true, y_anomaly_scores):
        """
        Compute the point-adjusted AUC-PR score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series.
            y_anomaly_scores (np.ndarray):
                Continuous anomaly scores assigned to each point.

        Returns:
            float:
                The point-adjusted AUC-PR score.
        """
        unique_thresholds = np.unique(y_anomaly_scores)
        unique_thresholds = np.sort(unique_thresholds)[::-1]  # descending

        precisions, recalls = [], []

        for threshold in unique_thresholds:
            y_pred_binary = (y_anomaly_scores >= threshold).astype(int)
            precision, recall = self._compute_point_adjusted(y_true, y_pred_binary)
            precisions.append(precision)
            recalls.append(recall)

        # Add endpoints for PR curve
        recalls = [0.0] + recalls + [1.0]
        precisions = [1.0] + precisions + [0.0]

        # Sort by recall (increasing order)
        sorted_indices = np.argsort(recalls)
        recalls_sorted = np.array(recalls)[sorted_indices]
        precisions_sorted = np.array(precisions)[sorted_indices]

        return auc(recalls_sorted, precisions_sorted)

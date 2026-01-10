from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise

class SegmentwiseFScore(Metric):
    """
    Segment-wise F-score for anomaly detection in time series.

    This metric computes the F-score at the segment level rather than point-wise.
    Each contiguous segment of anomalies in the ground truth is treated as a unit.
    - True positive (TP): at least one predicted anomaly within a ground-truth segment.
    - False negative (FN): no predicted anomaly in a ground-truth segment.
    - False positive (FP): predicted segment with no overlap with any ground-truth segment.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://doi.org/10.1145/3219819.3219845

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"swf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    
    Parameters:
        beta (float): Weight of precision in the harmonic mean.
                      Default is 1.0 (balanced F1-score).
    """

    name = "swf"
    binary_prediction = True
    param_schema = {
        "beta": {"default": 1.0, "type": float}
    }

    def __init__(self, **kwargs):
        """
        Initialize the SegmentwiseFScore metric.

        Parameters:
            **kwargs: Additional parameters matching param_schema.
        """
        super().__init__(name="swf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Compute the segment-wise F-score.

        Parameters:
            y_true (np.array): Ground truth binary labels.
            y_pred (np.array): Predicted binary labels.

        Returns:
            float: Segment-wise F-score.
        """
        beta = self.params["beta"]

        # Count true positives and false negatives per ground-truth segment
        tp, fn = 0, 0
        for gt_segment in full_series_to_segmentwise(y_true):
            if np.any(y_pred[gt_segment[0]:gt_segment[1]+1]):
                tp += 1
            else:
                fn += 1

        # Count false positives per predicted segment
        fp = 0
        for pred_segment in full_series_to_segmentwise(y_pred):
            if not np.any(y_true[pred_segment[0]:pred_segment[1]+1]):
                fp += 1

        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Return F-beta score
        if precision == 0 or recall == 0:
            return 0.0
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

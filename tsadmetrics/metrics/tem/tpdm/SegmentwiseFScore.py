from ....base.Metric import Metric
import numpy as np

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

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    @staticmethod
    def _count_segments_with_overlap(a_starts, a_ends, b_starts, b_ends):
        if a_starts.size == 0 or b_starts.size == 0:
            return 0

        i, j = 0, 0
        overlap_count = 0
        n_a = int(a_starts.size)
        n_b = int(b_starts.size)

        while i < n_a and j < n_b:
            if a_ends[i] < b_starts[j]:
                i += 1
            elif b_ends[j] < a_starts[i]:
                j += 1
            else:
                overlap_count += 1
                i += 1

        return overlap_count

    def _compute(self, y_true, y_pred):
        """
        Compute the segment-wise F-score.

        Parameters:
            y_true (np.array): Ground truth binary labels.
            y_pred (np.array): Predicted binary labels.

        Returns:
            float: Segment-wise F-score.
        """
        beta = float(self.params["beta"])
        beta2 = beta * beta

        gt_starts, gt_ends = self._segment_bounds(y_true)
        pred_starts, pred_ends = self._segment_bounds(y_pred)

        tp = self._count_segments_with_overlap(gt_starts, gt_ends, pred_starts, pred_ends)
        fn = int(gt_starts.size) - tp
        pred_overlap = self._count_segments_with_overlap(pred_starts, pred_ends, gt_starts, gt_ends)
        fp = int(pred_starts.size) - pred_overlap

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision == 0.0 or recall == 0.0:
            return 0.0
        return (1.0 + beta2) * precision * recall / (beta2 * precision + recall)

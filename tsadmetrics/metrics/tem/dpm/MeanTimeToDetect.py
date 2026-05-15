from ....base.Metric import Metric
import numpy as np
import warnings

class MeanTimeToDetect(Metric):
    """
    Calculate mean time to detect for anomaly detection in time series.
    
    This metric quantifies the average delay between the start of each
    ground-truth anomaly segment and the first predicted anomaly point that
    appears at or after that start index.

    For each ground-truth anomaly segment, let i be the index where the
    segment starts. The implementation searches for the first index j such
    that :math:`{j \\geq i}` and the model predicts an anomaly at j. The predicted
    anomaly point j does not need to fall inside the corresponding ground-truth
    anomaly segment.

    The detection delay for that event is defined as:

    .. math::
        \\Delta t = j - i

    The returned MTTD is computed as the sum of the detection delays found for
    matched ground-truth anomaly starts, divided by the total number of
    ground-truth anomaly segments.

    Ground-truth anomaly segments with no predicted anomaly point at or after
    their start do not add any delay to the numerator, but they are still included
    in the denominator. Therefore, missed anomaly segments are not penalized with
    an increased delay.

    If no predicted anomaly points are found at or after any
    ground-truth anomaly start, the metric emits a warning and returns ``0.0`` as
    a fallback value for an undefined/non-informative delay. In this edge case,
    ``0.0`` should be interpreted as the absence of computable detections, not as
    immediate detection.

    Reference:
        For more information, see the original paper:
            https://dl.acm.org/doi/10.1145/3691338

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"mttd"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    """
    name = "mttd"
    binary_prediction = True
    def __init__(self, **kwargs):
        super().__init__(name="mttd", **kwargs)

    @staticmethod
    def _segment_starts(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0)
        return np.flatnonzero(transitions == 1)

    def _compute(self, y_true, y_pred):
        """
        Calculate the mean time to detect.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The mean time to detect.
        """
        starts = self._segment_starts(y_true)
        if starts.size == 0:
            return 0.0

        pred_idxs = np.flatnonzero(y_pred)
        pred_count = int(pred_idxs.size)
        t_sum = 0
        matched_segment_count = 0
        for a in starts:
            idx = int(np.searchsorted(pred_idxs, a, side="left"))
            if idx < pred_count:
                t_sum += int(pred_idxs[idx]) - int(a)
                matched_segment_count += 1
        if matched_segment_count == 0:
            warnings.warn(
                "No predicted anomaly points were found at or after any ground-truth anomaly start. "
                "Returning 0.0 for MeanTimeToDetect.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        return t_sum / len(starts)

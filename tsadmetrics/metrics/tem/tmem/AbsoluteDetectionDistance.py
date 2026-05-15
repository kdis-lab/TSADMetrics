from ....base.Metric import Metric
import numpy as np
import warnings
class AbsoluteDetectionDistance(Metric):
    """
    Calculate absolute detection distance for anomaly detection in time series.

    This metric computes, for each predicted anomaly point that overlaps a ground-truth anomaly
    segment, the distance from that point to the temporal center of the corresponding segment,
    normalized by the segment length. It then averages those normalized distances over the
    predicted anomaly points that correctly overlap a real anomaly event.

    
    If there are no predicted anomaly points, the metric returns ``0.0``
    because there are no detections for which a distance to the center of a
    ground-truth anomaly segment can be computed. In this edge case, ``0.0``
    should be interpreted as the absence of computable detections, not as a
    perfectly centered detection.

    Reference:
        For more information, see the original paper:
            https://ceur-ws.org/Vol-1226/paper31.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"add"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    """
    name = "add"
    binary_prediction = True
    param_schema = {}
    def __init__(self, **kwargs):
        super().__init__(name="add", **kwargs)

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _compute(self, y_true, y_pred):
        """
        Calculate the absolute detection distance.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The absolute detection distance.
        """
        starts, ends = self._segment_bounds(y_true)
        a_points = np.flatnonzero(y_pred)
        if a_points.size == 0:
            return 0

        if starts.size == 0:
            return 0.0

        prefix_points = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(a_points, dtype=np.int64))
        )

        distance = 0.0
        matched_point_count = 0
        for start, end in zip(starts, ends):
            left = int(np.searchsorted(a_points, start, side="left"))
            right = int(np.searchsorted(a_points, end, side="right"))
            if left >= right:
                continue
            matched_point_count += right - left

            center = int((int(start) + int(end)) / 2)
            norm = max(1, int(end) - int(start) + 1)

            mid = int(np.searchsorted(a_points, center, side="right"))
            mid = max(left, min(mid, right))

            left_count = mid - left
            left_sum = int(prefix_points[mid] - prefix_points[left])
            right_count = right - mid
            right_sum = int(prefix_points[right] - prefix_points[mid])

            left_contrib = center * left_count - left_sum
            right_contrib = right_sum - center * right_count
            distance += (left_contrib + right_contrib) / float(norm)

        if matched_point_count == 0:
            warnings.warn(
                "No predicted anomaly points overlap any ground-truth anomaly segment. "
                "Returning 0.0 for AbsoluteDetectionDistance.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0.0
        return distance / matched_point_count

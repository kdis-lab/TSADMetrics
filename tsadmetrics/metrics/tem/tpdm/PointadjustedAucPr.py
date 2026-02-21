from ....base.Metric import Metric
import numpy as np
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

    @staticmethod
    def _segment_bounds(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return starts, ends

    def _prepare_true_state(self, y_true):
        y_true_bool = np.asarray(y_true, dtype=np.bool_)
        starts, ends = self._segment_bounds(y_true_bool)
        segment_lengths = (ends - starts + 1).astype(np.int64, copy=False)

        point_to_segment = np.full(y_true_bool.shape[0], -1, dtype=np.int64)
        for seg_id, (start, end) in enumerate(zip(starts, ends)):
            point_to_segment[start:end + 1] = seg_id

        total_true = int(np.sum(segment_lengths, dtype=np.int64))
        return y_true_bool, point_to_segment, segment_lengths, total_true

    def _threshold_tp_fp(self, y_true, y_anomaly_scores):
        _, point_to_segment, segment_lengths, total_true = self._prepare_true_state(y_true)
        scores = np.asarray(y_anomaly_scores)
        if scores.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), total_true

        desc_idx = np.argsort(scores, kind="quicksort")[::-1]
        sorted_scores = scores[desc_idx]
        distinct_value_indices = np.flatnonzero(np.diff(sorted_scores))
        if distinct_value_indices.size > 0:
            threshold_ends = np.concatenate(
                (distinct_value_indices, np.array([scores.size - 1], dtype=distinct_value_indices.dtype))
            )
        else:
            threshold_ends = np.array([scores.size - 1], dtype=np.int64)

        segment_detected = np.zeros(segment_lengths.size, dtype=np.bool_)
        tps = np.empty(threshold_ends.size, dtype=np.float64)
        fps = np.empty(threshold_ends.size, dtype=np.float64)
        tp = 0
        fp = 0
        start = 0

        for idx, end in enumerate(threshold_ends):
            group_indices = desc_idx[start:end + 1]
            group_segments = point_to_segment[group_indices]

            fp += int(np.sum(group_segments < 0, dtype=np.int64))

            valid_segments = group_segments[group_segments >= 0]
            if valid_segments.size > 0:
                unique_segments = np.unique(valid_segments)
                new_segments = unique_segments[~segment_detected[unique_segments]]
                if new_segments.size > 0:
                    segment_detected[new_segments] = True
                    tp += int(np.sum(segment_lengths[new_segments], dtype=np.int64))

            tps[idx] = float(tp)
            fps[idx] = float(fp)
            start = end + 1

        return tps, fps, total_true

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
        y_true_bool = np.asarray(y_true, dtype=np.bool_)
        y_pred_bool = np.asarray(y_pred, dtype=np.bool_)
        starts, ends = self._segment_bounds(y_true_bool)

        total_true = int(np.sum(y_true_bool, dtype=np.int64))
        tp = 0
        if starts.size > 0:
            pred_prefix = np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(y_pred_bool, dtype=np.int64))
            )
            detected = (pred_prefix[ends + 1] - pred_prefix[starts]) > 0
            lengths = (ends - starts + 1).astype(np.int64, copy=False)
            tp = int(np.sum(lengths[detected], dtype=np.int64))

        fp = int(np.sum(y_pred_bool & ~y_true_bool, dtype=np.int64))
        fn = total_true - tp

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
        tps, fps, total_true = self._threshold_tp_fp(y_true, y_anomaly_scores)
        if tps.size > 0:
            fns = float(total_true) - tps
            precisions = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
            recalls = np.divide(tps, tps + fns, out=np.zeros_like(tps), where=(tps + fns) > 0)
        else:
            precisions = np.array([], dtype=np.float64)
            recalls = np.array([], dtype=np.float64)

        # Add endpoints for PR curve
        recalls = np.concatenate((np.array([0.0]), recalls, np.array([1.0])))
        precisions = np.concatenate((np.array([1.0]), precisions, np.array([0.0])))

        # Sort by recall (increasing order)
        sorted_indices = np.argsort(recalls)
        recalls_sorted = recalls[sorted_indices]
        precisions_sorted = precisions[sorted_indices]

        return auc(recalls_sorted, precisions_sorted)

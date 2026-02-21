from ....base.Metric import Metric
import numpy as np

class MeanTimeToDetect(Metric):
    """
    Calculate mean time to detect for anomaly detection in time series.
    
    This metric quantifies the average detection delay across all true anomaly events.  
    For each ground-truth anomaly segment, let i be the index where the segment starts, 
    and let :math:`{j \\geq i}` be the first index within that segment where the model predicts an anomaly.  
    The detection delay for that event is defined as:

    .. math::
        \\Delta t = j - i

    The MTTD is the mean of all such :math:`{\\Delta t}` values, one per true anomaly segment, and expresses 
    the average number of time steps between the true onset of an anomaly and its first detection.

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
        pred_idxs = np.flatnonzero(y_pred)
        pred_count = int(pred_idxs.size)
        t_sum = 0
        for a in starts:
            idx = int(np.searchsorted(pred_idxs, a, side="left"))
            if idx < pred_count:
                t_sum += int(pred_idxs[idx]) - int(a)

        return t_sum / len(starts)

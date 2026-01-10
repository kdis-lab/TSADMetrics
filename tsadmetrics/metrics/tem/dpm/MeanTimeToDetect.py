from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise

class MeanTimeToDetect(Metric):
    """
    Calculate mean time to detect for anomaly detection in time series.
    
    This metric quantifies the average detection delay across all true anomaly events.  
    For each ground-truth anomaly segment, let i be the index where the segment starts, 
    and let :math:`{j \geq i}` be the first index within that segment where the model predicts an anomaly.  
    The detection delay for that event is defined as:

    .. math::
        \Delta t = j - i

    The MTTD is the mean of all such :math:`{\Delta t}` values, one per true anomaly segment, and expresses 
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

        a_events = full_series_to_segmentwise(y_true)
        t_sum = 0
        for a, _ in a_events:
            for i in range(a, len(y_pred)):
                if y_pred[i] == 1:
                    t_sum += i - a
                    break

        return t_sum / len(a_events)

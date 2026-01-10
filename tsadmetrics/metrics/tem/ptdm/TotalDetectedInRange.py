from ....base.Metric import Metric
from ....utils.functions_counting_metrics import counting_method
import numpy as np

class TotalDetectedInRange(Metric):
    """
    Calculate total detected in range for anomaly detection in time series.

    This metric measures the proportion of true anomaly events that are correctly detected.
    It is defined as:

    .. math::
        \\text{TDIR} = \\frac{EM + DA}{EM + DA + MA}

    Where:

    - EM (Exact Match):
        Number of predicted anomaly segments that exactly match a true anomaly segment.
    - DA (Detected Anomaly):
        Number of true anomaly points not exactly matched where at least one prediction falls
        within a window [i-k, i+k] around the true point index i or within the true segment range.
    - MA (Missed Anomaly):
        Number of true anomaly segments that do not overlap any predicted anomaly segment
        even within a k-step tolerance window around true points.

    Reference:
        For more information, see the original paper:
            https://acta.sapientia.ro/content/docs/evaluation-metrics-for-anomaly-detection.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"tdir"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        k (int):
            Half-window size for tolerance around each true anomaly point. A prediction within k
            time steps of a true point counts toward detection.
    """
    name = "tdir"
    binary_prediction = True
    param_schema = {
        "k": {
            "default": 5,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="tdir", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the total detected in range score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The total detected in range score.
        """

        if np.sum(y_pred) == 0:
            return 0

        k = self.params["k"]
        em, da, ma, _ = counting_method(y_true, y_pred, k)
        return (em + da) / (em + da + ma)

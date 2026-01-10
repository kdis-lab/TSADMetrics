from ....base.Metric import Metric
import numpy as np
from ....utils.functions_latency_sparsity_aware import calc_twseq

class LatencySparsityawareFScore(Metric):
    """
    Calculate latency and sparsity aware F-score for anomaly detection in time series.
    
    This metric is based on the standard F-score, but applies a temporal adjustment 
    to the predictions before computing it. Specifically, for each ground-truth anomalous segment, 
    all points in the segment are marked as correctly detected only after the first true positive 
    is predicted within that segment. This encourages early detection by delaying credit for correct 
    predictions until the anomaly is initially detected. Additionally, to reduce the impact of 
    scattered false positives, predictions are subsampled using a sparsity factor n, so that 
    only one prediction is considered every n time steps. The adjusted predictions are then used 
    to _compute the standard point-wise F-score.

    Reference:
        Implementation based on:
            https://dl.acm.org/doi/10.1145/3447548.3467174
        For more information, see the original paper:
            https://doi.org/10.1145/3447548.3467174

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"lsaf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        ni (int):
            The batch size used in the implementation to handle latency and sparsity.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """
    name = "lsaf"
    binary_prediction = True
    param_schema = {
        "ni": {
            "default": 1,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="lsaf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the latency and sparsity aware F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The latency and sparsity aware F-score, which is the harmonic mean 
            of precision and recall, adjusted by the beta value.
        """

        if np.sum(y_pred) == 0:
            return 0

        _, precision, recall, _, _, _, _, _ = calc_twseq(
            y_pred,
            y_true,
            normal=0,
            threshold=0.5,
            tw=self.params["ni"],
        )

        if precision == 0 or recall == 0:
            return 0

        beta = self.params["beta"]
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

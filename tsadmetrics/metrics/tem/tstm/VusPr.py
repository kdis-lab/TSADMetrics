from ....base.Metric import Metric
from ....utils.functions_vus import generate_curve
import numpy as np
class VusPr(Metric):
    """
    Calculate the VUS-PR (Volume Under the PR Surface) score for anomaly detection in time series.

    This metric is an extension of the classical AUC-PR, incorporating a temporal tolerance parameter `window`
    that smooths the binary ground-truth labels. It allows for some flexibility in the detection of 
    anomalies that are temporally close to the true events. The final metric integrates the PR-AUC
    over several levels of temporal tolerance (from 0 to `window`), yielding a volume under the PR surface.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://dl.acm.org/doi/10.14778/3551793.3551830

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"vus_pr"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.

    Parameters:
        window (int):
            Maximum temporal tolerance used to smooth the evaluation.
            Default is 4.
    """
    name = "vus_pr"
    binary_prediction = False
    param_schema = {
        "window": {
            "default": 4,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="vus_pr", **kwargs)

    def _compute(self, y_true, y_anomaly_scores):
        """
        Calculate the VUS-PR score.

        Parameters:
            y_true (np.array):
                Ground-truth binary labels (0 = normal, 1 = anomaly).
            y_anomaly_scores (np.array):
                Anomaly scores for each time point.

        Returns:
            float: VUS-PR score.
        """
        window = self.params["window"]
        _, _, _, _, _, _, _, pr = generate_curve(y_true, y_anomaly_scores, slidingWindow=window)

        return pr

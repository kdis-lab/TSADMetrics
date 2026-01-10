from ....base.Metric import Metric
from pate.PATE_metric import PATE
import numpy as np


class Pate(Metric):
    """
    Calculate PATE score for anomaly detection in time series using real-valued anomaly scores.

    This version of PATE operates on continuous anomaly scores rather than binary predictions.
    It assigns weights to each score according to its temporal proximity to the true anomaly
    intervals. An early buffer of length `early` and a delay buffer of length `delay` define
    the tolerance regions before and after each anomaly. High scores within the true interval
    receive full weight, while scores in the buffer zones are linearly decayed based on their
    distance from the interval edges. Scores outside all tolerance zones contribute as false
    positives, and intervals with insufficiently high scores are penalized as false negatives.

    The final PATE score aggregates these weighted contributions to produce a smooth,
    continuous performance measure sensitive to both timing and confidence.

    Reference:
        Implementation based on:
            https://arxiv.org/abs/2405.12096
        For more information, see the original paper:
            https://arxiv.org/abs/2405.12096

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pate"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.

    Parameters:
        early (int):
            Length of the early buffer zone before each anomaly interval.
        delay (int):
            Length of the delay buffer zone after each anomaly interval.
    """
    name = "pate"
    binary_prediction = False
    param_schema = {
        "early": {
            "default": 5,
            "type": int
        },
        "delay": {
            "default": 5,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="pate", **kwargs)

    def _compute(self, y_true, y_anomaly_scores):
        """
        Calculate the real-valued PATE score.

        Parameters:
            y_true (np.array):
                Ground truth binary labels (0 = normal, 1 = anomaly).
            y_anomaly_scores (np.array):
                Real-valued anomaly scores for each time point.

        Returns:
            float: The real-valued PATE score.
        """

        return PATE(y_true, y_anomaly_scores, self.params["early"], self.params["delay"], binary_scores=False)
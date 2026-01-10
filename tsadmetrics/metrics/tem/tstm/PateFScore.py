from ....base.Metric import Metric
from pate.PATE_metric import PATE


class PateFScore(Metric):
    """
    Calculate PATE F-Score for anomaly detection in time series using binary predictions.

    This metric evaluates how well the predicted binary anomalies align with the ground truth,
    considering temporal proximity around each real anomaly interval. It defines two tolerance
    zones: an early buffer of length `early` preceding the true interval and a delay buffer of
    length `delay` following it. Detections within the true interval receive full credit, while
    those in the buffer zones receive linearly decaying weights depending on their temporal
    distance from the true anomaly. Predictions outside these regions are treated as false
    positives, and missed intervals as false negatives.

    The weighted contributions of precision and recall are combined into a final 
    F-Score, measuring both timing accuracy and detection completeness.

    Reference:
        Implementation based on:
            https://arxiv.org/abs/2405.12096
        For more information, see the original paper:
            https://arxiv.org/abs/2405.12096

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pate_f1"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    
    Parameters:
        early (int):
            The maximum number of time steps before an anomaly must be predicted to be considered early.
        delay (int):
            The maximum number of time steps after an anomaly must be predicted to be considered delayed.
    """
    name = "pate_f1"
    binary_prediction = True
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
        super().__init__(name="pate_f1", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the PATE score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The PATE score.
        """

        early = self.params["early"]
        delay = self.params["delay"]

        return PATE(y_true, y_pred, early, delay, binary_scores=True)
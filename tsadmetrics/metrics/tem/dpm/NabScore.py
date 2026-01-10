from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_pointwise
from ....utils.functions_nabscore import Sweeper, calculate_scores

class NabScore(Metric):
    """
    Calculate NAB score for anomaly detection in time series.

    This metric rewards early and accurate detections of anomalies while penalizing false positives. 
    For each ground truth anomaly segment, only the first correctly predicted anomaly point contributes 
    positively to the score, with earlier detections receiving higher rewards. In contrast, every false 
    positive prediction contributes negatively.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://doi.org/10.1109/ICMLA.2015.141

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"nab_score"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomay scores.
    """
    name = "nab_score"
    binary_prediction = True
    def __init__(self, **kwargs):
        super().__init__(name="nab_score", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the NAB score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The computed NAB score.
        """
        sweeper = Sweeper(probationPercent=0, costMatrix={"tpWeight": 1, "fpWeight": 0.11, "fnWeight": 1})

        if len(full_series_to_pointwise(y_pred)) == 0:
            return 0
        if len(full_series_to_pointwise(y_true)) == 0:
            return np.nan

        try:
            sweeper, null_score, raw_score = calculate_scores(
                sweeper,
                full_series_to_pointwise(y_true),
                full_series_to_pointwise(y_pred),
                len(y_true)
            )
            sweeper, null_score, perfect_score = calculate_scores(
                sweeper,
                full_series_to_pointwise(y_true),
                full_series_to_pointwise(y_true),
                len(y_true)
            )
            return (raw_score - null_score) / (perfect_score - null_score) * 100
        except Exception:
            return 0

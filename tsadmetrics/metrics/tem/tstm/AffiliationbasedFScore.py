from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise
from ....utils.functions_affiliation import pr_from_events, reformat_segments

class AffiliationbasedFScore(Metric):
    """
    Calculate affiliation based F-score for anomaly detection in time series.

    This metric combines the affiliation-based precision and recall into a single score 
    using the harmonic mean, adjusted by a weight :math:`{\\beta}` to control the relative importance 
    of recall versus precision. Since both precision and recall are distance-based, 
    the F-score reflects a balance between how well predicted anomalies align with true 
    anomalies and vice versa.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://dl.acm.org/doi/10.1145/3534678.3539339

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"aff_f"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    
    Parameters:
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """
    name = "aff_f"
    binary_prediction = True
    param_schema = {
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="aff_f", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the affiliation based F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The affiliation based F-score.
        """

        if np.sum(y_pred) == 0:
            return 0

        pr_output = pr_from_events(
            reformat_segments(full_series_to_segmentwise(y_pred)),
            reformat_segments(full_series_to_segmentwise(y_true)),
            (0, len(y_true)),
        )

        precision = pr_output['precision']
        recall = pr_output['recall']

        if precision == 0 or recall == 0:
            return 0

        beta = self.params["beta"]
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

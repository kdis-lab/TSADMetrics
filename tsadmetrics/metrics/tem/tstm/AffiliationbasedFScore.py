from ....base.Metric import Metric
import numpy as np
from ....utils.functions_affiliation import pr_from_events

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

    @staticmethod
    def _events_from_full_series(series):
        series_bool = np.asarray(series, dtype=np.bool_)
        transitions = np.diff(series_bool.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        return [(int(start), int(end) + 1) for start, end in zip(starts, ends)]

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

        if not np.any(y_pred):
            return 0

        pr_output = pr_from_events(
            self._events_from_full_series(y_pred),
            self._events_from_full_series(y_true),
            (0, len(y_true)),
        )

        precision = pr_output['precision']
        recall = pr_output['recall']

        if precision == 0 or recall == 0:
            return 0

        beta = self.params["beta"]
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

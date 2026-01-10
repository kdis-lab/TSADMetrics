from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise
from ....utils.functions_affiliation import pr_from_events, reformat_segments

class NormalizedAffiliationbasedFScore(Metric):
    """
    Calculate normalized affiliation-based F-score for anomaly detection in time series.

    This metric combines the affiliation-based precision and recall into a single score,
    weighted by :math:`\\beta`, but first applies an affine normalization to the
    affiliation-based precision using a threshold parameter :math:`\\alpha`:

    .. math::
        P_{u}^{aff} = \\frac{\\mathrm{precision}_{aff} - \\alpha}{1 - \\alpha}

    Then the (signed) F-score is computed as:

    .. math::
        F_{\\beta} = \\frac{(1 + \\beta^2)\\,|P_{u}^{aff}|\\,\\mathrm{recall}_{aff}}
                           {\\beta^2\\,P_{u}^{aff} + \\mathrm{recall}_{aff}}
                     \\times \\operatorname{sign}(P_{u}^{aff})

    Notes:
        - If there are no predicted anomalies, the score is 0.
        - If :math:`1-\\alpha = 0`, the score is 0 to avoid division by zero.
        - If the denominator :math:`\\beta^2 P_{u}^{aff} + \\mathrm{recall}_{aff}` is 0,
          the score is 0 to avoid division by zero.

    Reference:
        For more information, see the original paper:
            https://ieeexplore.ieee.org/abstract/document/11099055

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"naff_f"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    
    Parameters:
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
        alpha (float):
            Normalization threshold applied to affiliation-based precision before computing
            the F-score. Must satisfy :math:`\\alpha < 1`. Default is 0.
    """
    name = "naff_f"
    binary_prediction = True
    param_schema = {
        "alpha": {
            "default": 0.0,
            "type": float
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="naff_f", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the normalized affiliation-based F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The normalized affiliation-based F-score.
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

        if recall == 0:
            return 0

        beta = self.params["beta"]
        alpha = self.params["alpha"]

        # Avoid division by zero if alpha == 1
        if (1 - alpha) == 0:
            return 0

        Puaff = (precision - alpha) / (1 - alpha)
        denom = (beta**2) * Puaff + recall

        # Guard against zero denominator
        if denom == 0:
            return 0

        fscore_unsigned = ((1 + beta**2) * np.abs(Puaff) * recall) / denom
        sign = -1.0 if Puaff < 0 else 1.0

        return float(sign * fscore_unsigned)

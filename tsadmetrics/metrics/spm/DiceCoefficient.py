from ...base.Metric import Metric
import numpy as np

class DiceCoefficient(Metric):
    """
    Calculate the Dice Coefficient for anomaly detection in time series.

    The Dice Coefficient is a similarity measure between the predicted and ground-truth 
    binary anomaly segments. It is mathematically equivalent to the F1-score but derived 
    from a set-theoretic perspective. The metric quantifies the overlap between the 
    predicted anomalies and the actual anomalies, taking values between 0 and 1, where 
    1 indicates perfect agreement.

    The Dice Coefficient is defined as:

    .. math::
        \\mathrm{DiceCoefficient} = \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}

    where:
        - :math:`TP` is the number of true positives (correctly detected anomaly points),
        - :math:`FP` is the number of false positives (incorrectly predicted anomalies),
        - :math:`FN` is the number of false negatives (missed true anomalies).

    Notes:
        - The Dice Coefficient is symmetric: swapping prediction and ground truth yields 
          the same result.
        - If both `y_true` and `y_pred` are all zeros (no anomalies), the metric returns 1.0 
          to represent perfect agreement in the absence of anomalies.

    Reference:
        For more information, see the original paper:
            https://www.sciencedirect.com/science/article/pii/S0094576522003162

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"dicec"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        (none)
    """
    name = "dicec"
    binary_prediction = True
    param_schema = {
        # No parameters required for Dice Coefficient
    }

    def __init__(self, **kwargs):
        super().__init__(name="dicec", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the Dice Coefficient.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The Dice Coefficient score.
        """
        if y_true.size == 0:
            return 0.0

        tp = float(np.dot(y_true, y_pred))
        true_positive_total = float(np.sum(y_true))
        pred_positive_total = float(np.sum(y_pred))
        denom = true_positive_total + pred_positive_total

        if denom == 0:
            return 1.0

        return float((2.0 * tp) / denom)

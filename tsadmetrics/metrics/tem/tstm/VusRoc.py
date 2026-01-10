from ....base.Metric import Metric
from ....utils.functions_vus import generate_curve
import numpy as np

class VusRoc(Metric):
    """
    Calculate the VUS-ROC (Volume Under the ROC Surface) score for anomaly detection in time series.

    This metric extends the classical AUC-ROC by introducing a temporal tolerance parameter `l`, which
    smooths the binary ground-truth labels. The idea is to allow a flexible evaluation that tolerates
    small misalignments in the detection of anomalies. The final score is computed by integrating 
    the ROC-AUC over different values of the tolerance parameter, from 0 to `window`, thus producing
    a volume under the ROC surface.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://dl.acm.org/doi/10.14778/3551793.3551830

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"vus_roc"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.

    Parameters:
        window (int):
            Maximum temporal tolerance `l` used to smooth the evaluation.
            Default is 4.
    """
    name = "vus_roc"
    binary_prediction = False
    param_schema = {
        "window": {
            "default": 4,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="vus_roc", **kwargs)

    def _compute(self, y_true, y_anomaly_scores):
        """
        Calculate the VUS-ROC score.

        Parameters:
            y_true (np.array):
                Ground-truth binary labels (0 = normal, 1 = anomaly).
            y_anomaly_scores (np.array):
                Anomaly scores for each time point.

        Returns:
            float: VUS-ROC score.
        """

        _, _, _, _, _, _, roc, _ = generate_curve(
            y_true, y_anomaly_scores, self.params["window"]
        )

        return roc

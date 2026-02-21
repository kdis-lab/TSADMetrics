from ...base.Metric import Metric
import numpy as np

class PrecisionAtK(Metric):
    """
    Precision at K (P@K) for anomaly detection in time series.

    This metric evaluates how many of the top-`k` points with the highest anomaly
    scores correspond to true anomalies. It is particularly useful when focusing
    on identifying the most anomalous points rather than setting a global threshold.

    By definition, `k` is automatically set to the number of true anomalies present
    in `y_true`.

    .. math::
        k = \\sum(y\\_true)

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"pak"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `False`
            since it requires continuous anomaly scores.
    """

    name = "pak"
    binary_prediction = False

    def __init__(self, **kwargs):
        """
        Initialize the PrecisionAtK metric.

        Parameters:
            **kwargs:
                Additional keyword arguments passed to the base `Metric` class.
                These may include configuration parameters or overrides.
        """
        super().__init__(name="pak", **kwargs)

    def _compute(self, y_true, y_anomaly_scores):
        """
        Compute the Precision at K (P@K) score.

        Parameters:
            y_true (np.ndarray):
                Ground-truth binary labels for the time series.
                Values must be 0 (normal) or 1 (anomaly).
            y_anomaly_scores (np.ndarray):
                Continuous anomaly scores assigned to each point in the series.

        Returns:
            float:
                The precision at K score, where K = number of anomalies in `y_true`.

        Raises:
            AssertionError:
                If `k = sum(y_true)` is 0.
                If fewer than `k` points are predicted as anomalies.
        """
        k = int(np.count_nonzero(y_true))
        assert k > 0, "The number of true anomalies (k) must be greater than zero."
        n_samples = int(y_anomaly_scores.size)
        threshold_idx = n_samples - k
        threshold = np.partition(y_anomaly_scores, threshold_idx)[threshold_idx]

        pred = y_anomaly_scores >= threshold
        pred_count = int(np.count_nonzero(pred))
        assert pred_count >= k, (
            f"Number of predicted positives ({pred_count}) should be >= k ({k})."
        )
        return float(np.dot(pred, y_true) / pred_count)

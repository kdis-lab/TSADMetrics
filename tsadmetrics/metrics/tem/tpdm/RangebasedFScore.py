from ....base.Metric import Metric
import numpy as np

class RangebasedFScore(Metric):
    """
    Range-based F-score for anomaly detection in time series.

    This metric evaluates anomaly detection performance over temporal ranges,
    combining range-based precision and recall into a harmonic mean. It accounts
    for positional bias, existence and overlap rewards, and cardinality penalties,
    allowing fine-grained control over missed detections and false alarms.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://proceedings.neurips.cc/paper_files/paper/2018/file/8f468c873a32bb0619eaeb2050ba45d1-Paper.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"rbf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        p_alpha (float): 
            Relative importance of existence reward for precision (0 <= alpha_p <= 1).
        r_alpha (float): 
            Relative importance of existence reward for recall (0 <= alpha_r <= 1).
        p_bias (str): 
            Positional bias for precision ("flat", "front", "middle", "back").
        r_bias (str): 
            Positional bias for recall ("flat", "front", "middle", "back").
        cardinality_mode (str, optional): 
            Cardinality factor type ("one", "reciprocal", "gamma").
        beta (float): 
            Weight of precision in the F-score. Default = 1.
    """

    name = "rbf"
    binary_prediction = True
    param_schema = {
        "beta": {"default": 1.0, "type": float},
        "p_alpha": {"default": 0.5, "type": float},
        "r_alpha": {"default": 0.5, "type": float},
        "p_bias": {"default": "flat", "type": str},
        "r_bias": {"default": "flat", "type": str},
        "cardinality_mode": {"default": "one", "type": str},
    }

    def __init__(self, **kwargs):
        """
        Initialize the RangebasedFScore metric.

        Parameters:
            **kwargs: Additional parameters matching the param_schema.
        """
        super().__init__(name="rbf", **kwargs)

    # -----------------------
    # Utility functions
    # -----------------------

    def _gamma(self):
        """User-defined gamma function (default implementation)."""
        return 1.0

    def _gamma_select(self, gamma: str, overlap: int) -> float:
        """
        Select gamma value based on cardinality mode and overlap.

        Args:
            gamma (str): 'one', 'reciprocal', or 'gamma'.
            overlap (int): Number of overlapping ranges.

        Returns:
            float: Selected gamma factor.
        """
        assert isinstance(overlap, int), TypeError("overlap must be int")

        if gamma == "one":
            return 1.0
        elif gamma == "reciprocal":
            return 1.0 / overlap if overlap > 1 else 1.0
        elif gamma == "gamma":
            return 1.0 / self._gamma() if overlap > 1 else 1.0
        else:
            raise ValueError(f"Invalid gamma type: {gamma}")

    def _gamma_function(self, overlap_count):
        """Compute gamma factor from overlap count based on cardinality mode."""
        return self._gamma_select(self.params['cardinality_mode'], overlap_count[0])

    def _compute_omega_reward(self, bias, r1, r2, overlap_count):
        """
        Compute omega reward based on overlap of two ranges with positional bias.

        Args:
            bias (str): Positional bias type.
            r1 (np.array): First range [start, end].
            r2 (np.array): Second range [start, end].
            overlap_count (list): List to track number of overlaps.

        Returns:
            float: Omega reward.
        """
        if r1[1] < r2[0] or r1[0] > r2[1]:
            return 0
        overlap_count[0] += 1
        overlap = np.zeros(r1.shape)
        overlap[0] = max(r1[0], r2[0])
        overlap[1] = min(r1[1], r2[1])
        return self._omega_function(bias, r1, overlap)

    def _omega_function(self, bias, rrange, overlap):
        """Compute normalized positional omega function for range overlap."""
        anomaly_length = rrange[1] - rrange[0] + 1
        my_positional_bias, max_positional_bias = 0, 0

        for i in range(1, anomaly_length + 1):
            temp_bias = self._delta_function(bias, i, anomaly_length)
            max_positional_bias += temp_bias
            j = rrange[0] + i - 1
            if overlap[0] <= j <= overlap[1]:
                my_positional_bias += temp_bias

        return my_positional_bias / max_positional_bias if max_positional_bias > 0 else 0

    def _delta_function(self, bias, t, anomaly_length):
        """Compute positional delta function for omega."""
        return self._delta_select(bias, t, anomaly_length)

    def _delta_select(self, delta, t, anomaly_length):
        """Select positional bias value based on delta type."""
        if delta == "flat":
            return 1.0
        elif delta == "front":
            return float(anomaly_length - t + 1)
        elif delta == "middle":
            return float(t if t <= anomaly_length / 2 else anomaly_length - t + 1)
        elif delta == "back":
            return float(t)
        elif delta == "udf_delta":
            return self._udf_delta(t, anomaly_length)
        else:
            raise ValueError("Invalid positional bias value")

    def _udf_delta(self, t=None, anomaly_length=None):
        """User-defined delta function (default returns 1.0)."""
        return 1.0

    def _shift(self, arr, num, fill_value=np.nan):
        """Shift array by `num` positions, filling empty slots."""
        arr = np.roll(arr, num)
        if num < 0:
            arr[num:] = fill_value
        elif num > 0:
            arr[:num] = fill_value
        return arr

    def _prepare_data(self, values_real, values_pred):
        """
        Prepare ranges for real and predicted anomalies.

        Returns:
            Tuple of numpy arrays: (real_anomalies, predicted_anomalies)
        """
        assert len(values_real) == len(values_pred)
        predicted_anomalies_ = np.argwhere(values_pred == 1).ravel()
        predicted_anomalies = self._extract_ranges(predicted_anomalies_)
        real_anomalies_ = np.argwhere(values_real == 1).ravel()
        real_anomalies = self._extract_ranges(real_anomalies_)
        return real_anomalies, predicted_anomalies

    def _extract_ranges(self, anomaly_indices):
        shifted_fwd = self._shift(anomaly_indices, 1, fill_value=anomaly_indices[0])
        shifted_bwd = self._shift(anomaly_indices, -1, fill_value=anomaly_indices[-1])
        start = np.argwhere((shifted_fwd - anomaly_indices) != -1).ravel()
        end = np.argwhere((anomaly_indices - shifted_bwd) != -1).ravel()
        return np.hstack([anomaly_indices[start].reshape(-1, 1), anomaly_indices[end].reshape(-1, 1)])

    # -----------------------
    # Range-based precision & recall
    # -----------------------

    def _compute_precision(self, y_true, y_pred):
        """Compute range-based precision."""
        alpha = self.params["p_alpha"]
        if len(y_pred) == 0:
            return 0
        precision = 0
        for range_p in y_pred:
            omega_reward, overlap_count = 0, [0]
            for range_r in y_true:
                omega_reward += self._compute_omega_reward(self.params["p_bias"], range_p, range_r, overlap_count)
            overlap_reward = self._gamma_function(overlap_count) * omega_reward
            existence_reward = 1 if overlap_count[0] > 0 else 0
            precision += alpha * existence_reward + (1 - alpha) * overlap_reward
        return precision / len(y_pred)

    def _compute_recall(self, y_true, y_pred):
        """Compute range-based recall."""
        alpha = self.params["r_alpha"]
        if len(y_true) == 0:
            return 0
        recall = 0
        for range_r in y_true:
            omega_reward, overlap_count = 0, [0]
            for range_p in y_pred:
                omega_reward += self._compute_omega_reward(self.params["r_bias"], range_r, range_p, overlap_count)
            overlap_reward = self._gamma_function(overlap_count) * omega_reward
            existence_reward = 1 if overlap_count[0] > 0 else 0
            recall += alpha * existence_reward + (1 - alpha) * overlap_reward
        return recall / len(y_true)

    # -----------------------
    # Metric computation
    # -----------------------

    def _compute(self, y_true, y_pred):
        """
        Compute the range-based F-score.

        Returns:
            float: F-beta score using range-based precision and recall.
        """
        if np.sum(y_pred) == 0:
            return 0

        beta = self.params['beta']
        y_true_mod, y_pred_mod = self._prepare_data(y_true, y_pred)
        precision = self._compute_precision(y_true_mod, y_pred_mod)
        recall = self._compute_recall(y_true_mod, y_pred_mod)

        return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if precision + recall > 0 else 0

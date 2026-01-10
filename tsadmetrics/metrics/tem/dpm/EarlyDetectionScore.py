from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise

class EarlyDetectionScore(Metric):
    """
    Calculate the Early Detection (ED) score for anomaly detection in time series. 

    This metric quantifies **how early** an anomaly is detected relative to its true occurrence 
    by evaluating the timing of the first prediction within a defined anomaly window around each 
    ground-truth anomaly. 

    For every anomaly window :math:`\\mathrm{aw}_d(a) = [t_b, t_e]`, the ED score is computed as 
    :math:`\\mathrm{ed}_a = 1 - \\frac{\\mathrm{pos}_d(T(a)) - b}{e - b}` if a detection occurs inside 
    the window, and 0 otherwise, where :math:`\\mathrm{pos}_d(T(a))` denotes the position of the first 
    predicted anomaly within the window and :math:`b` and :math:`e` represent the window’s start and end 
    indices, respectively. Consequently, detections closer to the start of the window yield scores 
    approaching 1 (indicating early detection), while detections near the end yield values closer to 0. 

    Following the specification in the ADE paper, the anomaly window length is defined as 
    :math:`\\lVert \\mathrm{aw}_d \\rVert = 0.1 \\cdot \\lVert d \\rVert / \\lVert A_d \\rVert`, 
    where :math:`\\lVert d \\rVert` is the length of the time series and :math:`\\lVert A_d \\rVert` 
    is the total number of anomalies. Each window is symmetrically centered on the anomaly and bounded 
    within the series limits. 

    The final ED score is obtained by averaging the individual :math:`\\mathrm{ed}_a` values 
    for all ground-truth anomalies, thus providing a single measure of how promptly the model detects 
    anomalies across the entire series.


    Reference:
        For more information, see the original paper:
            https://ieeexplore.ieee.org/document/7987370

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"eds"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly labels.
    """
    name = "eds"
    binary_prediction = True
    param_schema = {
        # No parameters for this metric
    }

    def __init__(self, **kwargs):
        super().__init__(name="eds", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the Early Detection (ED) score.

        Parameters:
            y_true (np.array):
                Binary ground-truth labels for the time series (1 = anomaly).
            y_pred (np.array):
                Binary predicted labels for the time series (1 = predicted anomaly).

        Returns:
            float: The average ED score across all ground-truth anomalies.
        """
        n = len(y_true)
        if n == 0:
            return 0.0

        segs = full_series_to_segmentwise(y_true)
        if len(segs) == 0:
            return 0.0
        win_len = int(max(1, round(0.1 * n / len(segs))))

        ed_values = []

        pred_idxs = np.flatnonzero(y_pred == 1)

        for (s, e) in segs:
            a = s

            half = win_len // 2
            tb = max(0, a - half)
            te = tb + win_len - 1
            if te >= n:
                te = n - 1
                tb = max(0, te - (win_len - 1))

            if te < tb:
                ed_values.append(0.0)
                continue

            in_window = pred_idxs[(pred_idxs >= tb) & (pred_idxs <= te)]
            if in_window.size == 0:
                ed_values.append(0.0)
                continue

            pos = int(in_window[0])  

            denom = te - tb
            if denom <= 0:
                ed_a = 1.0 if pos == tb else 0.0
            else:
                ed_a = 1.0 - (pos - tb) / denom

            ed_a = float(np.clip(ed_a, 0.0, 1.0))
            ed_values.append(ed_a)

        if len(ed_values) == 0:
            return 0.0

        return float(np.mean(ed_values))

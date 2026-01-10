from ....base.Metric import Metric
import numpy as np

class TimeTolerantFScore(Metric):
    """
    Calculate time tolerant F-score for anomaly detection in time series.
    This metric is based on the standard F-score, but applies a temporal adjustment 
    to the predictions before computing it. Specifically, a predicted anomalous point is considered 
    a true positive if it lies within a temporal window of size :math:`{\\tau}` around any ground-truth anomalous point. 
    This allows for small temporal deviations in the predictions to be tolerated. The adjusted predictions are then used 
    to _compute the standard point-wise F-Score.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://arxiv.org/abs/2008.05788

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"ttf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        t (int):
            The time tolerance parameter.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
    """
    name = "ttf"
    binary_prediction = True
    param_schema = {
        "t": {
            "default": 5,
            "type": int
        },
        "beta": {
            "default": 1.0,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="ttf", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the time tolerant F-score (optimized version).
        """
        t = self.params['t']
        beta = self.params['beta']
        
        # Precompute masks for efficiency
        true_anomalies = y_true == 1
        predictions = y_pred == 1
        
        # Create P′1 for recall: for each true anomaly, check if any prediction within ±t
        p_prime1 = np.zeros_like(y_true, dtype=bool)
        
        for i in np.where(true_anomalies)[0]:
            start = max(0, i - t)
            end = min(len(y_pred), i + t + 1)
            if np.any(predictions[start:end]):
                p_prime1[i] = True
        
        # Create P′2 for precision: for each prediction, check if any true anomaly within ±t
        p_prime2 = np.zeros_like(y_pred, dtype=bool)
        
        for j in np.where(predictions)[0]:
            start = max(0, j - t)
            end = min(len(y_true), j + t + 1)
            if np.any(true_anomalies[start:end]):
                p_prime2[j] = True
        
        # Calculate recall using P′1
        tp_recall = np.sum(true_anomalies & p_prime1)
        fn_recall = np.sum(true_anomalies & ~p_prime1)
        recall = tp_recall / (tp_recall + fn_recall) if (tp_recall + fn_recall) > 0 else 0.0
        
        # Calculate precision using P′2  
        tp_precision = np.sum(predictions & p_prime2)
        fp_precision = np.sum(predictions & ~p_prime2)
        precision = tp_precision / (tp_precision + fp_precision) if (tp_precision + fp_precision) > 0 else 0.0
        
        # Calculate F-score
        if precision == 0 and recall == 0:
            return 0.0
        
        f_score = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
        return f_score

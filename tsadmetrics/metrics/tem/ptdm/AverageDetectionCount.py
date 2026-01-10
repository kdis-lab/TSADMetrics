from ....base.Metric import Metric
import numpy as np
from ....utils.functions_conversion import full_series_to_segmentwise, full_series_to_pointwise

class AverageDetectionCount(Metric):
    """
    Calculate average detection count for anomaly detection in time series.

    This metric computes, for each ground-truth anomalous segment, the percentage of points within that segment 
    that are predicted as anomalous. It then averages these percentages across all true anomaly events, 
    providing an estimate of detection coverage per event.

    Reference:
        Implementation based on:
            https://ceur-ws.org/Vol-1226/paper31.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"adc"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    """

    name = "adc"
    binary_prediction = True
    param_schema = {}  

    def __init__(self, **kwargs):
        super().__init__(name="adc", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the average detection count.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The average detection count score.
        """

        
        azs = full_series_to_segmentwise(y_true)
        a_points = full_series_to_pointwise(y_pred)

        counts = []
        for az in azs:
            count = 0
            for ap in a_points:
                if ap >= az[0] and ap <= az[1]:
                    count+=1
            counts.append(count/(az[1] - az[0] + 1))  # Normalize by segment length
        
        return np.mean(counts)

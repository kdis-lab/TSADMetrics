from ....base.Metric import Metric
from ....utils.functions_conversion import full_series_to_segmentwise, full_series_to_pointwise

class AbsoluteDetectionDistance(Metric):
    """
    Calculate absolute detection distance for anomaly detection in time series.

    This metric computes, for each predicted anomaly point that overlaps a ground-truth anomaly segment, 
    the relative distance from that point to the temporal center of the corresponding segment. It then sums all 
    those distances and divides by the total number of such matching predicted points, yielding the 
    mean distance to segment centers for correctly detected points.

    Reference:
        For more information, see the original paper:
            https://ceur-ws.org/Vol-1226/paper31.pdf

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"add"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.
    """
    name = "add"
    binary_prediction = True
    param_schema = {}
    def __init__(self, **kwargs):
        super().__init__(name="add", **kwargs)

    def _compute(self, y_true, y_pred):
        """
        Calculate the absolute detection distance.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The absolute detection distance.
        """

        azs = full_series_to_segmentwise(y_true)
        a_points = full_series_to_pointwise(y_pred)
        if len(a_points) == 0:
            return 0

        distance = 0
        for az in azs:
            for ap in a_points:
                if az[0] <= ap <= az[1]:
                    center = int((az[0] + az[1]) / 2)
                    distance += abs(ap - center) / max(1, center)

        return distance / len(a_points)

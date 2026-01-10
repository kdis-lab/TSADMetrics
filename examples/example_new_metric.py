from ....base.Metric import Metric
import numpy as np

class NewMetric(Metric):
    """
    ...
    """

    name = "new_metric_name"
    binary_prediction = True
    param_schema = {
        "param1": {
            "default": 1.0,
            "type": float
        },
        "param2": {
            "default": 5,
            "type": int
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="new_metric_name", **kwargs)

    def _compute(self, y_true, y_pred):
        # Metric computation logic goes here
        return 1

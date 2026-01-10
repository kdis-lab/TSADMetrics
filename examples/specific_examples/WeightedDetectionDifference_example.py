from tsadmetrics.metrics.tem.ptdm.WeightedDetectionDifference import WeightedDetectionDifference
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = WeightedDetectionDifference()
result = metric.compute(y_true, y_pred)
print("WeightedDetectionDifference:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("wdd", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

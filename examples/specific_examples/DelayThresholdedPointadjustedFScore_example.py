from tsadmetrics.metrics.tem.dpm.DelayThresholdedPointadjustedFScore import DelayThresholdedPointadjustedFScore
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = DelayThresholdedPointadjustedFScore()
result = metric.compute(y_true, y_pred)
print("DelayThresholdedPointadjustedFScore:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("dtpaf", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

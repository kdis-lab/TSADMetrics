from tsadmetrics.metrics.tem.ptdm.AverageDetectionCount import AverageDetectionCount
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = AverageDetectionCount()
result = metric.compute(y_true, y_pred)
print("AverageDetectionCount:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("adc", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

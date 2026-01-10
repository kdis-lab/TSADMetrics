from tsadmetrics.metrics.tem.tmem.TemporalDistance import TemporalDistance
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = TemporalDistance()
result = metric.compute(y_true, y_pred)
print("TemporalDistance:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("td", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

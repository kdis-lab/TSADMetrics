from tsadmetrics.metrics.tem.tstm.VusPr import VusPr
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# Direct usage
metric = VusPr()
result = metric.compute(y_true, y_pred)
print("VusPr:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("vus_pr", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

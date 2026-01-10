from tsadmetrics.metrics.tem.tpdm.BalancedPointadjustedFScore import BalancedPointadjustedFScore
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0]

# Direct usage
metric = BalancedPointadjustedFScore(beta=1.0, w=3)
result = metric.compute(y_true, y_pred)
print("BalancedPointadjustedFScore:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("bpaf", {"w": 3, "beta": 1.0})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

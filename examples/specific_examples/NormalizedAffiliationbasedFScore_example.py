from tsadmetrics.metrics.tem.tstm.NormalizedAffiliationbasedFScore import NormalizedAffiliationbasedFScore
from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = NormalizedAffiliationbasedFScore(alpha=0.0, beta=1.0)
result = metric.compute(y_true, y_pred)
print("NormalizedAffiliationbasedFScore:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("aff_f", {"alpha": 0.5, "beta": 1.0})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

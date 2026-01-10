from tsadmetrics.metrics.tem.ptdm.PointadjustedAtKLFScore import PointadjustedAtKLFScore
from tsadmetrics.evaluation.Runner import Runner

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,0]

# Direct usage
metric = PointadjustedAtKLFScore(k=0.5, l=1, beta=1.0)
result = metric.compute(y_true, y_pred)
print("PointadjustedAtKLFScore:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("paklf", {"k":0.5, "l":1, "beta":1.0})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

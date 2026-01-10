from tsadmetrics.metrics.tem.dpm.EarlyDetectionScore import EarlyDetectionScore
from tsadmetrics.evaluation.Runner import Runner

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = EarlyDetectionScore()
result = metric.compute(y_true, y_pred)
print("EarlyDetectionScore:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("early_detection", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

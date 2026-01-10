from tsadmetrics.metrics.spm.DiceCoefficient import DiceCoefficient
from tsadmetrics.evaluation.Runner import Runner

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

# Direct usage
metric = DiceCoefficient()
result = metric.compute(y_true, y_pred)
print("DiceCoefficient:", result)

# Usage with Runner
dataset_evaluations = [
    ("dataset1", y_true, (y_pred, y_pred))
]

metrics = [
    ("dicec", {})
]

runner = Runner(dataset_evaluations, metrics)
results = runner.run()
print(results)

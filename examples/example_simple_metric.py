from tsadmetrics.metrics.tem.mdpt import PointadjustedFScore

y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
metric = PointadjustedFScore()

result = metric.compute(y_true, y_pred)
print(f"PointadjustedFScore: {result}")

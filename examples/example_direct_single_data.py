from tsadmetrics.evaluation.Runner import Runner


y_true1 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_true2 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred1 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_pred2 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

dataset_evaluations = [
    ("dataset1", y_true1, (y_pred1, y_pred1)),
    ("dataset2", y_true2, (y_pred2, y_pred2))

]

metrics = [
    ("adc",{}),
    ("dair",{}),
    ("pakf",{"k":0.2}),
    ("pakf",{"k":0.4})
    ]

runner = Runner(dataset_evaluations, metrics)
results = runner.run(generate_report=True, report_file="./example_output/example_direct_single_data_report.csv")
print(results)


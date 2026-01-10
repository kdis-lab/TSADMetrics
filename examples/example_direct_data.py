from tsadmetrics.evaluation.Runner import Runner


y_true1 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_true2 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_pred1 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_pred1_cont = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_pred2 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_pred2_cont = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

dataset_evaluations = [
    ("dataset1", y_true1, (y_pred1, y_pred1_cont)),
    ("dataset2", y_true2, (y_pred2, y_pred2_cont))

]

metrics = [
    ("adc",{}),
    ("dair",{}),
    ("pakf",{"k":0.2}),
    ("pakf",{"k":0.4}),
    ("pakf",{"k":0.5}),
    ]

runner = Runner(dataset_evaluations, metrics)
results = runner.run(generate_report=True, report_file="./example_output/example_direct_data_report.csv")
print(results)


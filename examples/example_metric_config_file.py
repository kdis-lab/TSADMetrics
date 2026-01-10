from tsadmetrics.evaluation.Runner import Runner


y_true1 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_true2 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]


dataset_evaluations = [
    ("dataset1", "example_input/results1.csv"),
    ("dataset2", "example_input/results2.csv")

]

metrics_file = "example_input/example_metrics_config.yaml"

runner = Runner(dataset_evaluations, metrics_file)
results = runner.run(generate_report=True, report_file="./example_output/example_metric_config_file_report.csv")
print(results)


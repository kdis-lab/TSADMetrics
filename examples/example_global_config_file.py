from tsadmetrics.evaluation.Runner import Runner
import numpy as np

y_true1 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
y_true2 = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]


global_config_file = "example_input/example_evaluation_config.yaml"

runner = Runner(global_config_file)
results = runner.run(generate_report=True, report_file="./example_output/example_global_config_file_report.csv")
print(results)


Using TSADmetrics
=================

This guide shows how to use the **TSADmetrics** library for evaluating time series anomaly detection algorithms.
It includes examples using the `Runner` class for evaluating datasets with multiple metrics.

Example: Direct Data
--------------------

In this example, we pass the datasets and predictions directly as Python lists.

.. code-block:: python

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
        ("adc", {}),
        ("dair", {}),
        ("pakf", {"k": 0.2}),
        ("pakf", {"k": 0.4}),
        ("pakf", {"k": 0.5}),
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run(generate_report=True, report_file="./example_output/example_direct_data_report.csv")
    print(results)

Example: Direct Single Data
---------------------------

This example evaluates datasets using a single prediction array per dataset.

.. code-block:: python

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
        ("adc", {}),
        ("dair", {}),
        ("pakf", {"k": 0.2}),
        ("pakf", {"k": 0.4}),
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run(generate_report=True, report_file="./example_output/example_direct_single_data_report.csv")
    print(results)

Example: File Reference
-----------------------

Datasets can also be provided as CSV files.

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", "example_input/results1.csv"),
        ("dataset2", "example_input/results2.csv")
    ]

    metrics = [
        ("adc", {}),
        ("dair", {}),
        ("pakf", {"k": 0.2}),
        ("pakf", {"k": 0.4})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run(generate_report=True, report_file="./example_output/example_file_reference_report.csv")
    print(results)

Example: Metric Config File
---------------------------

Metrics can be loaded from a YAML configuration file.

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", "example_input/results1.csv"),
        ("dataset2", "example_input/results2.csv")
    ]

    metrics_file = "example_input/example_metrics_config.yaml"

    runner = Runner(dataset_evaluations, metrics_file)
    results = runner.run(generate_report=True, report_file="./example_output/example_metric_config_file_report.csv")
    print(results)

Example: Global Config File
---------------------------

You can define the entire evaluation through a single global configuration file.

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    global_config_file = "example_input/example_evaluation_config.yaml"

    runner = Runner(global_config_file)
    results = runner.run(generate_report=True, report_file="./example_output/example_global_config_file_report.csv")
    print(results)

Direct Metric Usage
-------------------

You can also use metrics directly by instantiating the metric class and calling the `compute` method.

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm import PointadjustedFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # Instantiate the metric
    metric = PointadjustedFScore()

    # Compute the metric
    result = metric.compute(y_true, y_pred)
    print(f"PointadjustedFScore: {result}")

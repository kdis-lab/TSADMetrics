import warnings
from ..metrics.Registry import Registry
import numpy as np
from .Report import Report
import pandas as pd
import yaml
class Runner:
    """
    Orchestrates the evaluation of datasets using a set of metrics.

    The `Runner` class provides functionality to:

    - Load datasets from direct data, file references, or a global YAML configuration file.
    - Load metrics either directly from a list or from a configuration file.
    - Evaluate all datasets against all metrics.
    - Optionally generate a report summarizing the evaluation results.

    Parameters:
        dataset_evaluations (list or str):
            Accepted formats:
            
            1. **Global config file (str)**  
               If a string is provided and `metrics` is None, it is assumed to be
               the path to a configuration file that defines both datasets and metrics.

            2. **Direct data (list of tuples)**  
               Example::

                   [
                       ("dataset1", y_true1, (y_pred_binary1, y_pred_continuous1)),
                       ("dataset2", y_true2, (y_pred_binary2, y_pred_continuous2)),
                       ("dataset3", y_true3, y_pred3)
                   ]

               where `y_pred` may be binary or continuous.

            3. **File references (list of tuples)**  
               Example::

                   [
                       ("dataset1", "result1.csv"),
                       ("dataset2", "result2.csv")
                   ]

               Each file must contain:

               - `y_true`
               - Either:
                 * (`y_pred_binary` and `y_pred_continuous`)
                 * or (`y_pred`)

        metrics (list or str, optional):
            - **List of metrics**: Each element is a tuple of the form  
              [(metric_name, {param_name: value, ...}), ...]  

              Example::

                  [
                      ("pwf", {"beta": 1.0}),
                      ("rpate", {"alpha": 0.5}),
                      ("adc", {})
                  ]

            - **Config file (str)**: Path to a YAML file containing metric definitions.

    Attributes:
        dataset_evaluations (list):
            Loaded datasets in normalized format:  
            (name, y_true, y_pred_binary, y_pred_continuous, y_pred)

        metrics (list):
            List of metrics with their configurations.

    Raises:
        ValueError:
            If a configuration file is invalid or required fields are missing.
    """

    def __init__(self, dataset_evaluations, metrics=None):
        

        # Case 1: global config file -> load datasets and metrics from config
        if isinstance(dataset_evaluations, str) and metrics is None:
            config_file = dataset_evaluations
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # unir lista de dicts en un solo dict
            if isinstance(config, list):
                merged_config = {}
                for entry in config:
                    if not isinstance(entry, dict):
                        raise ValueError(f"Invalid entry in config file: {entry}")
                    merged_config.update(entry)
                config = merged_config

            if not isinstance(config, dict):
                raise ValueError("Global config file must define datasets and metrics_config as a mapping.")

            if "metrics_config" not in config:
                raise ValueError("Global config file must contain 'metrics_config'.")

            # separar datasets de la ruta de métricas
            datasets = [(name, path) for name, path in config.items() if name != "metrics_config"]

            self.dataset_evaluations = self._load_datasets(datasets)
            self.metrics = Registry.load_metrics_from_file(config["metrics_config"])
            return

        # Case 2: datasets provided directly, metrics may be list or str
        self.dataset_evaluations = self._load_datasets(dataset_evaluations)

        if isinstance(metrics, str):
            self.metrics = Registry.load_metrics_from_file(metrics)
        else:
            self.metrics = metrics

    def _load_datasets(self, dataset_evaluations):
        loaded = []
        for entry in dataset_evaluations:
            name, data = entry[0], entry[1:]
            if len(data) == 1 and isinstance(data[0], str):
                # Case: File reference
                df = pd.read_csv(data[0], sep=';')
                y_true = df["y_true"].values

                if "y_pred_binary" in df.columns and "y_pred_continuous" in df.columns:
                    y_pred_binary = df["y_pred_binary"].values
                    y_pred_continuous = df["y_pred_continuous"].values
                elif "y_pred" in df.columns:
                    y_pred = df["y_pred"].values
                    y_pred_binary, y_pred_continuous = None, None
                else:
                    raise ValueError(
                        f"File {data[0]} must contain either "
                        f"(y_pred_binary, y_pred_continuous) or y_pred column."
                    )
            else:
                # Case: Direct data
                if len(data) == 2 and isinstance(data[1], tuple):
                    # Format: y_true, (y_pred_binary, y_pred_continuous)
                    y_true, (y_pred_binary, y_pred_continuous) = data
                elif len(data) == 2:
                    # Format: y_true, y_pred
                    y_true, y_pred = data
                    y_pred_binary, y_pred_continuous = None, None
                else:
                    raise ValueError("Invalid dataset format.")
            loaded.append(
                (name, y_true, y_pred_binary, y_pred_continuous, locals().get("y_pred", None))
            )
        return loaded

    def run(self, generate_report=False, report_file="evaluation_report.csv"):
        """
        Run the evaluation for all datasets and metrics.

        Parameters:
            generate_report (bool, optional):
                If True, generates a report of the evaluation results.
                Defaults to False.

            report_file (str, optional):
                Path where the report will be saved if `generate_report` is True.
                Defaults to "evaluation_report.csv".

        Returns:
            pd.DataFrame:
                DataFrame structured as follows:

                - The **first row** contains the parameters of each metric.
                - The **subsequent rows** contain the metric values for each dataset.
                - The **index** column represents the dataset names, with the first row labeled as 'params'.

                Example::

                    dataset   | metric1       | metric2
                    ----------|---------------|--------
                    params    | {'param1':0.2}| {}
                    dataset1  | 0.5           | 1.0
                    dataset2  | 0.125         | 1.0
        """
        results = {}
        metric_keys = {}  

        for dataset_name, y_true, y_pred_binary, y_pred_continuous, y_pred in self.dataset_evaluations:
            dataset_results = {}
            for metric_name, params in self.metrics:
                metric = Registry.get_metric(metric_name, **params)

                # Computar valor según tipo de métrica
                if getattr(metric, "binary_prediction", False):
                    if y_pred_binary is not None:
                        value = metric.compute(y_true, y_pred_binary)
                    elif y_pred is not None and set(np.unique(y_pred)).issubset({0, 1}):
                        value = metric.compute(y_true, y_pred)
                    else:
                        warnings.warn(
                            f"Metric {metric_name} requires binary input, "
                            f"but dataset {dataset_name} provided non-binary predictions. Skipped.",
                            UserWarning
                        )
                        value = None
                else:
                    if y_pred_continuous is not None:
                        value = metric.compute(y_true, y_pred_continuous)
                    elif y_pred is not None and not set(np.unique(y_pred)).issubset({0, 1}):
                        value = metric.compute(y_true, y_pred)
                    else:
                        warnings.warn(
                            f"Metric {metric_name} requires continuous input, "
                            f"but dataset {dataset_name} provided binary predictions. Skipped.",
                            UserWarning
                        )
                        value = None

                # Generar clave única usando parámetros si ya existe
                base_key = metric_name
                key = f"{metric_name}({params})" if params else metric_name
                if key in metric_keys and metric_keys[key] != params:
                    key = f"{metric_name}({params})"
                metric_keys[key] = params

                dataset_results[key] = value

            results[dataset_name] = dataset_results

        # Construir DataFrame con primera fila = parámetros
        if results:
            first_dataset = next(iter(results.values()))
            metric_names = []
            metric_params = []

            for metric_config in first_dataset.keys():
                if "(" in metric_config:
                    name, params_str = metric_config.split("(", 1)
                    params_str = params_str.rstrip(")")
                else:
                    name, params_str = metric_config, ""
                metric_names.append(name)
                metric_params.append(params_str if params_str != "{}" else "")

            df_data = [metric_params]  # primera fila = parámetros
            for dataset_metrics in results.values():
                df_data.append([dataset_metrics[m] for m in dataset_metrics.keys()])

            df = pd.DataFrame(df_data, columns=metric_names)
            df.index = ["params"] + list(results.keys())
            df.index.name = 'dataset'
        else:
            df = pd.DataFrame()

        if generate_report:
            report = Report()
            report.generate_report(df, report_file)

        return df

from typing import Type
import yaml
from ..base.Metric import Metric
from .spm import *
from .tem.tpdm import *
from .tem.ptdm import *
from .tem.tmem import *
from .tem.dpm import *
from .tem.tstm import *

class Registry:
    """
    Central registry for anomaly detection metrics.

    This class provides a centralized interface to register, retrieve,
    and load metric classes for anomaly detection tasks.
    """

    _registry = {}

    @classmethod
    def register(cls, metric_cls: Type[Metric]):
        """
        Register a metric class using its `name` attribute.

        Args:
            metric_cls (Type[Metric]): The metric class to register. 
                The class must define a ``name`` attribute.

        Raises:
            ValueError: If the metric class does not define a ``name`` 
                attribute or if a metric with the same name is already registered.
        """
        if not hasattr(metric_cls, "name"):
            raise ValueError(f"Metric class {metric_cls.__name__} must define a 'name' attribute.")

        name = metric_cls.name
        if name in cls._registry:
            raise ValueError(f"Metric '{name}' is already registered.")

        cls._registry[name] = metric_cls

    @classmethod
    def get_metric(cls, name: str, **params) -> Metric:
        """
        Retrieve and instantiate a registered metric by name.

        Args:
            name (str): Name of the metric to retrieve.
            \*\*params: Parameters to initialize the metric instance.

        Returns:
            Metric: An instance of the requested metric.

        Raises:
            ValueError: If the metric name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Metric '{name}' is not registered.")
        return cls._registry[name](**params)

    @classmethod
    def available_metrics(cls):
        """
        List all registered metric names.

        Returns:
            list[str]: A list of registered metric names.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def load_metrics_info_from_file(cls, filepath: str):
        """
        Load metric definitions (names and parameters) from a YAML configuration file.

        Args:
            filepath (str): Path to the YAML file.

        Returns:
            list[tuple[str, dict]]: A list of tuples containing the metric name and 
            its parameters, e.g. ``[("metric_name", {"param1": value, ...}), ...]``.

        Raises:
            ValueError: If the YAML file contains invalid entries or unsupported format.
        """
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        metrics_info = []

        if isinstance(config, list):
            # If YAML is a list of metric names or dicts
            for entry in config:
                if isinstance(entry, str):
                    metrics_info.append((entry, {}))
                elif isinstance(entry, dict):
                    for name, params in entry.items():
                        metrics_info.append((name, params or {}))
                else:
                    raise ValueError(f"Invalid metric entry: {entry}")
        elif isinstance(config, dict):
            # If YAML is a dictionary: {metric: params}
            for name, params in config.items():
                if params is None:
                    params = {}
                metrics_info.append((name, params))
        else:
            raise ValueError("YAML format must be a list or dict.")
        return metrics_info

    @classmethod
    def load_metrics_from_file(cls, filepath: str):
        """
        Load and instantiate metrics from a YAML configuration file.

        Args:
            filepath (str): Path to the YAML configuration file.

        Returns:
            list[tuple[str, dict]]: A list of tuples containing the metric name and 
            the parameters used to instantiate it.
        """
        metrics_info = cls.load_metrics_info_from_file(filepath)
        metrics = []
        for name, params in metrics_info:
            metric = cls.get_metric(name, **params)
            metrics.append((name, params))
        return metrics


# --- Auto-discovery
def auto_register():
    """
    Automatically register all subclasses of ``Metric`` found in the project.

    This function inspects the current inheritance tree of ``Metric`` and 
    registers each subclass in the central registry.
    """
    for metric_cls in Metric.__subclasses__():
        Registry.register(metric_cls)

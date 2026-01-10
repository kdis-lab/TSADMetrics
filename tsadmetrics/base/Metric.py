import yaml
import numpy as np

class Metric:
    """
    Base class for time series anomaly detection metrics.

    This class provides common functionality for metric configuration, including
    parameter validation from a YAML configuration file and support for a parameter
    schema defined in each subclass.

    Attributes:
        name (str):
            Name of the metric instance.
        params (dict):
            Dictionary of parameters used by the metric.
        binary_prediction (bool):
            Whether the metric expects binary predictions (True) or continuous scores (False).
    
    Parameters:
        name (str, optional):
            The name of the metric. If not provided, it defaults to the lowercase
            name of the subclass.
        config_file (str, optional):
            Path to a YAML configuration file. Parameters defined in the file under
            the metric's name will be loaded automatically.
        \*\*params:
            Additional parameters passed directly to the metric. These override
            those loaded from the configuration file.

    

    Raises:
        ValueError:
            If a required parameter is missing or if the configuration file is not found.
        TypeError:
            If a parameter does not match its expected type as defined in the schema.
    """

    def __init__(self, name=None, config_file=None, **params):
        self.name = name or self.__class__.__name__.lower()

        # Ensure subclasses define binary_prediction
        if not hasattr(self.__class__, "binary_prediction"):
            raise ValueError(
                f"Subclass {self.__class__.__name__} must define class attribute 'binary_prediction' (True/False)."
            )
        if not isinstance(self.__class__.binary_prediction, bool):
            raise TypeError(
                f"'binary_prediction' in {self.__class__.__name__} must be of type bool."
            )

        self.binary_prediction = self.__class__.binary_prediction
        self.params = {}
        self.configure(config_file=config_file, **params)


    def configure(self, config_file=None, **params):
        """
        Load and validate metric parameters from a YAML configuration file
        and/or from explicit keyword arguments.

        Parameters:
            config_file (str, optional):
                Path to the configuration file. If provided, it will load parameters
                under the section with the metric's name.
            \*\*params:
                Parameters passed directly to the metric instance.

        Raises:
            ValueError:
                If a required parameter is not specified or the configuration file is missing.
            TypeError:
                If a parameter value does not match the expected type.
        """
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    file_params = config.get(self.name.lower(), {})
                    self.params.update(file_params)
            except FileNotFoundError:
                raise ValueError(f"Configuration file '{config_file}' not found.")

        self.params.update(params)

        schema = getattr(self.__class__, 'param_schema', {})
        for key, rules in schema.items():
            if key not in self.params:
                if 'default' in rules:
                    self.params[key] = rules['default']
                else:
                    raise ValueError(f"Required parameter '{key}' not specified.")

            if 'type' in rules and key in self.params:
                expected_type = rules['type']
                if expected_type is float:
                    if not isinstance(self.params[key], (float, int)):
                        raise TypeError(f"Parameter '{key}' must be of type float, got {type(self.params[key]).__name__} instead.")
                else:
                    if not isinstance(self.params[key], expected_type):
                        raise TypeError(f"Parameter '{key}' must be of type {expected_type.__name__}, got {type(self.params[key]).__name__} instead.")

    def _validate_inputs(self, y_true, y_pred):
        """
        Validate that y_true and y_pred are valid sequences of the same length.

        If binary_prediction = True:
            Both y_true and y_pred must be binary (0 or 1).
        If binary_prediction = False:
            y_true must be binary (0 or 1), y_pred can be continuous values.

        Raises:
            ValueError: If lengths differ or values are not valid.
            TypeError: If inputs are not array-like.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true has shape {y_true.shape}, y_pred has shape {y_pred.shape}."
            )

        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("y_true and y_pred must be 1D arrays.")

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("y_true must contain only 0 or 1.")

        if self.binary_prediction:
            if not np.isin(y_pred, [0, 1]).all():
                raise ValueError("y_pred must contain only 0 or 1 (binary_prediction=True).")

        return y_true, y_pred
    
    def _compute(self, y_true, y_pred):
        """
        Compute the value of the metric (core implementation).

        This method contains the actual logic of the metric and must be 
        implemented by subclasses. It is automatically called by 
        `compute()` after input validation.

        Parameters:
            y_true (array-like):
                Ground truth binary labels.
            y_pred (array-like):
                Predicted binary labels.

        Returns:
            float: The value of the metric.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement _compute().")



    def compute(self, y_true, y_pred):
        """
        Compute the value of the metric (wrapper method).

        This method performs input validation and then calls the internal
        `_compute()` method, which contains the actual metric logic. 

        **Important:** Subclasses **should not override** this method. 
        Instead, implement `_compute()` to define the behavior of the metric.

        Parameters:
            y_true (array-like):
                Ground truth binary labels.
            y_pred (array-like):
                Predicted binary labels.
        Returns:
            float: The value of the metric..
        Raises:
            NotImplementedError
                If `_compute()` is not implemented by the subclass.
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        return self._compute(y_true, y_pred)


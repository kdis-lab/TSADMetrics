Quickstart
==========

This guide presents the simplest way to compute a metric with
**TSADmetrics**. Metrics receive two time series of equal length:

* ``y_true``: the ground-truth labels of the time series;
* ``y_pred``: the predictions produced by the detector.

Before selecting a metric, consult :doc:`data_and_predictions` to determine
whether it requires binary predictions or accepts continuous anomaly scores.

Installation
------------

Install the library from PyPI:

.. code-block:: console

    pip install tsadmetrics

First evaluation with binary predictions
-----------------------------------------

Binary metrics receive a time series containing only ``0`` and ``1``. For
example, ``PointadjustedFScore`` evaluates whether anomalous events have been
detected while taking their temporal structure into account:

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm import PointadjustedFScore

    y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    y_pred = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    metric = PointadjustedFScore()
    score = metric.compute(y_true, y_pred)

    print(score)

``y_true`` and ``y_pred`` must have the same length. Binary metrics do not
automatically convert continuous scores into labels; if the detector produces
scores, apply a threshold before passing the predictions to the metric.

Evaluation using continuous anomaly scores
------------------------------------------

Score-based metrics, such as ``PointwiseAucRoc``, receive a continuous value for
each time point. In general, larger values should indicate a higher degree of
anomalousness.

.. code-block:: python

    from tsadmetrics.metrics.spm import PointwiseAucRoc

    y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    anomaly_scores = [0.05, 0.10, 0.40, 0.80, 0.70,
                      0.15, 0.10, 0.30, 0.25, 0.05]

    metric = PointwiseAucRoc()
    score = metric.compute(y_true, anomaly_scores)

    print(score)

For this type of metric, ``y_true`` remains binary, whereas ``y_pred`` may
contain continuous values. The metric does not apply a threshold internally.

Next steps
----------

To evaluate multiple datasets and metrics in a single execution, consult
:doc:`usage`. For a detailed description of the supported input formats and
how to select the appropriate prediction type, consult
:doc:`data_and_predictions`.

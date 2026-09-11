Data and predictions
====================

All **TSADmetrics** metrics compare a time series of ground-truth labels with
a time series of predictions. Both series must refer to the same temporal
instants and have exactly the same shape.

The ``y_true`` format
---------------------

``y_true`` is a one-dimensional series of binary labels:

* ``0`` represents a normal time point;
* ``1`` represents an anomalous time point.

Por ejemplo:

.. code-block:: python

    y_true = [0, 0, 0, 1, 1, 1, 0, 0]

Labels must contain only ``0`` and ``1``. Dates, event names, and intervals of
the form ``(start, end)`` must not be passed directly. Intervals must first be
converted into a binary time series.

The ``y_pred`` format
---------------------

The appropriate format for ``y_pred`` depends on the selected metric.

Binary predictions
~~~~~~~~~~~~~~~~~~

A binary prediction directly indicates whether the detector classifies each
time point as anomalous:

.. code-block:: python

    y_pred = [0, 0, 1, 1, 0, 0, 0, 1]

It must contain only ``0`` and ``1``. This is the format used by metrics whose
``binary_prediction`` attribute is ``True``, including:

* ``PointadjustedFScore``;
* ``SegmentwiseFScore``;
* ``DetectionAccuracyInRange``;
* ``TimeseriesAwareFScore``.

If the detector produces continuous scores, apply the threshold outside
TSADmetrics:

.. code-block:: python

    threshold = 0.5
    y_pred = [int(score >= threshold) for score in anomaly_scores]

The selected threshold may substantially affect the result and should
therefore be reported alongside the evaluation.

Continuous anomaly scores
~~~~~~~~~~~~~~~~~~~~~~~~~

A continuous anomaly score expresses the degree of anomalousness at each time
point:

.. code-block:: python

    anomaly_scores = [0.05, 0.10, 0.85, 0.90, 0.20, 0.05, 0.15, 0.70]

This format is used, among others, by AUC metrics such as ``PointwiseAucRoc``
and ``PointwiseAucPr``. In general, larger values should indicate a higher
probability or intensity of anomalousness. These metrics evaluate the ranking
of the scores and do not require a single operating threshold.

The ground-truth series remains binary; only ``y_pred`` contains continuous
values.

Determining the required prediction type
-----------------------------------------

Each metric class declares the ``binary_prediction`` attribute:

.. code-block:: python

    from tsadmetrics.metrics.spm import PointwiseAucRoc
    from tsadmetrics.metrics.tem.tpdm import PointadjustedFScore

    print(PointwiseAucRoc.binary_prediction)       # False
    print(PointadjustedFScore.binary_prediction)   # True

As a practical rule:

* ``True``: provide binary labels;
* ``False``: provide continuous scores while keeping ``y_true`` binary.

When both prediction types are required with ``Runner``, they can be supplied
explicitly:

.. code-block:: python

    dataset = (
        "my_dataset",
        y_true,
        (y_pred_binary, anomaly_scores),
    )

This ensures that each metric receives the representation it requires.

CSV files for ``Runner``
------------------------

When datasets are loaded from files, the CSV must include ``y_true`` and one
of the following alternatives:

* ``y_pred`` for a single prediction representation;
* ``y_pred_binary`` and ``y_pred_continuous`` to provide both representations.

CSV files used by ``Runner`` are read with ``;`` as the separator. Columns must
have the same length and must follow the same temporal ordering.

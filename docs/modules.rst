API Reference
#############

Base
====
.. automodule:: tsadmetrics.base.Metric
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:

Evaluation
==========
.. automodule:: tsadmetrics.evaluation.Report
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tsadmetrics.evaluation.Runner
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
=======

Registry
--------
.. automodule:: tsadmetrics.metrics.Registry
   :members:
   :undoc-members:
   :show-inheritance:

Metric Types
------------

Single-Point Based Metrics (SPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These metrics evaluate predictions by considering each point independently, without taking into account the temporal context in which anomalies occur. In other words, they treat each instant in isolation, ignoring the continuity or structure of anomalies over time.

.. automodule:: tsadmetrics.metrics.spm
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: param_schema, binary_prediction, name


Temporal Evaluation Metrics (TEM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This category includes metrics that incorporate temporal context in the evaluation process. They consider not only whether an anomaly was detected, but also when and how it occurred relative to the original sequence. These metrics are specifically designed for time series anomaly detection and are particularly useful for analyzing one or more properties of the model related to the temporal structure of anomalies, such as their duration, anticipation, coverage, or overlap.

Tolerant Partial Detection Metrics (TPDM)
"""""""""""""""""""""""""""""""""""""""""
These metrics consider a predicted anomaly valid if it occurs at any point within the interval of a real anomaly. They assume that partial detection is sufficient to signal a potential anomaly, allowing further verification.

.. automodule:: tsadmetrics.metrics.tem.tpdm
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: param_schema, binary_prediction, name


Precise Temporal Detection Metrics (PTDM)
"""""""""""""""""""""""""""""""""""""""""
These metrics require the predicted anomaly to cover a significant portion of the real anomaly’s duration. They are stricter about temporal accuracy than other types and value precise detection over partial alignment.

.. automodule:: tsadmetrics.metrics.tem.ptdm
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: param_schema, binary_prediction, name


Temporal Matching Evaluation Metrics (TMEM)  
"""""""""""""""""""""""""""""""""""""""""""
These metrics measure how well real and predicted anomalies are aligned, penalizing temporal deviations in start, duration, or end of the events.

.. automodule:: tsadmetrics.metrics.tem.tmem
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: param_schema, binary_prediction, name


Delay-Penalized Metrics (DPM)
"""""""""""""""""""""""""""""
These metrics penalize predictions that occur significantly after the real anomaly starts, incentivizing early detection.

.. automodule:: tsadmetrics.metrics.tem.dpm
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: param_schema, binary_prediction, name


Temporal Shift-Tolerant Metrics (TSTM)
""""""""""""""""""""""""""""""""""""""
These metrics allow a temporal tolerance in detecting an anomaly, considering predictions correct if they occur near the real event, even if they do not exactly match its start or end. This flexibility is useful when exact timing is less critical, but detection within a reasonable window is important.

.. automodule:: tsadmetrics.metrics.tem.tstm
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: param_schema, binary_prediction, name


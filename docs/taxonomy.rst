Metric taxonomy
===============

**TSADmetrics** organizes its metrics according to the amount and type of
temporal context considered during evaluation. This taxonomy is not merely an
implementation detail: it determines what constitutes a correct detection and
how partial detections, temporal overlap, detection delay, and timing
tolerance are treated.

The taxonomy introduced by TSADmetrics is discussed in the accompanying
software publication. The appropriate metric depends on the operational
meaning of a successful detection, rather than on a universally optimal
metric.

Taxonomy at a glance
--------------------

The library is organized into point-based metrics and temporal evaluation
metrics:

.. code-block:: text

    TSADmetrics
    |-- SPM  Single-Point-based Metrics
    `-- TEM  Temporal Evaluation Metrics
        |-- TPDM  Tolerant Partial Detection Metrics
        |-- PTDM  Precise Temporal Detection Metrics
        |-- TMEM  Temporal Matching Evaluation Metrics
        |-- DPM   Delay-Penalized Metrics
        `-- TSTM  Temporal Shift-Tolerant Metrics

The main distinction is whether the evaluation treats each point
independently or models the temporal structure of anomalous events.

Selecting a metric family
-------------------------

Use the following questions as a starting point:

* **Are individual time points the primary unit of evaluation?** Use SPM.
* **Is detecting any part of an anomalous event sufficient?** Use TPDM.
* **Does the proportion of an event that was detected matter?** Use PTDM.
* **Does the temporal alignment between real and predicted events matter?** Use TMEM.
* **Does delayed detection have an operational cost?** Use DPM.
* **Should slightly early or late detections receive partial credit?** Use TSTM.

These choices are complementary rather than mutually exclusive. In a
scientific evaluation, it is often appropriate to report several metrics from
different families when they measure different aspects of performance.

Metric families
---------------

Single-Point-based Metrics (SPM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPM metrics evaluate predictions independently at each time point. They are
appropriate when temporal continuity and event structure are not part of the
evaluation objective.

Examples: :ref:`specific-spm`; API reference: :ref:`api-spm`.

.. list-table:: SPM metrics
   :header-rows: 1
   :widths: 35 20 45

   * - Metric
     - Alias
     - Prediction type
   * - :py:class:`~tsadmetrics.metrics.spm.PointwiseFScore.PointwiseFScore`
     - ``pwf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.spm.DiceCoefficient.DiceCoefficient`
     - ``dicec``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.spm.PointwiseAucRoc.PointwiseAucRoc`
     - ``pw_auc_roc``
     - Continuous scores
   * - :py:class:`~tsadmetrics.metrics.spm.PointwiseAucPr.PointwiseAucPr`
     - ``pw_auc_pr``
     - Continuous scores
   * - :py:class:`~tsadmetrics.metrics.spm.PrecisionAtK.PrecisionAtK`
     - ``pak``
     - Continuous scores

Temporal Evaluation Metrics (TEM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TEM metrics incorporate temporal context. They consider not only whether an
anomaly was detected, but also how the prediction relates to the duration,
location, or timing of the real event.

Tolerant Partial Detection Metrics (TPDM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TPDM metrics consider a prediction valid when it detects part of a real
anomalous event. They are suitable when an initial signal is sufficient to
trigger further investigation.

Examples: :ref:`specific-tpdm`; API reference: :ref:`api-tpdm`.

.. list-table:: TPDM metrics
   :header-rows: 1
   :widths: 35 20 45

   * - Metric
     - Alias
     - Prediction type
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.PointadjustedFScore.PointadjustedFScore`
     - ``paf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.BalancedPointadjustedFScore.BalancedPointadjustedFScore`
     - ``bpaf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.SegmentwiseFScore.SegmentwiseFScore`
     - ``swf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.CompositeFScore.CompositeFScore`
     - ``cf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.PointadjustedAucPr.PointadjustedAucPr`
     - ``pa_auc_pr``
     - Continuous scores
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.PointadjustedAucRoc.PointadjustedAucRoc`
     - ``pa_auc_roc``
     - Continuous scores
   * - :py:class:`~tsadmetrics.metrics.tem.tpdm.RangebasedFScore.RangebasedFScore`
     - ``rbf``
     - Binary

Precise Temporal Detection Metrics (PTDM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PTDM metrics require the prediction to cover a meaningful portion of the real
anomalous event. They are appropriate when event coverage is more important
than detecting only its first point.

Examples: :ref:`specific-ptdm`; API reference: :ref:`api-ptdm`.

.. list-table:: PTDM metrics
   :header-rows: 1
   :widths: 35 20 45

   * - Metric
     - Alias
     - Prediction type
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.AverageDetectionCount.AverageDetectionCount`
     - ``adc``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.TotalDetectedInRange.TotalDetectedInRange`
     - ``tdir``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.DetectionAccuracyInRange.DetectionAccuracyInRange`
     - ``dair``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.WeightedDetectionDifference.WeightedDetectionDifference`
     - ``wdd``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.PointadjustedAtKFScore.PointadjustedAtKFScore`
     - ``pakf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.PointadjustedAtKLFScore.PointadjustedAtKLFScore`
     - ``paklf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.ptdm.TimeseriesAwareFScore.TimeseriesAwareFScore`
     - ``taf``
     - Binary

Temporal Matching Evaluation Metrics (TMEM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TMEM metrics measure the temporal alignment between real and predicted
events. They penalize deviations in event start, duration, or end position.

Examples: :ref:`specific-tmem`; API reference: :ref:`api-tmem`.

.. list-table:: TMEM metrics
   :header-rows: 1
   :widths: 35 20 45

   * - Metric
     - Alias
     - Prediction type
   * - :py:class:`~tsadmetrics.metrics.tem.tmem.AbsoluteDetectionDistance.AbsoluteDetectionDistance`
     - ``add``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tmem.TemporalDistance.TemporalDistance`
     - ``td``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tmem.EnhancedTimeseriesAwareFScore.EnhancedTimeseriesAwareFScore`
     - ``etaf``
     - Binary

Delay-Penalized Metrics (DPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DPM metrics penalize predictions that occur after the beginning of a real
anomalous event. They are appropriate when rapid detection is operationally
important.

Examples: :ref:`specific-dpm`; API reference: :ref:`api-dpm`.

.. list-table:: DPM metrics
   :header-rows: 1
   :widths: 35 20 45

   * - Metric
     - Alias
     - Prediction type
   * - :py:class:`~tsadmetrics.metrics.tem.dpm.DelayThresholdedPointadjustedFScore.DelayThresholdedPointadjustedFScore`
     - ``dtpaf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.dpm.EarlyDetectionScore.EarlyDetectionScore`
     - ``eds``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.dpm.LatencySparsityawareFScore.LatencySparsityawareFScore`
     - ``lsaf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.dpm.MeanTimeToDetect.MeanTimeToDetect`
     - ``mttd``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.dpm.NabScore.NabScore`
     - ``nab_score``
     - Binary

Temporal Shift-Tolerant Metrics (TSTM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TSTM metrics allow a degree of temporal tolerance. Predictions may receive
credit when they occur near the real event, even when their boundaries do not
coincide exactly.

Examples: :ref:`specific-tstm`; API reference: :ref:`api-tstm`.

.. list-table:: TSTM metrics
   :header-rows: 1
   :widths: 35 20 45

   * - Metric
     - Alias
     - Prediction type
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.AffiliationbasedFScore.AffiliationbasedFScore`
     - ``aff_f``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.NormalizedAffiliationbasedFScore.NormalizedAffiliationbasedFScore`
     - ``naff_f``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.PateFScore.PateFScore`
     - ``pate_f1``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.Pate.Pate`
     - ``pate``
     - Continuous scores
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.TimeTolerantFScore.TimeTolerantFScore`
     - ``ttf``
     - Binary
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.VusRoc.VusRoc`
     - ``vus_roc``
     - Continuous scores
   * - :py:class:`~tsadmetrics.metrics.tem.tstm.VusPr.VusPr`
     - ``vus_pr``
     - Continuous scores

Further information
--------------------

For the input contract shared by the metrics, see
:doc:`data_and_predictions`. For copy-pasteable examples, see
:doc:`specific_usage`. For complete class and method documentation, see
:doc:`modules`.

The taxonomy and its terminology are discussed in the TSADmetrics software
publication:

Velasco, P. R. and Zafra, A. (2026). *TSADmetrics: A library for evaluating
time series anomaly detection methods*. Neurocomputing, 698, 134154.
https://doi.org/10.1016/j.neucom.2026.134154

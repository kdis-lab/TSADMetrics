

Specific Metric Usage
=====================

This page provides simple, copy-pasteable examples for using individual metrics.
Each section below shows how to use one metric directly and with the Runner.

SPM Metrics
-----------


PointwiseFScore
~~~~~~~~~~~~~~~

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.spm.PointwiseFScore import PointwiseFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = PointwiseFScore(beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("PointwiseFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pwf", {"beta":1})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PrecisionAtK
~~~~~~~~~~~~

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.spm.PrecisionAtK import PrecisionAtK

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = PrecisionAtK()
    result = metric.compute(y_true, y_pred)
    print("PrecisionAtK:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pak", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PointwiseAucRoc
~~~~~~~~~~~~~~~

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.spm.PointwiseAucRoc import PointwiseAucRoc

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = PointwiseAucRoc()
    result = metric.compute(y_true, y_pred)
    print("PointwiseAucRoc:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pw_auc_roc", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PointwiseAucPr
~~~~~~~~~~~~~~

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.spm.PointwiseAucPr import PointwiseAucPr

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = PointwiseAucPr()
    result = metric.compute(y_true, y_pred)
    print("PointwiseAucPr:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pw_auc_pr", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

MET Metrics
-----------

TPDM Metrics
~~~~~~~~~~~~


CompositeFScore
^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm.CompositeFScore import CompositeFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = CompositeFScore(beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("CompositeFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("cf", {"beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PointadjustedAucPr
^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm.PointadjustedAucPr import PointadjustedAucPr

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = PointadjustedAucPr()
    result = metric.compute(y_true, y_pred)
    print("PointadjustedAucPr:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pa_auc_pr", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PointadjustedAucRoc
^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm.PointadjustedAucRoc import PointadjustedAucRoc

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = PointadjustedAucRoc()
    result = metric.compute(y_true, y_pred)
    print("PointadjustedAucRoc:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pa_auc_roc", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

SegmentwiseFScore
^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm.SegmentwiseFScore import SegmentwiseFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = SegmentwiseFScore(beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("SegmentwiseFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("swf", {"beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

RangebasedFScore
^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm.RangebasedFScore import RangebasedFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = RangebasedFScore(beta=1.0, p_alpha=0.5, r_alpha=0.5, p_bias="flat", r_bias="flat", cardinality_mode="one")
    result = metric.compute(y_true, y_pred)
    print("RangebasedFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("rbf", {"beta":1.0, "p_alpha":0.5, "r_alpha":0.5, "p_bias":"flat", "r_bias":"flat", "cardinality_mode":"one"})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PointadjustedFScore
^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tpdm.PointadjustedFScore import PointadjustedFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = PointadjustedFScore(beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("PointadjustedFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("paf", {"beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PTDM Metrics
~~~~~~~~~~~~


AverageDetectionCount
^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.ptdm.AverageDetectionCount import AverageDetectionCount

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = AverageDetectionCount()
    result = metric.compute(y_true, y_pred)
    print("AverageDetectionCount:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("adc", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

DetectionAccuracyInRange
^^^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.ptdm.DetectionAccuracyInRange import DetectionAccuracyInRange

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = DetectionAccuracyInRange(k=5)
    result = metric.compute(y_true, y_pred)
    print("DetectionAccuracyInRange:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("dair", {"k":5})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PointadjustedAtKFScore
^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.ptdm.PointadjustedAtKFScore import PointadjustedAtKFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = PointadjustedAtKFScore(k=0.5, beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("PointadjustedAtKFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pakf", {"k":0.5, "beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

TimeseriesAwareFScore
^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.ptdm.TimeseriesAwareFScore import TimeseriesAwareFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = TimeseriesAwareFScore(beta=1.0,alpha=0.5, delta=5, theta=0.5,past_range=False)
    result = metric.compute(y_true, y_pred)
    print("TimeseriesAwareFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("taf", {"beta":1.0, "alpha":0.5, "delta":5, "theta":0.5, "past_range":False})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

TotalDetectedInRange
^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.ptdm.TotalDetectedInRange import TotalDetectedInRange

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = TotalDetectedInRange(k=5)
    result = metric.compute(y_true, y_pred)
    print("TotalDetectedInRange:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("tdir", {"k":5})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

WeightedDetectionDifference
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.ptdm.WeightedDetectionDifference import WeightedDetectionDifference

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = WeightedDetectionDifference(k=5)
    result = metric.compute(y_true, y_pred)
    print("WeightedDetectionDifference:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("wdd", {"k":5})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

TMEM Metrics
~~~~~~~~~~~~


AbsoluteDetectionDistance
^^^^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tmem.AbsoluteDetectionDistance import AbsoluteDetectionDistance

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = AbsoluteDetectionDistance()
    result = metric.compute(y_true, y_pred)
    print("AbsoluteDetectionDistance:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("add", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

TemporalDistance
^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tmem.TemporalDistance import TemporalDistance

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = TemporalDistance(distance=0)
    result = metric.compute(y_true, y_pred)
    print("TemporalDistance:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("td", {"distance":0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

EnhancedTimeseriesAwareFScore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tmem.EnhancedTimeseriesAwareFScore import EnhancedTimeseriesAwareFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = EnhancedTimeseriesAwareFScore(theta_p=0.5, theta_r=0.5)
    result = metric.compute(y_true, y_pred)
    print("EnhancedTimeseriesAwareFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("etaf", {"theta_p":0.5, "theta_r":0.5})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

DPM metrics
~~~~~~~~~~~


DelayThresholdedPointadjustedFScore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.dpm.DelayThresholdedPointadjustedFScore import DelayThresholdedPointadjustedFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = DelayThresholdedPointadjustedFScore(k=1, beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("DelayThresholdedPointadjustedFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("dtpaf", {"k":1, "beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

LatencySparsityawareFScore
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.dpm.LatencySparsityawareFScore import LatencySparsityawareFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = LatencySparsityawareFScore(ni=1, beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("LatencySparsityawareFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("lsaf", {"ni":1, "beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

MeanTimeToDetect
^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.dpm.MeanTimeToDetect import MeanTimeToDetect

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = MeanTimeToDetect()
    result = metric.compute(y_true, y_pred)
    print("MeanTimeToDetect:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("mttd", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

NabScore
^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.dpm.NabScore import NabScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = NabScore()
    result = metric.compute(y_true, y_pred)
    print("NabScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("nab_score", {})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

TSTM Metrics
~~~~~~~~~~~~


AffiliationbasedFScore
^^^^^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tstm.AffiliationbasedFScore import AffiliationbasedFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = AffiliationbasedFScore(beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("AffiliationbasedFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("aff_f", {"beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

TimeTolerantFScore
^^^^^^^^^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tstm.TimeTolerantFScore import TimeTolerantFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = TimeTolerantFScore(t=5, beta=1.0)
    result = metric.compute(y_true, y_pred)
    print("TimeTolerantFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("ttf", {"t":5, "beta":1.0})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

VusPr
^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tstm.VusPr import VusPr

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = VusPr(window=4)
    result = metric.compute(y_true, y_pred)
    print("VusPr:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("vus_pr", {"window":4})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

VusRoc
^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tstm.VusRoc import VusRoc

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = VusRoc(window=4)
    result = metric.compute(y_true, y_pred)
    print("VusRoc:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("vus_roc", {"window":4})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

PateFScore
^^^^^^^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tstm.PateFScore import PateFScore

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]

    metric = PateFScore(early=5, delay=5)
    result = metric.compute(y_true, y_pred)
    print("PateFScore:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pate_f1", {"early":5, "delay":5})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

Pate
^^^^

**Manual evaluation**

.. code-block:: python

    from tsadmetrics.metrics.tem.tstm.Pate import Pate

    y_true = [0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    y_pred = [0,0,0,0,0,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    metric = Pate(early=5, delay=5)
    result = metric.compute(y_true, y_pred)
    print("Pate:", result)

**Automated evaluation**

.. code-block:: python

    from tsadmetrics.evaluation.Runner import Runner

    dataset_evaluations = [
        ("dataset1", y_true, y_pred)  
    ]

    metrics = [
        ("pate", {"early":5, "delay":5})
    ]

    runner = Runner(dataset_evaluations, metrics)
    results = runner.run()
    print(results)

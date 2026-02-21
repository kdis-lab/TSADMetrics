# TSADmetrics - Time Series Anomaly Detection Metrics

**TSADmetrics** is a Python library for evaluating anomaly detection algorithms in time series data.  
It provides a comprehensive set of metrics specifically designed to handle the temporal nature of anomalies.

---

## Key features

- **Metric Taxonomy**: Metrics are categorized into types based on how they handle temporal context:

  - **MPI Metrics**: Evaluate predictions at each point independently, ignoring temporal continuity.
  - **MET Metrics**: Consider temporal context, analyzing when and how anomalies occur.
    - **MDPT**: Partial detection within a real anomaly event counts as correct.
    - **MDTP**: Requires detection to cover a significant fraction of the real anomaly.
    - **MECT**: Measures alignment of real vs predicted anomaly events.
    - **MPR**: Penalizes late detections.
    - **MTDT**: Allows temporal tolerance for early or late detections.

- **Direct Metric Usage**: Instantiate any metric class and call `compute()` for individual evaluation.

- **Batch Evaluation**: Use `Runner` to evaluate multiple datasets and metrics at once, with support for both direct data and CSV/JSON input.

- **Flexible Configuration**: Load metrics from YAML configuration files or global evaluation config files.


---

## Getting the library

The project can be download in this repository. Detailed information about getting and running the library can be consulted in user’s manual where all dependencies and libraries necessaries are commented.

---

## Installation

Install TSADmetrics via pip:

```bash
pip install tsadmetrics
```

## Tutorials and documentation

The complete documentation for TSADmetrics is available at:  
📚 [https://tsadmetrics.readthedocs.io/](https://tsadmetrics.readthedocs.io/)

User's manual can be found in the main directory documentation of this repository and includes:
- Getting and installing the library.
- A description of library architecture.
- Run metrics in the library.
- Add new metrics in the library.
- API reference.

## Metric included

TSADmetrics includes 34 metrics, organized according to the proposed taxonomy for time series anomaly detection metrics.

| Category | Subcategory | Metric                              |
| -------- | ----------- | ----------------------------------- |
| SPM      | —           | PointwiseFScore                     |
| SPM      | —           | DiceCoefficient                     |
| SPM      | —           | PointwiseAucRoc                     |
| SPM      | —           | PointwiseAucPr                      |
| SPM      | —           | PrecisionAtK                        |
| TEM      | TPDM        | PointadjustedFScore                 |
| TEM      | TPDM        | BalancedPointadjustedFScore         |
| TEM      | TPDM        | SegmentwiseFScore                   |
| TEM      | TPDM        | CompositeFScore                     |
| TEM      | TPDM        | PointadjustedAucPr                  |
| TEM      | TPDM        | PointadjustedAucRoc                 |
| TEM      | TPDM        | RangebasedFScore                    |
| TEM      | DPM         | DelayThresholdedPointadjustedFScore |
| TEM      | DPM         | EarlyDetectionScore                 |
| TEM      | DPM         | LatencySparsityawareFScore          |
| TEM      | DPM         | MeanTimeToDetect                    |
| TEM      | DPM         | NabScore                            |
| TEM      | PTDM        | AverageDetectionCount               |
| TEM      | PTDM        | TotalDetectedInRange                |
| TEM      | PTDM        | DetectionAccuracyInRange            |
| TEM      | PTDM        | WeightedDetectionDifference         |
| TEM      | PTDM        | PointadjustedAtKFScore              |
| TEM      | PTDM        | PointadjustedAtKLFScore             |
| TEM      | PTDM        | TimeseriesAwareFScore               |
| TEM      | TMEM        | AbsoluteDetectionDistance           |
| TEM      | TMEM        | EnhancedTimeseriesAwareFScore       |
| TEM      | TMEM        | TemporalDistance                    |
| TEM      | TSTM        | AffiliationbasedFScore              |
| TEM      | TSTM        | NormalizedAffiliationbasedFScore    |
| TEM      | TSTM        | PateFScore                          |
| TEM      | TSTM        | Pate                                |
| TEM      | TSTM        | TimeTolerantFScore                  |
| TEM      | TSTM        | VusRoc                              |
| TEM      | TSTM        | VusPr                               |


## Acknowledgements

This library was supported in part by the PID2023-148396NB-I00 project of Spanish Ministry of Science and Innovation and the European Regional Development Fund, by the ProyExcel-0069 project of the Andalusian University, Research and Innovation Department.

## Citation
<>

## References
This library is based on the concepts and implementations from:  
Sørbø, S., & Ruocco, M. (2023). *Navigating the metric maze: a taxonomy of evaluation metrics for anomaly detection in time series*. https://doi.org/10.1007/s10618-023-00988-8

## Licence

The tools is free and open source, under the GNU General Public GPLv3 license.

## Reporting bugs

Feel free to open an issue at Github if anything is not working as expected. Merge request are also encouraged, it will be carefully reviewed and merged if everything is all right.




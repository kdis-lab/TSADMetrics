.. TSADmetrics documentation master file

Welcome to TSADmetrics - Time Series Anomaly Detection Metrics
==============================================================

**TSADmetrics** is a Python library for evaluating anomaly detection algorithms in time series data.  
It provides a comprehensive set of binary and non-binary metrics designed specifically for the challenges of anomaly detection in temporal contexts. The metrics are organized into families according to the temporal characteristics they evaluate, such as point-wise performance, event coverage, temporal alignment, detection delay, and timing tolerance. An overview of this organization and guidance for selecting a metric are provided in the :doc:`taxonomy` section.




Documentation Contents
======================
.. toctree::
   :maxdepth: 2


   installation
   quickstart
   taxonomy
   data_and_predictions
   uninstallation
   usage

.. toctree::
   :maxdepth: 2
   :titlesonly:

   specific_usage

.. toctree::
   :maxdepth: 2

   modules


Acknowledgements
================

This library was supported in part by the PID2023-148396NB-I00 project of Spanish Ministry of Science and Innovation and the European Regional Development Fund, by the ProyExcel-0069 project of the Andalusian University, Research and Innovation Department.


References
==========

This library is based on the concepts and implementations from:  
Sørbø, S., & Ruocco, M. (2023). *Navigating the metric maze: a taxonomy of evaluation metrics for anomaly detection in time series*. https://doi.org/10.1007/s10618-023-00988-8


Citing TSADmetrics
==================

If TSADmetrics has been useful in your research, please cite the following
paper published in *Neurocomputing*:

.. code-block:: bibtex

   @article{velasco2026tsadmetrics,
     author = {Velasco, Pedro R. and Zafra, Amelia},
     title = {TSADmetrics: A library for evaluating time series anomaly detection methods},
     journal = {Neurocomputing},
     volume = {698},
     pages = {134154},
     year = {2026},
     doi = {10.1016/j.neucom.2026.134154},
     url = {https://doi.org/10.1016/j.neucom.2026.134154}
   }

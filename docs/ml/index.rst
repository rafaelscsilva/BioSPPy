Machine Learning
================

The ``biosppy.ml`` package is an optional extension for machine-learning
workflows on biosignals. The first available model is
:py:class:`biosppy.ml.ecg_ml.AFibDetection`, a pre-trained bidirectional LSTM
that detects atrial fibrillation (AFib) from RR interval sequences.

Installation
------------

Install the optional dependencies with:

.. code-block:: bash

   pip install biosppy[ml]

Package structure
-----------------

- ``biosppy.ml.utils_ml``: base utilities for Keras-based classifiers.
- ``biosppy.ml.ecg_ml``: ECG-related ML models, including AFib detection.
- ``biosppy/ml/_models``: packaged pre-trained model files and metadata.

Model architecture
------------------

:py:class:`biosppy.ml.utils_ml.KerasClassifier` is the base class used by ML
models. It validates model files, loads model metadata from JSON, and provides
shared prediction/preprocessing behavior.

:py:class:`biosppy.ml.ecg_ml.AFibDetection` extends this base class and uses a
windowed RR-interval pipeline:

1. Segment the RR sequence into windows (default ``win_len=20``, ``step=1``).
2. Reshape to ``(n_windows, win_len, 1)``.
3. Run the BiLSTM model to obtain one probability per window.
4. Return ``True`` if any probability exceeds the configured threshold.

Quick example
-------------

.. code-block:: python

   from biosppy import storage
   from biosppy.ml.ecg_ml import AFibDetection

   # RR intervals in ms
   rri, _ = storage.load_txt('examples/rri.txt')

   model = AFibDetection()
   afib = model.predict(rri)
   print(f"AFib detected: {afib}")

API links
---------

- :doc:`../biosppy.ml`
- :py:mod:`biosppy.ml.ecg_ml`
- :py:mod:`biosppy.ml.utils_ml`


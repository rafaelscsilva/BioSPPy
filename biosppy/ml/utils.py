# -*- coding: utf-8 -*-
"""
biosppy.ml.utils
----------------
This module provides utility classes and functions for Keras-based models, which can be used for various machine
learning tasks with biosignals, such as classification, segmentation and anomaly detection.

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
import numpy as np
import keras
import joblib
import json
import logging
from abc import ABC
import pathlib
import os

# suppress TensorFlow warnings
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

logger = logging.getLogger(__name__)


class KerasClassifier(ABC):
    """
    A wrapper for Keras models to perform classification tasks.

    This class loads a Keras model and an optional scaler, and provides methods
    for preprocessing input signals and making predictions. It also supports loading additional details such as label
    mappings and sampling rates from a JSON file.

    Parameters
    ----------
    model_path : str
        Path to the Keras model file.
    details_path : str
        Path to a JSON file containing the model details such as label map and sampling rate.
    scaler_path : str, optional
        Path to a scaler file (e.g., joblib file) for scaling input data. Default is None.
    """
    def __init__(self, model_path, details_path, scaler_path=None):
        # validate paths before loading anything
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not pathlib.Path(details_path).exists():
            raise FileNotFoundError(f"Details file not found: {details_path}")

        # load model
        self.model = keras.models.load_model(model_path)

        # load details
        with open(details_path, 'r') as f:
            details = json.load(f)
        for key, value in details.items():
            setattr(self, key, value)

        # load scaler
        self.scaler = None
        if scaler_path is not None:
            if not pathlib.Path(scaler_path).exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)

        # ensure minimum attributes are set
        if not hasattr(self, 'sampling_rate'):
            self.sampling_rate = None
        if not hasattr(self, 'label_map'):
            self.label_map = None
        if not hasattr(self, 'threshold'):
            self.threshold = 0.5

    def preprocess_signal(self, signal, sampling_rate=1000.0):
        """
        Preprocess the input signal based on the model's requirements, such as scaling and resampling.

        Parameters
        ----------
        signal : array-like
            The input signal to preprocess.
        sampling_rate : float, optional
            The sampling rate of the input signal. Default is 1000.0 Hz.

        Returns
        -------
        X : array-like
            The preprocessed signal ready for prediction.
        """
        logger.debug("Preprocessing signal...")
        X = np.array(signal).reshape(1, -1)

        # If a sampling rate is specified, ensure it matches the model's expected rate
        if self.sampling_rate is not None and sampling_rate != self.sampling_rate:
            from biosppy.signals.tools import resample_signal
            X = resample_signal(X, sampling_rate, self.sampling_rate)
            logger.debug("Resampling to %s Hz applied.", self.sampling_rate)

        # If a scaler is provided, apply it to the input signal
        if self.scaler:
            X = self.scaler.transform(X)
            logger.debug("Scaling applied.")

        return X

    def predict(self, signal, sampling_rate=1000.0, **kwargs):
        # preprocess the signal if needed
        X = self.preprocess_signal(signal, sampling_rate)

        # check if the model expects a specific input shape
        input_shape = self.model.input_shape
        expected_dim = input_shape[1]
        if X.shape[1] != expected_dim:
            raise ValueError(f"Input shape mismatch: expected {expected_dim} features, got {X.shape[1]} features.")

        # predict using the model
        probs = self.model.predict(X, verbose=0)

        # apply threshold if binary output (single output unit)
        if probs.shape[-1] == 1:
            class_index = 0 if probs[0, 0] < self.threshold else 1
        # otherwise, use argmax for multi-class classification
        else:
            class_index = int(np.argmax(probs[0]))

        # map class index to class name
        class_name = self.label_map[str(class_index)] if self.label_map else str(class_index)

        return class_name

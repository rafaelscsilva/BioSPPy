import numpy as np
import keras
import joblib
import json
from abc import ABC, abstractmethod
import pathlib
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    """
    def __init__(self, model_path):
        # load model
        self.model = keras.models.load_model(model_path)

        # load scaler if it exists
        self.scaler = None
        self.scaler_path = model_path.replace('.h5', '_scaler.joblib')
        if pathlib.Path(self.scaler_path).exists():
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler_path = None

        # load details if it exists
        self.details_path = model_path.replace('.h5', '_details.json')
        if pathlib.Path(self.details_path).exists():
            with open(self.details_path, 'r') as f:
                details = json.load(f)
            for key, value in details.items():
                setattr(self, key, value)
        else:
            raise FileNotFoundError(f"Details file not found: {self.details_path}")

        # ensure minimum attributes are set
        if not hasattr(self, 'sampling_rate'):
            self.sampling_rate = None
        if not hasattr(self, 'label_map'):
            self.label_map = None
        if not hasattr(self, 'threshold'):
            self.threshold = 0.5

    def preprocess_signal(self, signal, sampling_rate=1000.0):
        """
        Preprocess the input signal based on the model's requirements.
        This may include scaling and resampling.

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
        print("Preprocessing signal...")
        X = np.array(signal).reshape(1, -1)

        # If a sampling rate is specified, ensure it matches the model's expected rate
        if self.sampling_rate is not None and sampling_rate != self.sampling_rate:
            from biosppy.signals.tools import resample_signal
            X = resample_signal(X, sampling_rate, self.sampling_rate)
            print(f"- Resampling to {self.sampling_rate} Hz applied.")

        # If a scaler is provided, apply it to the input signal
        if self.scaler:
            X = self.scaler.transform(X)
            print("- Scaling applied.")

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

        # apply threshold if single output is expected
        if len(probs) == 1:
            if probs[0] < self.threshold:
                class_index = 0
            else:
                class_index = 1
        # otherwise, use argmax for multi-class classification
        else:
            class_index = int(np.argmax(probs))

        # map class index to class name
        class_name = self.label_map[class_index] if self.label_map else str(class_index)

        return class_name

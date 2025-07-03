# -*- coding: utf-8 -*-
"""
biosppy.ml.ecg
--------------

This module provides classes for machine learning models specifically designed for ECG signal analysis or derived
signals or features (e.g., RR intervals).

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
import numpy as np
import os
from .utils import KerasClassifier


class AFibDetection(KerasClassifier):
    """
    A class for detecting atrial fibrillation using a pre-trained Keras model from [Silva23]. This model uses the
    RR interval sequence as input and applies a bidirectional LSTM architecture to classify the signal.

    The signal should be provided as a one-dimensional array of RR intervals in milliseconds.

    Methods
    -------
    `predict(signal, **kwargs)`: Predicts whether the input RR interval sequence indicates atrial fibrillation.

    Usage
    -----
    from biosppy.ml.ecg import AFibDetection
    afib_model = AFibDetection()
    rri_signal = [601., 593., 585., 601., 601., 609.]  # Example RR interval sequence
    result = afib_model.predict(rri_signal)

    References
    ----------
    .. [Silva23] R. Silva, L. Abrunhosa Rodrigues, A. Lourenço, H. Plácido da Silva, "Temporal Dynamics of Drowsiness
    Detection Using LSTM-Based Models", International Work-Conference on Artificial Neural Networks, pp. 211-220, 2023.
    """

    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, '.models', 'ecg_afibdetection_bilstm.h5')
        details_path = os.path.join(base_dir, '.models', 'ecg_afibdetection_bilstm_details.json')
        super().__init__(model_path=model_path, details_path=details_path)

    def _preprocess_signal(self, signal, win_len=20, step=1):
        win_len = self.win_len if 'win_len' in self.__dict__ else win_len
        step = self.step if 'step' in self.__dict__ else step

        X_ = []
        # check the length of the RR interval sequence
        if len(signal) >= win_len:
            res = len(signal) % win_len
            if res != 0:
                signal = signal[:-res]  # slice array
            signal = np.array(signal)

            # no step
            if step == 0:
                signal = np.split(signal, int(len(signal) / win_len))  # split array
                X_.append(signal)

            else:
                n_splits = int(((len(signal) - win_len) / step) + 1)
                for j in range(0, n_splits * step, step):
                    split = signal[j:j + win_len]
                    X_.append(split)
        else:
            raise ValueError(f"RR interval sequence is too short: {len(signal)} samples, expected at least"
                             f"{win_len} samples.")

        X = np.vstack(X_)
        return X

    def predict(self, signal, **kwargs):
        """
        Predicts whether the input RR interval sequence indicates atrial fibrillation.

        Parameters
        ----------
        signal : array-like
            The input RR interval sequence as a one-dimensional array, in milliseconds.

        Returns
        -------
        bool
            Returns True if atrial fibrillation is detected, otherwise returns False.

        """
        # ensure signal has only one channel
        if signal.ndim > 1:
            raise ValueError("Input signal must be one-dimensional (single channel).")

        # predict using the model on every window
        X = self._preprocess_signal(signal, **kwargs)
        probs = self.model.predict(X, verbose=0)

        # if any prediction is above the threshold, return afib
        if np.any(probs > self.threshold):
            return True
        else:
            return False

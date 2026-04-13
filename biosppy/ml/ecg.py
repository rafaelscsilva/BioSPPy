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
    """A class for detecting atrial fibrillation using a pre-trained Keras model from [Silva23]. This model uses the
    RR interval sequence as input and applies a bidirectional LSTM architecture to classify the signal.

    The signal should be provided as a one-dimensional array of RR intervals in milliseconds.

    Methods
    -------
    predict(signal, **kwargs)
        Predicts whether the input RR interval sequence indicates atrial fibrillation.

    Examples
    --------
    >>> from biosppy.ml.ecg import AFibDetection
    >>> afib_model = AFibDetection()
    >>> rri_signal = [601., 593., 585., 601., 601., 609., ...]  # RR interval sequence (ms)
    >>> result = afib_model.predict(rri_signal)

    References
    ----------
    .. [Silva23] R. Silva, L. Abrunhosa Rodrigues, A. Lourenço, H. Plácido da Silva, "Temporal Dynamics of Drowsiness
       Detection Using LSTM-Based Models", International Work-Conference on Artificial Neural Networks,
       pp. 211-220, 2023.
    """

    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, '_models', 'ecg_afibdetection_bilstm.h5')
        details_path = os.path.join(base_dir, '_models', 'ecg_afibdetection_bilstm_details.json')
        super().__init__(model_path=model_path, details_path=details_path)

    def _preprocess_signal(self, signal, win_len=20, step=1):
        """Segment the RR interval sequence into overlapping or non-overlapping windows.

        Parameters
        ----------
        signal : array
            One-dimensional RR interval sequence, in milliseconds.
        win_len : int, optional
            Window length in number of RR intervals. Overridden by ``self.win_len`` if set
            in the model details file. Default is 20.
        step : int, optional
            Step size between consecutive windows. Use 0 for non-overlapping windows.
            Overridden by ``self.step`` if set in the model details file. Default is 1.

        Returns
        -------
        X : array
            2D array of shape (n_windows, win_len) containing the segmented windows.

        Raises
        ------
        ValueError
            If the signal is shorter than ``win_len``.

        """
        win_len = getattr(self, 'win_len', win_len)
        step = getattr(self, 'step', step)

        if len(signal) < win_len:
            raise ValueError(
                f"RR interval sequence is too short: {len(signal)} samples, "
                f"expected at least {win_len} samples."
            )

        signal = np.array(signal, dtype=float)
        X_ = []

        if step == 0:
            # non-overlapping windows: trim signal to a multiple of win_len
            res = len(signal) % win_len
            if res != 0:
                signal = signal[:-res]
            X_.extend(np.split(signal, len(signal) // win_len))
        else:
            n_splits = int(((len(signal) - win_len) / step) + 1)
            for j in range(0, n_splits * step, step):
                X_.append(signal[j:j + win_len])

        return np.vstack(X_)

    def predict(self, signal, **kwargs):
        """Predict whether the input RR interval sequence indicates atrial fibrillation.

        Parameters
        ----------
        signal : array
            One-dimensional RR interval sequence, in milliseconds.

        Returns
        -------
        afib : bool
            True if atrial fibrillation is detected, False otherwise.

        Raises
        ------
        TypeError
            If ``signal`` is None.
        ValueError
            If the signal is not one-dimensional.
        ValueError
            If the signal is shorter than the required window length.

        """
        if signal is None:
            raise TypeError("Please specify an input signal.")

        # ensure numpy and one-dimensional
        signal = np.array(signal, dtype=float)
        if signal.ndim > 1:
            raise ValueError("Input signal must be one-dimensional (single channel).")

        # segment into windows and reshape to (n_windows, win_len, 1) for the BiLSTM
        X = self._preprocess_signal(signal, **kwargs)
        X = X[:, :, np.newaxis]

        probs = self.model.predict(X, verbose=0)

        # if any window's prediction is above the threshold, classify as AFib
        return bool(np.any(probs > self.threshold))

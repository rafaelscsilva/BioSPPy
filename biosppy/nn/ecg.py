from .utils import KerasClassifier
import numpy as np


class AFibDetection(KerasClassifier):
    """
    A class for detecting atrial fibrillation using a pre-trained Keras model from [].

    References
    ----------
    .. [Silva23] R. Silva, L. Abrunhosa Rodrigues, A. Lourenço, H. Plácido da Silva, "Temporal Dynamics of Drowsiness
    Detection Using LSTM-Based Models", International Work-Conference on Artificial Neural Networks, pp. 211-220, 2023.
    """

    def __init__(self):
        model_path = 'biosppy/nn/.models/ecg_afibdetection_bilstm.h5'
        super().__init__(model_path=model_path)

    def preprocess_signal(self, signal, sampling_rate=1000.0, win_len=20, step=1):
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

    def predict(self, signal, sampling_rate=1000.0, **kwargs):

        # predict using the model on every window
        X = self.preprocess_signal(signal, sampling_rate, **kwargs)
        probs = self.model.predict(X, verbose=0)

        # if any prediction is above the threshold, return afib
        if np.any(probs > self.threshold):
            return True
        else:
            return False

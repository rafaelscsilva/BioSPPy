Getting Started
================

One of the major goals of `biosppy` is to provide an easy starting point into
the world of biosignal processing. For that reason, we provide simple turnkey
solutions for each of the supported biosignal types. These functions implement
typical methods to filter, transform, and extract signal features. Let's see
how this works for the example of the ECG signal.

The GitHub repository includes a few example signals (see
`here <https://github.com/ScientISST/BioSPPy/tree/master/examples>`__). To load
and plot the raw ECG signal follow:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from biosppy import storage

    signal, mdata = storage.load_txt('.../examples/ecg.txt')
    Fs = mdata['sampling_rate']
    N = len(signal)  # number of samples
    T = (N - 1) / Fs  # duration
    ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
    plt.plot(ts, signal, lw=2)
    plt.grid()
    plt.show()

This should produce a similar output to the one shown below.

.. image:: images/ECG_raw.png
   :align: center
   :width: 100%
   :alt: Example of a raw ECG signal.

This signal is a Lead I ECG signal acquired at 1000 Hz, with a resolution of 12
bit. Although of good quality, it exhibits powerline noise interference, has a
DC offset resulting from the acquisition device, and we can also observe the
influence of breathing in the variability of R-peak amplitudes.

We can minimize the effects of these artifacts and extract a bunch of features
with the :py:class:`biosppy.signals.ecg.ecg` function:

.. code:: python

    >>> from biosppy.signals import ecg
    >>> out = ecg.ecg(signal=signal, sampling_rate=Fs, show=True)

It should produce a plot like the one below.

.. image:: images/ECG_summary.png
    :align: center
    :width: 100%
    :alt: Example of processed ECG signal.

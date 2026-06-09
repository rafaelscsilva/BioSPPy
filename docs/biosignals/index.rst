Biosignals
==========

`BioSPPy` includes processing pipelines for several biosignal types,
combining common preprocessing steps with feature extraction and visualization.
The pages below give a quick overview of each signal type and show a representative
example plot from the project dataset.

.. toctree::
   :maxdepth: 1

   acc
   ecg
   eda
   eeg
   egm
   emg
   pcg
   ppg
   resp
   rri

++++++++++++++++++++
What are Biosignals?
++++++++++++++++++++

Biosignals, in the most general sense, are measurements of physical properties
of biological systems. These include the measurement of properties at the
cellular level, such as concentrations of molecules, membrane potentials, and
DNA assays. On a higher level, for a group of specialized cells (i.e. an organ)
we are able to measure properties such as cell counts and histology, organ
secretions, and electrical activity (the electrical system of the heart, for
instance). Finally, for complex biological systems like the human being,
biosignals also include blood and urine test measurements, core body
temperature, motion tracking signals, and imaging techniques such as CAT and MRI
scans. However, the term biosignal is most often applied to bioelectrical,
time-varying signals, such as the electrocardiogram.

The task of obtaining biosignals of good quality is time-consuming,
and typically requires the use of costly hardware. Access to these instruments
is, therefore, usually restricted to research institutes, medical centers,
and hospitals. However, recent projects like `BITalino <http://bitalino.com/>`__
or `OpenBCI <http://openbci.com/>`__ have lowered the entry barriers of biosignal
acquisition, fostering the Do-It-Yourself and Maker communities to develop
physiological computing applications. You can find a list of biosignal
platform `here <https://opensource.com/life/15/4/five-diy-hardware-physiological-computing>`__.

The following sub-sections briefly describe the biosignals
covered by `biosppy`.


+++++++++++++++
Quick API links
+++++++++++++++

- ACC: :py:mod:`biosppy.signals.acc`
- ECG: :py:mod:`biosppy.signals.ecg`
- EDA: :py:mod:`biosppy.signals.eda`
- EEG: :py:mod:`biosppy.signals.eeg`
- EGM: :py:mod:`biosppy.signals.egm`
- EMG: :py:mod:`biosppy.signals.emg`
- PCG: :py:mod:`biosppy.signals.pcg`
- PPG: :py:mod:`biosppy.signals.ppg`
- RESP: :py:mod:`biosppy.signals.resp`
- RRI/HRV: :py:mod:`biosppy.signals.hrv`

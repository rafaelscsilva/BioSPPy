### 🎙️ Announcements
```
🌀 New module for signal quality assessment 🌀
With the biosppy.quality module you can now evaluate the quality of your signals!
So far, the EDA and ECG quality are available, but more could be added soon. 
```

# BioSPPy - Biosignal Processing in Python

[![PyPI version](https://img.shields.io/pypi/v/biosppy)](https://pypi.org/project/biosppy/)
[![PyPI downloads](https://img.shields.io/pypi/dm/biosppy)](https://pypi.org/project/biosppy/)
[![Documentation Status](https://readthedocs.org/projects/biosppy/badge/?version=latest)](https://biosppy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/biosppy)]()
[![GitHub issues](https://img.shields.io/github/issues/scientisst/BioSPPy)]()

[![GitHub stars](https://img.shields.io/github/stars/scientisst/BioSPPy)]()
[![GitHub forks](https://img.shields.io/github/forks/scientisst/BioSPPy)]()

*A toolbox for biosignal processing written in Python.*

<a href="https://biosppy.readthedocs.org/">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/logo/logo_400.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/logo_inverted_400.png">
  <img alt="Image" title="I know you're listening! - xkcd.com/525">
</picture>
</a>

The toolbox bundles together various signal processing and pattern recognition
methods geared towards the analysis of biosignals.

Highlights:

- Support for various biosignals: ECG, EDA, EEG, EMG, PCG, PPG, Respiration, HRV
- Signal analysis primitives: filtering, frequency analysis
- Feature extraction: time, frequency, and non-linear domain
- Signal quality assessment
- Signal synthesizers
- Clustering
- Biometrics

Documentation can be found at: <https://biosppy.readthedocs.org/>

## Installation

Installation can be easily done with `pip`:

```bash
$ pip install biosppy
```

Alternatively, you can install the latest version from the GitHub repository:

```bash
$ pip install git+https://github.com/scientisst/BioSPPy.git
```

## Simple Example

The code below loads an ECG signal from the `examples` folder, filters it,
performs R-peak detection, and computes the instantaneous heart rate.

```python
from biosppy import storage
from biosppy.signals import ecg

# load raw ECG signal
signal, mdata = storage.load_txt('./examples/ecg.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)
```

This should produce a plot similar to the one below.

![ECG summary example](docs/images/ECG_summary.png)

## Dependencies

- bidict
- h5py
- matplotlib
- numpy
- scikit-learn
- scipy
- shortuuid
- six
- joblib

## Citing
Please use the following if you need to cite BioSPPy:

- Carreiras C, Alves AP, Lourenço A, Canento F, Silva H, Fred A, *et al.*
  **BioSPPy - Biosignal Processing in Python**, 2015-,
  https://github.com/PIA-Group/BioSPPy/ [Online; accessed ```<year>-<month>-<day>```].

```latex
@Misc{,
  author = {Carlos Carreiras and Ana Priscila Alves and Andr\'{e} Louren\c{c}o and Filipe Canento and Hugo Silva and Ana Fred and others},
  title = {{BioSPPy}: Biosignal Processing in {Python}},
  year = {2015--},
  url = "https://github.com/PIA-Group/BioSPPy/",
  note = {[Online; accessed <today>]}
}
```

## License

BioSPPy is released under the BSD 3-clause license. See LICENSE for more details.

## Disclaimer

This program is distributed in the hope it will be useful and provided
to you "as is", but WITHOUT ANY WARRANTY, without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This
program is NOT intended for medical diagnosis. We expressly disclaim any
liability whatsoever for any direct, indirect, consequential, incidental
or special damages, including, without limitation, lost revenues, lost
profits, losses resulting from business interruption or loss of data,
regardless of the form of action or legal theory under which the
liability may be asserted, even if advised of the possibility of such
damages.

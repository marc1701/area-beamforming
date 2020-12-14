### AREA-Beamforming

## by Marc C. Green

A clustering approach to sound source tracking in Ambisonic audio. This modules contains code for:

* Spherical harmonic eamforming using Plane Wave Decomposition and Cross-pattern coherence beams.
* Rotation of non axis-symmetric spherical functions using Wigner-D matrices.
* Fibonacci, regular and geodesic spherical sampling schemes.
* Clustering (DBSCAN) and regression (SVR) for estimating coherent sound sources from power maps.
* `find_sources` wrapper function for automating estimation of source trajectories from an Ambisonic audio file.
* Functions to plot outputs.
* Implementations of _Frame Recall_ and _DOA Error_ performance metrics from DCASE 2019.

# Requirements
* [numpy](http://www.numpy.org/)
* [soundfile](https://pysoundfile.readthedocs.io/en/0.9.0/)
* [resampy](https://github.com/bmcfee/resampy)
* [matplotlib](https://matplotlib.org/)
* [scipy](https://www.scipy.org/)
* [seaborn](https://seaborn.pydata.org/)
* [progressbar](https://pypi.python.org/pypi/progressbar2)
* [dipy](https://dipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/)

# Usage

`find_sources(input, *args, **kwargs)`

`input` should be a path to an Ambisonic audio file.

`*kwargs` passed to `sph_peaks_t`:
- `max_n_peaks=20` - the maximum number of peaks that will be saved per frame.
- `audio_length_seconds=None` - optional variable replacing output frame numbers with time in seconds.

`*args` passed to `obj_trajectories`:
- `eps=0.1` - DBSCAN _Eps_ parameter.
- `min_samples=10` - DBSCAN _MinPts_ parameter.
- `relative_peak_threshold=0.5` - _dipy rel\_pk_ parameter.
- `min_separation_angle=5` - _dipy min\_sep_ parameter.

# What is it?

Script that compares the speed of several implementations of the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform):
* [numpy.fft](https://docs.scipy.org/doc/numpy/reference/routines.fft.html)
* [scipy.fftpack](https://docs.scipy.org/doc/scipy/reference/fftpack.html)
* [pyfftw.interfaces.numpy_fft](https://readthedocs.org/projects/pyfftw/downloads/pdf/latest/)
* [pyfftw.interfaces.scipy_fftpack](https://readthedocs.org/projects/pyfftw/downloads/pdf/latest/)
* [pyfftw.builders](https://readthedocs.org/projects/pyfftw/downloads/pdf/latest/)
* [pyfftw.FFTW](https://readthedocs.org/projects/pyfftw/downloads/pdf/latest/)

# Installation

* **Windows**:
```pwsh
virtualenv .venv
cd .venv/Scripts
activate
pip install -r <path_to_project>/requirements.txt
```

* **Linux**
```bash
virtualenv .venv -p python3
cd .venv/bin
source ./activate
pip install -r <path_to_project>/requirements.txt
```

# Comparison

For randomly generated one-dimensional and two-dimensional arrays, a direct FFT is performed, then the [power spectral density](https://en.wikipedia.org/wiki/Spectral_density) is calculated, and then the inverse FFT is calculated. The procedure is repeated several times, after which the average execution time of the indicated operations is calculated. For one-dimensional arrays, averaging occurs over 50,000 implementations, for two-dimensional arrays, over 500. The resulting times are normalized to the time of the FFT using the library *numpy*. The results are presented below:

| `1D array` | `2D array` |
| :----------: | :----------: |
| ![1d](https://github.com/VasilyevEvgeny/fft_comparator/blob/master/resources/comparison_1D.png) | ![2d](https://github.com/VasilyevEvgeny/fft_comparator/blob/master/resources/comparison_2D.png) |

It can be seen that in the case of a one-dimensional array, the *pyfftw* library modules, simulating the *numpy* and *scipy* interfaces, as well as the *pyfftw.builders* module, lose much in speed. The explanation is that for a one-dimensional array, the time for preliminary internal parameter adjustment, which always happens in *pyfftw*, is comparable to the time of the entire transfrom. In the case of a two-dimensional array, this time is short, therefore, the implementation of transformations in the indicated modules is faster than in those whose interface they repeat. Note that the FFT from *scipy* is always a little more efficient than from *numpy*, and the speed of transform in *pyfftw* with preliminary creation of plans is at least 2-3 times higher than all the others.

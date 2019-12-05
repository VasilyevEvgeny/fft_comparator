import numpy
import scipy
import pyfftw
from time import time
import logging
from tqdm import tqdm
import inspect
from numpy import array, zeros, random, float64, complex64
from collections import OrderedDict

from matplotlib import pyplot as plt
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


class FFTComparator:
    def __init__(self, **kwargs):
        self.__n_max = kwargs.get('n_max', 1024)
        self.__n_repeats = kwargs.get('n_repeats', (50000, 200))

        self.__verbose = kwargs.get('verbose', True)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)

        self.__statistics = self.__get_all_fourier_methods()

        arr_1d = array(random.randn(self.__n_max) +
                       1j * random.randn(self.__n_max), dtype=complex64)
        arr_2d = array(random.randn(self.__n_max, self.__n_max) +
                       1j * random.randn(self.__n_max, self.__n_max), dtype=complex64)

        self.__arrs = (arr_1d, arr_2d)

        self.__compare()

    def __standardize_keys(self, dictionary):
        max_name_length = 0
        for key in dictionary.keys():
            max_name_length = max(len(key[0]), max_name_length)

        old_keys = list(dictionary.keys())[:]
        for key in old_keys:
            new_name = key[0] + ' ' * (max_name_length - len(key[0]))
            dictionary[(new_name, key[1])] = dictionary.pop(key)

        return dictionary

    def __get_all_fourier_methods(self):
        all_methods = dict(inspect.getmembers(self, predicate=inspect.ismethod))
        fourier_methods = OrderedDict()
        for key in all_methods.keys():
            if 'fft_' in key:
                name = key[len(type(self).__name__) + 3:]
                fourier_methods.update({(name, all_methods[key]): {
                    'duration': 0.0,
                    'forward': [],
                    'spectral_density': [],
                    'backward': []
                }})

        return self.__standardize_keys(fourier_methods)

    def __clear_statistics(self):
        for key in self.__statistics.keys():
            self.__statistics[key]['duration'] = 0.0
            self.__statistics[key]['forward'] = []
            self.__statistics[key]['spectral_density'] = []
            self.__statistics[key]['backward'] = []



    def __log(self, data):
        logging.debug(data)

    def __fft_numpy(self, args):
        arr = args[0]

        dim = len(arr.shape)
        if dim == 1:
            forward = numpy.fft.fft(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = numpy.fft.ifft(forward)
        elif dim == 2:
            forward = numpy.fft.fft2(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = numpy.fft.ifft2(forward)
        else:
            raise Exception('Wrong input array dimension!')

        return forward, spectral_density, backward

    def __fft_scipy(self, args):
        arr = args[0]

        dim = len(arr.shape)
        if dim == 1:
            forward = scipy.fftpack.fft(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = scipy.fftpack.ifft(forward)
        elif dim == 2:
            forward = scipy.fftpack.fft2(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = scipy.fftpack.ifft2(forward)
        else:
            raise Exception('Wrong input array dimension!')

        return forward, spectral_density, backward

    def __fft_pyfftw_numpy(self, args):
        arr = args[0]

        dim = len(arr.shape)
        if dim == 1:
            forward = pyfftw.interfaces.numpy_fft.fft(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = pyfftw.interfaces.numpy_fft.ifft(forward)
        elif dim == 2:
            forward = pyfftw.interfaces.numpy_fft.fft2(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = pyfftw.interfaces.numpy_fft.ifft2(forward)
        else:
            raise Exception('Wrong input array dimension!')

        return forward, spectral_density, backward

    def __fft_pyfftw_scipy(self, args):
        arr = args[0]

        dim = len(arr.shape)
        if dim == 1:
            forward = pyfftw.interfaces.scipy_fftpack.fft(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = pyfftw.interfaces.scipy_fftpack.ifft(forward)
        elif dim == 2:
            forward = pyfftw.interfaces.scipy_fftpack.fft2(arr)
            spectral_density = forward.real**2 + forward.imag**2
            backward = pyfftw.interfaces.scipy_fftpack.ifft2(forward)
        else:
            raise Exception('Wrong input array dimension!')

        return forward, spectral_density, backward

    def __fft_pyfftw_builders(self, args):
        arr = args[0]

        dim = len(arr.shape)
        if dim == 1:
            forward = pyfftw.builders.fft(arr)()
            spectral_density = forward.real**2 + forward.imag**2
            backward = pyfftw.builders.ifft(forward)()
        elif dim == 2:
            forward = pyfftw.builders.fft2(arr)()
            spectral_density = forward.real**2 + forward.imag**2
            backward = pyfftw.builders.ifft2(forward)()
        else:
            raise Exception('Wrong input array dimension!')

        return forward, spectral_density, backward

    def __pyfftw(self, arr, function_name):
        dim = len(arr.shape)
        forward = zeros(arr.shape, dtype=complex64)
        spectral_density = zeros(arr.shape, dtype=float64)
        backward = zeros(arr.shape, dtype=complex64)

        if dim == 1:
            fft_plan = pyfftw.FFTW(arr, forward, direction='FFTW_FORWARD')
            ifft_plan = pyfftw.FFTW(arr, forward, direction='FFTW_BACKWARD')
        elif dim == 2:
            fft_plan = pyfftw.FFTW(arr, forward, direction='FFTW_FORWARD', axes=(0, 1))
            ifft_plan = pyfftw.FFTW(arr, forward, direction='FFTW_BACKWARD', axes=(0, 1))
        else:
            raise Exception('Wrong input array dimension!')

        t_start = time()
        for _ in tqdm(range(self.__n_repeats[dim-1]), desc='-->%s' % function_name):
            fft_plan()
            spectral_density = forward.real ** 2 + forward.imag ** 2
            ifft_plan()
        duration = time() - t_start

        return duration, forward, spectral_density, backward

    def __process_fft(self, pair, *args):
        fft_function_name, fft_function = pair

        forward, spectral_density, backward = None, None, None

        # n_repeats of forward and backward fft transforms
        t_start = time()
        for _ in tqdm(range(self.__n_repeats[args[0][1]]), desc='-->%s' % fft_function_name):
            forward, spectral_density, backward = fft_function(*args)
        duration = time() - t_start

        return duration, forward, spectral_density, backward

    def __compare(self):

        for arr in self.__arrs:
            dim = len(arr.shape)

            if self.__verbose:
                self.__log('%dD array.....' % dim)

            #
            # get statistics for ffts
            #

            for pair in self.__statistics.keys():

                duration, forward, spectral_density, backward = self.__process_fft(pair, [arr, dim-1])
                self.__statistics[pair]['duration'] = duration
                self.__statistics[pair]['forward'] = forward
                self.__statistics[pair]['spectral_density'] = spectral_density
                self.__statistics[pair]['backward'] = backward

            #
            # get statistics for pyfftw
            #

            name = 'fft_pyfftw'

            self.__statistics.update({(name, None): {
                'duration': 0.0,
                'forward': [],
                'spectral_density': [],
                'backward': []
            }})
            self.__standardize_keys(self.__statistics)
            for key in self.__statistics.keys():
                if key[0][:len(name)] == name and key[0][len(name)] == ' ':
                    target_name = key[0]

            duration, forward, spectral_density, backward = self.__pyfftw(arr, target_name)

            self.__statistics[(target_name, None)]['duration'] = duration
            self.__statistics[(target_name, None)]['forward'] = forward
            self.__statistics[(target_name, None)]['spectral_density'] = spectral_density
            self.__statistics[(target_name, None)]['backward'] = backward

            self.__normalize_durations()
            self.__plot_statistics(dim)

            self.__clear_statistics()
            self.__statistics.pop((target_name, None))

    def __normalize_durations(self, normalize_to='fft_numpy'):
        flag = False
        for key in self.__statistics:
            if normalize_to in key[0]:
                flag = True
                target_key = key

        if flag:
            normalize_to_val = self.__statistics[target_key]['duration']
        else:
            raise Exception('Wrong normalize_to value!')

        for key in self.__statistics:
            self.__statistics[key]['duration'] /= normalize_to_val

    def __plot_statistics(self, dim):

        font_size = 20
        font_weight = 'bold'

        plt.figure(figsize=(15, 10))

        durations, fft_names = [], []
        for key in self.__statistics.keys():
            durations.append(self.__statistics[key]['duration'])
            fft_names.append(key[0])

        y_pos = range(len(fft_names))

        plt.bar(y_pos, durations, color='blue', alpha=0.5)
        plt.xticks(y_pos, fft_names, rotation=45, fontsize=font_size, fontweight=font_weight)
        plt.yticks(fontsize=font_size, fontweight=font_weight)

        plt.ylabel('$\mathbf{\\bar{t}_{fft}}$ / $\mathbf{\\bar{t}_{fft\_numpy}}$',
                   fontsize=font_size + 5, fontweight=font_weight)

        plt.grid(color='gray', linewidth=2, alpha=0.5, ls='dotted')

        plt.savefig('comparison_%dD.png' % dim, bbox_inches='tight')
        plt.show()
        plt.close()


FFTComparator()

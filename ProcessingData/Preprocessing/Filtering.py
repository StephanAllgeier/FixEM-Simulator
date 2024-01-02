import scipy
from scipy.signal import butter, lfilter, freqz, filtfilt
import numpy as np
import matplotlib.pyplot as plt



class Filtering():
    @staticmethod
    def fft_transform(df, const_dict, coordinate, coordinate2):
        signal_x = df[const_dict[coordinate]].values
        signal_y = df[const_dict[coordinate2]].values
        signal = signal_x + 1j * signal_y
        fft_result = scipy.fft.fft(signal)
        fftfreq = scipy.fft.fftfreq(len(fft_result), 1 / const_dict['f'])

        return fft_result, fftfreq

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5, print_filter=False):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        if print_filter:
            w, h = freqz(b, a)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=f'Order={order}')
            plt.title(f'Band-pass filtered signal, lowcut={lowcut}, highcut={highcut}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Gain')
            plt.grid(True)
            plt.legend(loc='best')
        return b, a

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = Filtering.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def butter_bandpass_filter_zerophase(data, lowcut, highcut, fs, order=5):
        b, a = Filtering.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, padlen=3 * max(len(a), len(b)))
        return y

    @staticmethod
    def butter_lowpass(highcut, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = highcut / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    @staticmethod
    def butter_lowpass_filter(data, highcut, fs, order=5):
        b, a = Filtering.butter_lowpass(highcut, fs, order=order)
        y = filtfilt(b, a, data, padlen=3 * max(len(a), len(b)))
        return y
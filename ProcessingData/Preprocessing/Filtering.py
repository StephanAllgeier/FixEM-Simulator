import scipy
from scipy.signal import butter, lfilter
import numpy as np
import statsmodels.datasets


class Filtering():
    @staticmethod
    def fft_transform(df, const_dict, coordinate):
        fft = scipy.fft.fft(df[const_dict[coordinate]].values)
        fftfreq = scipy.fft.fftfreq(len(fft), 1 / const_dict['f'])
        return fft, fftfreq

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = Filtering.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def kalman_filter():
        return None

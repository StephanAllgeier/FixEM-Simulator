"""
Filtering Module
This module provides functions for signal processing, including Fourier Transform and Butterworth filtering.

Classes:
    Filtering: A class containing static methods for signal processing operations.

Methods:
    fft_transform(df, const_dict, coordinate, coordinate2):
    butter_bandpass(lowcut, highcut, fs, order=5, print_filter=False):
    butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    butter_bandpass_filter_zerophase(data, lowcut, highcut, fs, order=5):
    butter_lowpass(highcut, fs, order=5):
    butter_lowpass_filter(data, highcut, fs, order=5)

Author: Fabian Anzlinger
Date: 04.01.2024
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import butter, lfilter, freqz, filtfilt


class Filtering():
    @staticmethod
    def fft_transform(df, const_dict, coordinate, coordinate2):
        """
        Perform Fourier Transform on a 2D signal represented by two coordinates in a DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the signal data.
            const_dict (dict): A dictionary mapping coordinate names to column names in the DataFrame.
            coordinate (str): The name of the x-coordinate in the signal.
            coordinate2 (str): The name of the y-coordinate in the signal.

        Returns:
            tuple: A tuple containing the Fourier Transform result and corresponding frequencies.
                   - fft_result (numpy.ndarray): The complex Fourier Transform result.
                   - fftfreq (numpy.ndarray): The frequencies corresponding to the Fourier Transform.
        """
        signal_x = df[const_dict[coordinate]].values
        signal_y = df[const_dict[coordinate2]].values
        signal = signal_x + 1j * signal_y
        fft_result = scipy.fft.fft(signal)
        fftfreq = scipy.fft.fftfreq(len(fft_result), 1 / const_dict['f'])

        return fft_result, fftfreq

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5, print_filter=False):
        """
        Design a Butterworth bandpass filter and optionally visualize its frequency response.

        Parameters:
            lowcut (float): The low cutoff frequency of the bandpass filter.
            highcut (float): The high cutoff frequency of the bandpass filter.
            fs (float): The sampling frequency of the signal.
            order (int, optional): The order of the Butterworth filter. Defaults to 5.
            print_filter (bool, optional): If True, plot and display the frequency response. Defaults to False.

        Returns:
            tuple: A tuple containing filter coefficients (b, a).
                   - b (numpy.ndarray): Numerator coefficients of the filter.
                   - a (numpy.ndarray): Denominator coefficients of the filter.
        """
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
        """
        Apply a Butterworth bandpass filter to the input data.

        Parameters:
            data (numpy.ndarray): The input signal to be filtered.
            lowcut (float): The low cutoff frequency of the bandpass filter.
            highcut (float): The high cutoff frequency of the bandpass filter.
            fs (float): The sampling frequency of the signal.
            order (int, optional): The order of the Butterworth filter. Defaults to 5.

        Returns:
            numpy.ndarray: The filtered signal.
        """
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

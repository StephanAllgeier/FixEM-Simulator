import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Visualize():
    plt.rcParams['lines.linewidth'] = 1

    @staticmethod
    def plot_xy(dataset, const_dict):
        f = const_dict['f']
        t = dataset[const_dict['time_col']]*const_dict['TimeScaling']
        x = dataset[const_dict['x_col']]*const_dict['ValScaling']
        y = dataset[const_dict['y_col']]*const_dict['ValScaling']
        plt.plot(t, x, label='x')
        plt.plot(t, y, label='y')
        plt.xlabel('Time')
        plt.ylabel('Position in arcmin')
        plt.title('Position over Time')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_xy_trace(dataset, const_dict):
        x = dataset[const_dict['x_col']]*const_dict['ValScaling']
        y = dataset[const_dict['y_col']]*const_dict['ValScaling']
        plt.plot(x, y)
        plt.xlabel('X in arcminutes')
        plt.ylabel('Y in arcminutes')
        plt.title('Trace of Eye Movement')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_fft(fft, freq):
        plt.plot(freq, np.abs(fft))
        plt.xlabel('Frequency (Hz)')
        plt.xlim([0, max(freq)])
        plt.ylabel('Magnitude')
        plt.title('FFT - Magnitude Spectrum')
        plt.grid(True)
        plt.show()

    @staticmethod
    def print_microsacc(df, const_dict, micsacc):
        Visualize.plot_xy(df, const_dict)
        micsac_list = micsacc[0]
        for microsaccade in micsac_list:
            plt.axvline(microsaccade[0] / const_dict['f'], color='blue') # plotting onset of microsaccade
            plt.axvline(microsaccade[1] / const_dict['f'], color='red')  # plotting offset of microsaccade




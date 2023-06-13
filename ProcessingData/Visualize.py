import matplotlib.pyplot as plt
import numpy as np


class Visualize():
    plt.rcParams['lines.linewidth'] = 1

    @staticmethod
    def plot_xy(dataset, const_dict, color=None, labels=None):
        if labels is None:
            labels = ['x', 'y']
        if color is None:
            color = ['blue', 'orange']
        f = const_dict['f']
        t = dataset[const_dict['time_col']] * const_dict['TimeScaling']
        x = dataset[const_dict['x_col']] * const_dict['ValScaling']
        y = dataset[const_dict['y_col']] * const_dict['ValScaling']
        plt.plot(t, x, label=labels[0], color=color[0])
        plt.plot(t, y, label=labels[1], color=color[1])
        plt.xlabel('Time')
        plt.ylabel('Position in arcmin')
        plt.title('Position over Time')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_xy_µm(dataset, const_dict, color=None, labels=None):
        if labels is None:
            labels = ['x', 'y']
        if color is None:
            color = ['blue', 'orange']
        f = const_dict['f']
        t = dataset[const_dict['time_col']] * const_dict['TimeScaling']
        x = dataset[const_dict['x_µm']] * const_dict['ValScaling']
        y = dataset[const_dict['y_µm']] * const_dict['ValScaling']
        plt.plot(t, x, label=labels[0], color=color[0])
        plt.plot(t, y, label=labels[1], color=color[1])
        plt.xlabel('Time')
        plt.ylabel('Position in µm')
        plt.title('Position over Time')
        plt.legend()
        plt.show()
    @staticmethod
    def plot_xy_trace(dataset, const_dict, color='blue', label="Label"):
        x = dataset[const_dict['x_col']] * const_dict['ValScaling']
        y = dataset[const_dict['y_col']] * const_dict['ValScaling']
        plt.plot(x, y, color=color, label=label)
        plt.xlabel('X in arcminutes')
        plt.ylabel('Y in arcminutes')
        plt.title('Trace of Eye Movement')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_fft(fft, freq):
        plt.plot(freq, np.abs(fft))
        plt.xlabel('Frequency (Hz)')
        plt.xlim([0, 200])  # max(freq)])
        plt.ylabel('Magnitude')
        plt.title('FFT - Magnitude Spectrum')
        plt.grid(True)
        plt.show()

    @staticmethod
    def print_microsacc(df, const_dict, micsac, color=['red', 'blue']):
        Visualize.plot_xy(df, const_dict)
        if isinstance(micsac, tuple):
            micsac_list = micsac[0]  # Either tuple or list
        elif isinstance(micsac, list):
            micsac_list = micsac
        else:
            print('Microsaccade Input is neither a tupel nor a list')
        for microsaccade in micsac_list:
            plt.axvline(microsaccade[0] / const_dict['f'], color=color[0])  # plotting onset of microsaccade
            plt.axvline(microsaccade[1] / const_dict['f'], color=color[1])  # plotting offset of microsaccade


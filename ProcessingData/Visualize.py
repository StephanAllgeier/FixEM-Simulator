import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Visualize():
    plt.rcParams['lines.linewidth'] = 1

    @staticmethod
    def plot_xy(dataset, const_dict, color=None, labels=None, title='Eye Trace in x- and y-Position'):
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
        plt.xlabel('Time in s')
        plt.ylabel('Position in arcmin')
        plt.title(title)
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
        plt.xlabel('Time in s')
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
    def plot_microsacc(df, const_dict, title = 'Eye Trace in x- and y-Position', micsac=False, micsac2=False, color=['red', 'blue'], thickness=1, legend=['Onset','Offset']):
        if color is None:
            color = ['blue', 'orange']
        f = const_dict['f']
        t = df[const_dict['time_col']] * const_dict['TimeScaling']
        x = df[const_dict['x_col']] * const_dict['ValScaling']
        y = df[const_dict['y_col']] * const_dict['ValScaling']
        plt.plot(t, x,  color=color[0])
        plt.plot(t, y,  color=color[1])
        plt.xlabel('Time in s')
        plt.ylabel('Position in arcmin')
        plt.title(title)
        if isinstance(micsac, tuple):
            micsac_list = micsac[0]  # Either tuple or list
        elif isinstance(micsac, list):
            micsac_list = micsac
        if isinstance(micsac2, tuple):
            micsac_list2 = micsac2[0]  # Either tuple or list
        elif isinstance(micsac2, list):
            micsac_list2 = micsac2
        onsets = [micsac_list[i][0]/const_dict['f'] for i in range(len(micsac_list))]
        offsets= [micsac_list[i][1]/const_dict['f'] for i in range(len(micsac_list))]
        plt.vlines(x=onsets, ymin = min(x), ymax = max(x), colors=color[2], linewidth=thickness)
        plt.vlines(x=offsets, ymin = min(x), ymax = max(x),colors=color[3], linewidth=thickness)
        if micsac2:
            onsets2 = [micsac_list2[i][0] /const_dict['f']for i in range(len(micsac_list2))]
            offsets2 = [micsac_list2[i][1] /const_dict['f']for i in range(len(micsac_list2))]
            plt.vlines(x=onsets2,ymin = min(x), ymax = max(x),colors=color[4], linewidth=thickness)
            plt.vlines(x=offsets2, ymin = min(x), ymax = max(x),colors=color[5], linewidth=thickness)
        plt.gca().legend(legend)
        plt.show()

    @staticmethod
    def plot_prob_dist(data: list, title: str, x_value: str):
        # Takes a list of Data as input and plots the distribution
        mean = np.mean(data)
        median = np.median(data)
        std= np.std(data)
        # Wahrscheinlichkeitsverteilung erstellen
        x = np.linspace(min(data), max(data), 100)
        y = norm.pdf(x, mean, std)
        # Plot erstellen
        plt.plot(x, y, label='Wahrscheinlichkeitsverteilung')
        plt.hist(data, bins='auto', density=True, alpha=0.6, label='Histogram')
        plt.axvline(x=mean, color='r', linestyle='--', label=f'Mittelwert = {mean}', linewidth=2)
        plt.axvline(x=median, color='g', linestyle='--', label=f'Median = {median}', linewidth=2)
        plt.axvline(x=mean + std, color='b', linestyle='--', label=f'1 Sigma = {std}', linewidth=2)
        plt.axvline(x=mean - std, color='b', linestyle='--', linewidth=2)
        # Achsenbeschriftung und Legende hinzufügen
        plt.xlabel(x_value)
        plt.title(title)
        plt.ylabel('Wahrscheinlichkeitsdichte')
        plt.legend()
        plt.show()
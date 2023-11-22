import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class Visualize():
    plt.rcParams['lines.linewidth'] = 1

    @staticmethod
    def plot_xy(dataset, const_dict, color=None, labels=None, title='Augenbewegungen in x- und y-Koordinaten', ylabel= "Position", breite=12, höhe=6,savepath = None, filename=None, xlim=None, ylim=None):
        if labels is None:
            labels = ['x', 'y']
        if color is None:
            color = ['blue', 'orange']
        plt.figure(figsize=(breite, höhe))
        f = const_dict['f']
        if const_dict['time_col'] in dataset.columns:
            t = dataset[const_dict['time_col']] * const_dict['TimeScaling']
        else:
            t = dataset.index * const_dict['TimeScaling']
        x = dataset[const_dict['x_col']] * const_dict['ValScaling']
        y = dataset[const_dict['y_col']] * const_dict['ValScaling']
        plt.plot(t, x, label=labels[0], color=color[0])
        plt.plot(t, y, label=labels[1], color=color[1])
        plt.xlabel('Zeit in s', fontsize=14)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.legend()

        if savepath and filename:
            plt.savefig(f"{savepath}\{filename}.jpg", dpi=600)
        else:
            plt.show()
        plt.close()

    '''
    @staticmethod
    def plot_xy_dual(dataset1, dataset2, const_dict1, const_dict2, color1=None, color2=None, labels1=None, labels2=None,
                     title='Augenbewegungen in x- und y-Koordinaten'):
        if labels1 is None:
            labels1 = ['x', 'y']
        if labels2 is None:
            labels2 = ['x', 'y']
        if color1 is None:
            color1 = ['blue', 'orange']
        if color2 is None:
            color2 = ['blue', 'orange']

        f1 = const_dict1['f']
        t1 = dataset1[const_dict1['time_col']] * const_dict1['TimeScaling']
        x1 = dataset1[const_dict1['x_col']] * const_dict1['ValScaling']
        y1 = dataset1[const_dict1['y_col']] * const_dict1['ValScaling']

        f2 = const_dict2['f']
        t2 = dataset2[const_dict2['time_col']]
        x2 = dataset2[const_dict2['x_col']] * const_dict2['ValScaling']
        y2 = dataset2[const_dict2['y_col']] * const_dict2['ValScaling']

        mask1 = (t1 >= 12) & (t1 <= 15)
        mask2 = (t2 >= 12) & (t2 <= 15)
        t1 = np.array(t1[mask1])
        x1 = np.array(x1[mask1])
        y1 = np.array(y1[mask1])
        t2 = np.array(t2[mask2])
        x2 = np.array(x2[mask2])
        y2 = np.array(y2[mask2])
        plt.figure(figsize=(6, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t1, x1, label=labels1[0], color=color1[0])
        plt.plot(t1, y1, label=labels1[1], color=color1[1])
        plt.xlim(12, 15)
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
        plt.xlabel('Zeit in s')
        plt.ylabel('Position in Bogenminuten [arcmin]')
        plt.title(title + f' - Simulation')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t2, x2, label=labels2[0], color=color2[0])
        plt.plot(t2, y2, label=labels2[1], color=color2[1])
        plt.xlim(12, 15)
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
        plt.xlabel('Zeit in s')
        plt.ylabel('Position in Bogenminuten [arcmin]')
        plt.title(title + f' - GazeBase')
        plt.legend()

        
        plt.savefig(save_path + '.svg', format='svg', dpi=600)
        plt.savefig(save_path + '.jpeg', format='jpeg', dpi=600)
        plt.savefig(save_path + '.pdf', format='pdf')
        plt.show()
    '''

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

    def plot_xy_dual(dataset1, dataset2, const_dict1, const_dict2, subtitle1, subtitle2, savepath = None, color1=None, color2=None, labels1=None, labels2=None,
                     title='Augenbewegungen in x- und y-Koordinaten', t_on=0, t_off=5):
        if labels1 is None:
            labels1 = ['x', 'y']
        if labels2 is None:
            labels2 = ['x', 'y']
        if color1 is None:
            color1 = ['blue', 'orange']
        if color2 is None:
            color2 = ['blue', 'orange']

        f1 = const_dict1['f']
        t1 = dataset1[const_dict1['time_col']] * const_dict1['TimeScaling']
        x1 = dataset1[const_dict1['x_col']] * const_dict1['ValScaling']
        y1 = dataset1[const_dict1['y_col']] * const_dict1['ValScaling']

        f2 = const_dict2['f']
        t2 = dataset2[const_dict2['time_col']]
        x2 = dataset2[const_dict2['x_col']] * const_dict2['ValScaling']
        y2 = dataset2[const_dict2['y_col']] * const_dict2['ValScaling']

        mask1 = (t1 >= t_on) & (t1 <= t_off)
        mask2 = (t2 >= t_on) & (t2 <= t_off)
        t1 = np.array(t1[mask1])
        x1 = np.array(x1[mask1])
        y1 = np.array(y1[mask1])
        t2 = np.array(t2[mask2])
        x2 = np.array(x2[mask2])
        y2 = np.array(y2[mask2])
        plt.figure(figsize=(6,8))

        plt.subplot(2, 1, 1)
        plt.plot(t1, x1, label=labels1[0], color=color1[0])
        plt.plot(t1, y1, label=labels1[1], color=color1[1])
        plt.xlim(t_on, t_off)
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
        plt.xlabel('Zeit in s')
        plt.ylabel('Position in Grad [°]')
        plt.title(title +' - '+ subtitle1)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t2, x2, label=labels2[0], color=color2[0])
        plt.plot(t2, y2, label=labels2[1], color=color2[1])
        plt.xlim(t_on, t_off)
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
        plt.xlabel('Zeit in s')
        plt.ylabel('Position in Grad [°]')
        plt.title(title +' - ' +  subtitle2)
        plt.legend()
        plt.subplots_adjust(hspace=0.4)
        save_path = savepath
        plt.savefig(save_path + '.svg', format='svg', dpi=600)
        plt.savefig(save_path + '.jpeg', format='jpeg', dpi=600)
        plt.savefig(save_path + '.pdf', format='pdf')
        plt.show()

    @staticmethod
    def plot_fft(fft, freq, filename, title = 'FFT - Magnitude Spectrum'):
        plt.semilogy(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2])
        plt.xlabel('Frequenz in [Hz])')  # max(freq)])
        plt.ylabel('Magnitude')
        plt.title(title)
        plt.grid(True)
        plt.savefig(filename, dpi=600)
        plt.show()
        plt.close()


    @staticmethod
    def plot_microsacc(df, const_dict, title='Eye Trace in x- and y-Position', micsac=False, micsac2=False,
                       color=['red', 'blue'], thickness=1, legend=['Onset', 'Offset']):
        if color is None:
            color = ['blue', 'orange']
        f = const_dict['f']
        t = df[const_dict['time_col']] * const_dict['TimeScaling']
        x = df[const_dict['x_col']] * const_dict['ValScaling']
        y = df[const_dict['y_col']] * const_dict['ValScaling']
        plt.plot(t, x, color=color[0])
        plt.plot(t, y, color=color[1])
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
        onsets = [micsac_list[i][0] / const_dict['f'] for i in range(len(micsac_list))]
        offsets = [micsac_list[i][1] / const_dict['f'] for i in range(len(micsac_list))]
        plt.vlines(x=onsets, ymin=min(x), ymax=max(x), colors=color[2], linewidth=thickness)
        plt.vlines(x=offsets, ymin=min(x), ymax=max(x), colors=color[3], linewidth=thickness)
        if micsac2:
            onsets2 = [micsac_list2[i][0] / const_dict['f'] for i in range(len(micsac_list2))]
            offsets2 = [micsac_list2[i][1] / const_dict['f'] for i in range(len(micsac_list2))]
            plt.vlines(x=onsets2, ymin=min(x), ymax=max(x), colors=color[4], linewidth=thickness)
            plt.vlines(x=offsets2, ymin=min(x), ymax=max(x), colors=color[5], linewidth=thickness)
        plt.gca().legend(legend)
        plt.show()

    @staticmethod
    def plot_prob_dist(data: list, title: str, x_value: str):
        # Takes a list of Data as input and plots the distribution
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        # Wahrscheinlichkeitsverteilung erstellen
        x = np.linspace(min(data), max(data), 100)
        # y = norm.pdf(x, mean, std)
        # Plot erstellen
        # plt.plot(x, y, label='Wahrscheinlichkeitsverteilung')
        counts, bins , _ = plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram', edgecolor='black')
        # Bestimme den Index des Bins, dessen x-Wert größer als 1 ist
        plt.axvline(x=mean, color='r', linestyle='--', label=f'Mittelwert = {round(mean,3)}', linewidth=2)
        plt.axvline(x=median, color='g', linestyle='--', label=f'Median = {round(median,3)}', linewidth=2)
        plt.axvline(x=mean + std, color='b', linestyle='--', label=f'Stdev. = {round(std,3)}', linewidth=2)
        plt.axvline(x=mean - std, color='b', linestyle='--', linewidth=2)
        # Achsenbeschriftung und Legende hinzufügen
        plt.xlabel(x_value)
        plt.title(title)
        plt.ylabel('Wahrscheinlichkeitsdichte')
        plt.legend()
        plt.savefig(
            fr'C:\Users\uvuik\Desktop\{title}.jpeg',dpi=350)
        plt.show()

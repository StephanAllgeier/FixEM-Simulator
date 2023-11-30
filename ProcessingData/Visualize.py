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
    def plot_fft(fft, freq, filename, title = 'FFT - Magnitude Spectrum', xlim=(0,500), ylim = (10**(-2), 10**5)):
        plt.semilogy(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2])
        plt.xlabel('Frequenz in [Hz]', fontsize=16)
        plt.ylabel('Magnitude', fontsize=16)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(title, fontsize=18)
        plt.grid(True)
        plt.savefig(filename, dpi=600)
        plt.tight_layout()
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
        mean_str = str(round(mean,3)).replace('.',',')
        median = np.median(data)
        median_str = str(round(median,3)).replace('.',',')
        std = np.std(data)
        std_str = str(round(std,3)).replace('.',',')
        # Wahrscheinlichkeitsverteilung erstellen
        x = np.linspace(min(data), max(data), 100)
        # y = norm.pdf(x, mean, std)
        # Plot erstellen
        # plt.plot(x, y, label='Wahrscheinlichkeitsverteilung')
        counts, bins , _ = plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram', edgecolor='black')
        # Bestimme den Index des Bins, dessen x-Wert größer als 1 ist
        plt.axvline(x=mean, color='r', linestyle='--', label=f'Mittelwert = {mean_str}', linewidth=2)
        plt.axvline(x=median, color='g', linestyle='--', label=f'Median = {median_str}', linewidth=2)
        plt.axvline(x=mean + std, color='b', linestyle='--', label=f'Stdev. = {std_str}', linewidth=2)
        plt.axvline(x=mean - std, color='b', linestyle='--', linewidth=2)
        # Achsenbeschriftung und Legende hinzufügen
        plt.xlabel(x_value)
        plt.title(title)
        plt.ylabel('Wahrscheinlichkeitsdichte')
        plt.legend()
        plt.savefig(
            fr'C:\Users\uvuik\Desktop\{title}.jpeg',dpi=600)
        plt.show()

    @staticmethod
    def plot_three_traces(dataset1, dataset2, dataset3, f1, f2, f3, subtitle1, subtitle2, subtitle3, ylabel, savepath = None, color1=None, color2=None, labels1=None, labels2=None,
                 title='Augenbewegungen in x- und y-Koordinaten'):
        if labels1 is None:
            labels1 = ['x', 'y']
        if labels2 is None:
            labels2 = ['x', 'y']
        if color1 is None:
            color1 = ['blue', 'orange']
        if color2 is None:
            color2 = ['blue', 'orange']

        t1 = dataset1['TimeAxis']
        x1 = dataset1['x']
        y1 = dataset1['y']

        t2 = dataset2['Unnamed: 0'] / 1000
        x2 = dataset2['x']
        y2 = dataset2['y']

        t3 = dataset3['Time'][0:20000]
        x3 = dataset3['x'][10000:30000]
        y3 = dataset3['y'][10000:30000]

        t1 = np.array(t1)
        x1 = np.array(x1)
        y1 = np.array(y1)
        t2 = np.array(t2)
        x2 = np.array(x2)
        y2 = np.array(y2)
        t3 = np.array(t3)
        x3 = np.array(x3)
        y3 = np.array(y3)
        plt.figure(figsize=(26, 20))
        plt.suptitle(title, fontsize=18)
        plt.subplot(3, 1, 1)
        plt.plot(t1, x1, label=labels1[0], color=color1[0])
        plt.plot(t1, y1, label=labels1[1], color=color1[1])
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2), min(x3), min(y3)), max(max(x1), max(y1), max(x2), max(y2), max(x3), max(y3)))
        plt.xlim(0,14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(subtitle1, fontsize=16)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t2, x2, label=labels2[0], color=color2[0])
        plt.plot(t2, y2, label=labels2[1], color=color2[1])
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2), min(x3), min(y3)),
                 max(max(x1), max(y1), max(x2), max(y2), max(x3), max(y3)))
        plt.ylabel(ylabel, fontsize=14)
        plt.title(subtitle2, fontsize=16)
        plt.legend()
        plt.xlim(0, 14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.subplot(3, 1, 3)
        plt.plot(t3, x3, label=labels1[0], color=color1[0])
        plt.plot(t3, y3, label=labels1[1], color=color1[1])
        plt.ylim(min(min(x1), min(x2), min(y1), min(y2), min(x3), min(y3)),
                 max(max(x1), max(y1), max(x2), max(y2), max(x3), max(y3)))
        plt.xlabel('Zeit in s', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.title(subtitle3, fontsize=16)
        plt.legend()
        plt.xlim(0, 14)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(savepath + '.svg', format='svg', dpi=600)
        plt.savefig(savepath + '.jpeg', format='jpeg', dpi=600)
        plt.savefig(savepath + '.pdf', format='pdf')
        plt.show()

        @staticmethod
        def plot_two_traces(dataset1, dataset2, f1, f2, subtitle1, subtitle2, ylabel, savepath=None, color1=None,
                            color2=None, labels1=None, labels2=None,
                            title='Augenbewegungen in x- und y-Koordinaten'):
            if labels1 is None:
                labels1 = ['x', 'y']
            if labels2 is None:
                labels2 = ['x', 'y']
            if color1 is None:
                color1 = ['blue', 'orange']
            if color2 is None:
                color2 = ['blue', 'orange']

            t1 = dataset1['Time']
            x1 = dataset1['x']
            y1 = dataset1['y']

            t2 = dataset2['Time']
            x2 = dataset2['x']
            y2 = dataset2['y']

            t1 = np.array(t1)
            x1 = np.array(x1)
            y1 = np.array(y1)
            t2 = np.array(t2)
            x2 = np.array(x2)
            y2 = np.array(y2)
            plt.figure(figsize=(12, 20))
            plt.suptitle(title, fontsize=18)
            plt.subplot(2, 1, 1)
            plt.plot(t1, x1, label=labels1[0], color=color1[0])
            plt.plot(t1, y1, label=labels1[1], color=color1[1])
            plt.ylim(-60,
                     60)  # plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
            plt.xlabel('Zeit in s', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.ylabel(ylabel, fontsize=14)
            plt.title(subtitle1, fontsize=16)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(t2, x2, label=labels2[0], color=color2[0])
            plt.plot(t2, y2, label=labels2[1], color=color2[1])
            # plt.ylim(min(min(x1), min(x2), min(y1), min(y2)), max(max(x1), max(y1), max(x2), max(y2)))
            plt.ylim(-60, 60)
            plt.xlabel('Zeit in s', fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.title(subtitle2, fontsize=16)
            plt.legend()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.subplots_adjust(hspace=0.4)
            plt.savefig(savepath + '.svg', format='svg', dpi=600)
            plt.savefig(savepath + '.jpeg', format='jpeg', dpi=600)
            plt.savefig(savepath + '.pdf', format='pdf')
            plt.show()

    @staticmethod
    def plot_polar_hist(data, filepath):


        # Definiere die Anzahl der Bins für das Histogramm
        anzahl_bins = 36

        # Berechne das Histogramm
        hist, bin_edges = np.histogram(data, bins=np.linspace(-180, 180, anzahl_bins + 1), density=True)

        # Berechne die Mittelpunkte der Bins für die Darstellung auf dem Polarplot
        bin_mitte = np.deg2rad((bin_edges[:-1] + bin_edges[1:]) / 2)

        # Erstelle den Polarplot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Plotte Punkte und verbinde sie
        ax.plot(np.append(bin_mitte, bin_mitte[0]), np.append(hist, hist[0]), color='blue',
                linestyle='solid', markersize=8)
        plt.title('Verteilung der Richtung von Mikrosakkaden des Roorda-Datensatz')
        # Zeige den Plot
        plt.show()
        plt.savefig(filepath, dpi=600)

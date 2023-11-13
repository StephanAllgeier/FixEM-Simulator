import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
import math

from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox.microsac_detection import \
    vecvel
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades
from scipy.optimize import minimize

class Evaluation():

    @staticmethod
    def get_statistics(data):
        """
        Diese Funktion berechnet statistische Kennzahlen für eine gegebene Datenmenge.

        Parameter:
        data (list or numpy.ndarray): Eine Liste oder ein NumPy-Array von numerischen Werten.

        Rückgabewert:
        tuple: Ein Tupel, das den Mittelwert, den Median und die Standardabweichung der Daten enthält,
               in dieser Reihenfolge.

        - Der Mittelwert (Durchschnitt) gibt an, wie die Daten im Durchschnitt verteilt sind.
        - Der Median ist der Wert, der die Daten in zwei gleich große Hälften teilt.
        - Die Standardabweichung ist ein Maß für die Streuung der Daten.
        """
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        return mean, median, std

    @staticmethod
    def plot_histogram(data, bins):
        mean, _, std = Evaluation.get_statistics(data)
        plt.hist(data, bins=20, density=True, alpha=0.6, label='Daten')
        x = np.linspace(np.min(data), np.max(data), 100)
        norm_dist = stats.norm(loc=mean, scale=std)
        plt.plot(x, norm_dist.pdf(x), 'r', label='Normalverteilung')

    @staticmethod
    def get_csv_files_in_folder(folder_path):
        csv_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                csv_files.append(file_path)
        return csv_files

    @staticmethod
    def intermicrosacc_len(const_dict, micsac_list):
        '''
        Berechnet den intermikrosakkadischen Abstand für eine gegebene Liste mit Mikrosakkaden
        '''
        if isinstance(micsac_list, tuple):
            micsac_list = micsac_list[0]
        timedist = []
        micsac_onset = [micsac_list[i][0] for i in range(len(micsac_list))]
        micsac_offset = [micsac_list[i][1] for i in range(len(micsac_list))]
        if len(micsac_onset) != len(micsac_offset):
            print('Calucation wrong')
        for i in range(len(micsac_list) - 1):
            timedist.append((micsac_onset[i + 1] - micsac_offset[i]) / const_dict['f'])
            # timedist.append(df[const_dict['time_col']][micsac_list[i+1][0]] - df[const_dict['time_col']][micsac_list[i][1]])
        return timedist

    @staticmethod
    def intermicsac_distribution(const_dict, micsac_list, plot=False):
        intermicsac_list = Evaluation.intermicrosacc_len(const_dict, micsac_list)
        mean, median, stabw = Evaluation.get_statistics(intermicsac_list)
        if plot:
            Evaluation.plot_prob_dist(intermicsac_list)
        return mean, median, stabw

    @staticmethod
    def get_micsac_statistics(df, const_dict, micsac_list):
        # Berechnet ausgehend von einer Liste an Microsakkaden die intermicsac Abstände, Amplituden und die
        # Geschwindigkeiten der jeweiligen Mikrosakkaden und gibt eine Liste für die Amplituden und eine Liste für
        # die Geschwindigkeiten zurück
        intermic_dist = Evaluation.intermicrosacc_len(const_dict, micsac_list)
        if isinstance(micsac_list, tuple):
            micsac_list = micsac_list[0]
        amplitudes = []
        velocity = []
        duration = []
        x = df[[const_dict['x_col'], const_dict['y_col']]].to_numpy()
        vel = vecvel(x, sampling=const_dict['f'])
        for i in range(len(micsac_list)):
            start = micsac_list[i][0]
            end = micsac_list[i][1]
            x_onset_val = df[const_dict['x_col']][start]
            x_offset_val = df[const_dict['x_col']][end]
            y_onset_val = df[const_dict['y_col']][start]
            y_offset_val = df[const_dict['y_col']][end]
            x_diff = x_offset_val - x_onset_val
            y_diff = y_offset_val - y_onset_val
            amp = np.sqrt(np.square(x_diff) + np.square(y_diff))

            v = np.max(np.sqrt(vel[start:end, 0] ** 2 + vel[start:end, 1] ** 2))
            # v = amp / ((micsac_offset - micsac_onset) / const_dict['f'])
            # Gibt Geschwindigkeit in der Einheit der Eingabe zurück
            duration.append((end - start) / const_dict['f'])
            amplitudes.append(amp)
            velocity.append(v)
        # Hier ist ein Unterschied in der AMplitude detetiert worden: Entsteht durch Filtern bei Detektion der MS
        # Velocity in deg/s
        micsac_vel = [i / 60 for i in velocity]
        return intermic_dist, duration, amplitudes, velocity

    @staticmethod
    def get_drift_statistics(df, const_dict):  # df has to be without blinks
        drift = EventDetection.drift_only(df, const_dict)
        drift_indexes = EventDetection.drift_indexes_only(df, const_dict)
        amp = []
        amp2 = []
        velocity = []
        x = df[[const_dict['x_col'], const_dict['y_col']]].to_numpy()
        vel = vecvel(x, sampling=const_dict['f'])
        for drift_segment in drift_indexes:
            start = drift_segment[0]
            end = drift_segment[1]
            # a = Evaluation.berechne_weitester_abstand(df, const_dict, start, end)
            amplituden = np.sqrt(df[const_dict['x_col']][start:end] ** 2 + df[const_dict['y_col']][start:end] ** 2)
            amp.append(np.max(amplituden) - np.min(amplituden))
            # amp2.append(a)

            v = np.max(np.sqrt(vel[start:end, 0] ** 2 + vel[start:end, 1] ** 2))  # Caluclating max speed in segment
            velocity.append(v)
        return amp, velocity

    @staticmethod
    def get_tremor_statistics(df, const_dict):  # df has to be without blinks
        tremor = EventDetection.tremor_only(df, const_dict)
        tremor_segments = EventDetection.drift_indexes_only(df, const_dict)  # Remove Microsaccades from measurement
        amp = []
        velocity = []
        x = tremor[[const_dict['x_col'], const_dict['y_col']]].to_numpy()
        vel = vecvel(x, sampling=const_dict['f'])
        for tremor_segment in tremor_segments:
            start = tremor_segment[0]
            end = tremor_segment[1]
            amplituden = np.sqrt(
                tremor[const_dict['x_col']][start:end] ** 2 + tremor[const_dict['y_col']][start:end] ** 2)
            amp.append(np.max(amplituden) - np.min(amplituden))
            v = np.max(np.sqrt(vel[start:end, 0] ** 2 + vel[start:end, 1] ** 2))
            velocity.append(v)
        return amp, velocity

    def berechne_weitester_abstand(dataframe, const_dict, start, end):
        zeitbereich = dataframe[start:end - 1]
        x = zeitbereich[const_dict['x_col']].tolist()
        y = zeitbereich[const_dict['y_col']].tolist()

        weitester_abstand = 0.0

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                abstand = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                if abstand > weitester_abstand:
                    weitester_abstand = abstand

        return weitester_abstand

    @staticmethod
    def evaluation_of_files(files_list, const_dict):
        all_micsac_amp = []
        all_intermicsac = []
        all_micsac_dur = []
        all_micsac_vel = []
        all_drift_amp = []
        all_drift_vel = []
        all_tremor_amp = []
        all_tremor_vel = []
        for file in files_list:
            data = pd.read_csv(file)
            blink_rm, const_dict = Interpolation.remove_blink_annot(data, const_dict)
            micsac_detec = Microsaccades.find_micsac(blink_rm, const_dict)
            intermicsac, dur, amplitudes, velocity = Evaluation.get_micsac_statistics(blink_rm, const_dict,
                                                                                      micsac_detec)
            amp_drift, vel_drift = Evaluation.get_drift_statistics(blink_rm, const_dict)
            amp_tremor, vel_tremor = Evaluation.get_tremor_statistics(blink_rm, const_dict)
            all_micsac_amp.extend(amplitudes)
            all_intermicsac.extend(intermicsac)
            all_micsac_dur.extend(dur)
            all_micsac_vel.extend(velocity)
            all_drift_amp.extend(amp_drift)
            all_drift_vel.extend(vel_drift)
            all_tremor_amp.extend(amp_tremor)
            all_tremor_vel.extend(vel_tremor)

    @staticmethod
    def evaluate_json_hist(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
        micsac_amp = data["MicsacAmplitudes"] * 60
        intermic_dur = data["IntermicDur"]
        num_micsac = data["NumMicsac"]

        #Figure subplots
        fig, axs = plt.subplots(3,1,figsize=(8,12))

        axs[0].hist(micsac_amp, bins=50, alpha=0.5, edgecolor='black',
                 label="Amplituden der Mikrosakkaden in Bogenminuten [arcmin]", density=True)
        axs[1].hist(intermic_dur, bins=50, alpha=0.5, edgecolor='black', range=(0,2),
                 label="Intermikrosakkadische Intervalle in s",density=True)
        axs[2].hist(num_micsac, bins=50, alpha=0.5, edgecolor='black',
                 label="Anzahl an Mikrosakkaden pro Simulation", density=True)

        fig.suptitle(
            'Histogramme der Mikrosakkadenamplitude, intermikrosakkadischen Intervalle und Anzahl der Mikrosakkaden', fontsize=16)
        plt.legend(loc='upper right')

        for ax in axs:
            ax.set_ylabel('Häufigkeit')
        # Überschriften für jedes Histogramm
        axs[0].set_title("Amplituden der Mikrosakkaden in Bogenminuten [arcmin]")
        axs[0].set_xlabel('Amplituden in Bogenminuten [arcmin]')
        axs[0].set_xlim(0, max(micsac_amp))

        axs[1].set_title("Intermikrosakkadische Intervalle in s")
        axs[1].set_xlabel('Intermikrosakkadische Intervalldauer in s')
        axs[1].set_xlim(0, 2)#max(intermic_dur))

        axs[2].set_title("Anzahl an Mikrosakkaden pro Simulation")
        axs[2].set_xlabel('Anzahl Mikrosakkaden pro 30s Simulation')
        axs[2].set_xlim(min(num_micsac), max(num_micsac))
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f"{str(json_filepath)[:-5]}_histogram.jpeg", dpi=350)
        plt.close()
    @staticmethod
    def generate_histogram_with_logfit(json_filepath, range = (0,4), compare_to_roorda=False):
        """
        Erstellt ein Histogramm aus Daten in einer JSON-Datei und fügt eine Anpassungslinie auf einer logarithmischen Skala hinzu.

        Args:
            json_filepath (str): Der Dateipfad zur JSON-Datei mit den Daten.
            range_limits (tuple): Das Bereichslimit für das Histogramm (z.B., (0, 10)).

        Returns:
            None
        """
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
        intermic_dur = data["IntermicDur"]
        mean, median, stdev = np.mean(intermic_dur), np.median(intermic_dur), np.std(intermic_dur)
        #Figure subplots
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        hist_bins = 50
        range_limits = range
        hist_vals, hist_edges, _ = axs.hist(intermic_dur, bins=hist_bins, range=range_limits, edgecolor='black',
                                            label="Intermikrosakkadische Intervalle in s", density=True)

        # Anpassende Gerade im kleinen Diagramm plotten (logarithmische Skala)
        def line_function(params, x, y):
            slope, intercept = params
            y_fit = slope * x + intercept
            return np.sum((y - y_fit) ** 2)

        bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        initial_params = [0.1, 0.1]  # Beispiel-Startwerte für Steigung und y-Achsenabschnitt
        total_samples = len(intermic_dur)
        relative_freq = hist_vals / total_samples
        relative_freq_log = np.log10(hist_vals / total_samples)
        result = minimize(line_function, initial_params, args=(bin_centers, relative_freq_log))
        slope_fit, intercept_fit = result.x

        # Figure und Axes für das Haupt-Histogramm erstellen
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        # Punkte der Häufigkeit im Haupt-Histogramm plotten
        axs.hist(intermic_dur, bins=50, alpha=0.5, edgecolor='black', range=range_limits,
                     density=False)

        # Anpassende Gerade im kleinen Histogramm plotten
        x_fit = np.linspace(range_limits[0], range_limits[1], 100)
        y_fit = slope_fit * x_fit + intercept_fit# slope_fit * np.log10(x_fit) + intercept_fit

        # Kleines separates Diagramm erstellen (rechts oben)
        divider = axs.inset_axes([0.65, 0.65, 0.3, 0.3])
        divider.plot(bin_centers, relative_freq, 'o', label="Häufigkeit")
        divider.plot(x_fit, 10**y_fit, color='red', label="Anpassende Gerade (log)")  # Gerade auf logarithmischer Skala
        divider.set_xscale('linear')
        divider.set_yscale('log')
        divider.set_xlim(range_limits)
        divider.set_ylim(10**np.floor(np.log10(min(relative_freq))), 10**np.ceil(np.log10(max(relative_freq))))  # Anpassung der y-Achsenbegrenzung
        divider.set_xlabel('Intervalldauer in Sekunden [s]')
        divider.set_ylabel('relative Häufigkeit')
        divider.set_title("Semilogarithmische Darstellung")

        # Beschriftungen und Legende für das Haupt-Histogramm hinzufügen
        axs.set_xlabel("Intervalldauer in Sekunden [s]")
        axs.set_ylabel("Häufigkeit")
        axs.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Mean={round(mean,3)}, Median={round(median,3)}, Stdev={round(stdev,3)}")
        axs.legend()



        plt.savefig(f"{str(json_filepath)[:-5]}_{range}_histogram.jpeg", dpi=600)
        plt.close()

    @staticmethod
    def get_histogram_log_from_list(liste,savefigpath, const_dict):
        """
        Erstellt ein Histogramm aus einer gegebenen Liste von Werten und fügt eine Anpassungslinie auf einer logarithmischen Skala hinzu.

        Args:
            liste (list): Die Liste von Werten, für die das Histogramm erstellt werden soll.
            savefigpath (str): Der Dateipfad zur Speicherung des Histogramms als JPEG-Datei.
            const_dict (dict): Ein Dictionary mit Konstanten und Informationen für die Beschriftung des Histogramms.

        Returns:
            None
        """
        intermic_dur = liste
        mean, median, stdev = np.mean(intermic_dur), np.median(intermic_dur), np.std(intermic_dur)
        # Figure subplots
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))
        hist_bins = 50
        range_limits = (0, 4)
        hist_vals, hist_edges, _ = axs.hist(intermic_dur, bins=hist_bins, range=range_limits, edgecolor='black',
                                            label="Intermikrosakkadische Intervalle in s", density=True)
        hist_vals_small = [np.log10(i) for i in hist_vals]
        # Figure und Axes für das Haupt-Histogramm erstellen
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        # Anpassende Gerade im kleinen Diagramm plotten (logarithmische Skala)
        def line_function(params, x, y):
            slope, intercept = params
            y_fit = slope * x + intercept
            return np.sum((y - y_fit) ** 2)

        bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        initial_params = [0.1, 0.1]  # Beispiel-Startwerte für Steigung und y-Achsenabschnitt
        total_samples = len(intermic_dur)
        relative_freq = hist_vals / total_samples
        relative_freq_log = np.log10(hist_vals / total_samples)
        result = minimize(line_function, initial_params, args=(bin_centers, relative_freq_log))
        slope_fit, intercept_fit = result.x

        # Figure und Axes für das Haupt-Histogramm erstellen
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        # Punkte der Häufigkeit im Haupt-Histogramm plotten
        axs.hist(intermic_dur, bins=50, alpha=0.5, edgecolor='black', range=range_limits,
                 density=False)

        # Anpassende Gerade im kleinen Histogramm plotten
        x_fit = np.linspace(range_limits[0], range_limits[1], 100)
        y_fit = slope_fit * x_fit + intercept_fit  # slope_fit * np.log10(x_fit) + intercept_fit

        # Kleines separates Diagramm erstellen (rechts oben)
        divider = axs.inset_axes([0.65, 0.65, 0.3, 0.3])
        divider.plot(bin_centers, relative_freq, 'o', label="Häufigkeit")
        divider.plot(x_fit, 10 ** y_fit, color='red',
                     label="Anpassende Gerade (log)")  # Gerade auf logarithmischer Skala
        divider.set_xscale('linear')
        divider.set_yscale('log')
        divider.set_xlim(range_limits)
        divider.set_ylim(10 ** np.floor(np.log10(min(relative_freq))),
                         10 ** np.ceil(np.log10(max(relative_freq))))  # Anpassung der y-Achsenbegrenzung
        divider.set_xlabel('Intervalldauer in Sekunden [s]')
        divider.set_ylabel('relative Häufigkeit')
        divider.set_title("Semilogarithmische Darstellung")

        # Beschriftungen und Legende für das Haupt-Histogramm hinzufügen
        axs.set_xlabel("Intervalldauer in Sekunden [s]")
        axs.set_ylabel("Häufigkeit")
        axs.set_title(f"Histogramm der intermikrosakkadischen Intervalle - {const_dict['Name']}\n Mean={round(mean,3)}, Median={round(median,3)}, Stdev={round(stdev,3)}")
        axs.legend()

        plt.savefig(f"{savefigpath}\histogram_intermicsac.jpeg", dpi=600)
        plt.close()

    @staticmethod
    def normalized_histogram_difference(list1, list2, num_bins=10, normalize_01=False):
        """
        Berechnet die normalisierte Histogrammdifferenz zwischen zwei Listen.

        Args:
            list1 (list): Die erste Liste.
            list2 (list): Die zweite Liste.
            num_bins (int): Die Anzahl der Bins im Histogramm.

        Returns:
            float: Die normalisierte Histogrammdifferenz zwischen den beiden Listen.
        """
        # Erstellen Sie Histogramme für beide Listen und normalisieren Sie sie
        hist1, _ = np.histogram(list1, bins=num_bins, density=True)
        hist2, _ = np.histogram(list2, bins=num_bins, density=True)

        # Normalisieren Sie die Histogramme auf [0, 1]
        if normalize_01:
            hist1 /= np.max(hist1)
            hist2 /= np.max(hist2)

        # Berechnen Sie die Differenz zwischen den normalisierten Histogrammen
        diff = np.abs(hist1 - hist2)

        # Berechnen Sie die Gesamtdifferenz
        total_diff = np.sum(diff)

        return total_diff

    @staticmethod
    def dual_hist_w_histdiff(varlist1, varlist2, label1, label2, savefigpath, xlabel, range_limits = (0, 4), normalize_01=False):
        if not (isinstance(varlist1, list) and isinstance(varlist2, list)):
            return None
        intermic_dur1 = varlist1
        intermic_dur2 = varlist2

        # Berechnung der Statistiken für beide Datensätze
        mean1, median1, stdev1 = np.mean(intermic_dur1), np.median(intermic_dur1), np.std(intermic_dur1)
        mean2, median2, stdev2 = np.mean(intermic_dur2), np.median(intermic_dur2), np.std(intermic_dur2)

        # Figure und Axes für das Haupt-Histogramm erstellen
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        hist_bins = 50
        histdiff = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2, hist_bins, normalize_01)
        # Histogramm für den ersten Datensatz plotten
        axs.hist(intermic_dur1,label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
                  alpha=0.5, density=True)

        # Histogramm für den zweiten Datensatz plotten
        axs.hist(intermic_dur2, label=label2, bins=hist_bins, range=range_limits, edgecolor='black',
                  alpha=0.5, density=True)

        # Beschriftungen und Legende hinzufügen
        axs.set_xlabel(xlabel)
        axs.set_ylabel("Wahrscheinlichkeitsdichte")
        axs.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2}", fontweight='bold', fontsize=14)
        #axs.text(0.95, 0.85, f"HD = {histdiff}", transform=axs.transAxes, ha='right', va='top', fontsize=12)

        axs.legend()
        # Tabelle erstellen
        table_data = [["", "Mean", "Median", "Stdev"],
            [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
            [label2, round(mean2, 3), round(median2, 3), round(stdev2, 3)],
            [f"HD={round(histdiff,3)}", "", "", ""]
        ]
        table = axs.table(cellText=table_data, colLabels= None,loc='center right', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(0.6, 1.2)
        plt.tight_layout()
        plt.savefig(f"{savefigpath[:-5]}_histogram_intermicsac.jpeg", dpi=600)
        plt.close()

class Roorda():
    @staticmethod
    def get_intermic_hist_dataset_rooda(folderpath, speicherpfad, const_dict, return_data=False):
        """
           Generiert ein Histogramm und berechnet statistische Werte für die intermicrosakkadischen Intervalle basierend auf
           den gegebenen CSV-Dateien im angegebenen Ordner.

           Parameter:
           - folderpath (str): Der Pfad zum Ordner, der die CSV-Dateien enthält.
           - speicherpfad (str): Der Pfad, unter dem das Histogramm und die statistischen Werte gespeichert werden.
           - const_dict (dict): Ein Dictionary mit Konstanten, das Spaltennamen und andere Werte enthält.
           - return_data (bool): Ein optionaler Parameter, der angibt, ob die berechneten Daten zurückgegeben werden sollen.

           Rückgabewert (falls return_data=True):
           - intermic_dur (list): Eine Liste der intermicrosakkadischen Intervalle in Sekunden.
           - amp (list): Eine Liste der Amplituden der Intervalle.
           - micsac_vel (list): Eine Liste der Geschwindigkeiten der Intervalle.

           Hinweise:
           - Die Funktion liest CSV-Dateien aus dem angegebenen Ordner und führt verschiedene Berechnungen an den Daten aus.
           - Sie entfernt Blink-Annotationen aus den Daten und berechnet intermicrosakkadische Intervalle, Amplituden und Geschwindigkeiten.
           - Ein Histogramm der Intervalle wird erstellt und unter 'speicherpfad' als JPEG-Datei gespeichert.
           - Zusätzlich werden logarithmische Histogrammdaten gespeichert.

           Beispielaufruf:
           get_intermic_hist_dataset_rooda("pfad_zum_ordner", "speicherort", const_dict, return_data=True)

           """
        all_files = Evaluation.get_csv_files_in_folder(folderpath)
        intermic_dur = []
        amp = []
        micsac_vel = []
        for filepath in all_files:
            dataframe = pd.read_csv(filepath)
            dataframe, const_dict = Interpolation.remove_blink_annot(dataframe, const_dict)
            intermic_dist = Roorda.get_intermicsac_roorda(dataframe)
            durations = [(i[1] - i[0]) / 1920 for i in intermic_dist]
            x_amplitude = [(dataframe[const_dict['x_col']][i[1]] - dataframe[const_dict['x_col']][i[0]]) for i in
                           intermic_dist]
            y_amplitude = [(dataframe[const_dict['y_col']][i[1]] - dataframe[const_dict['y_col']][i[0]]) for i in
                           intermic_dist]
            amplitude = [np.sqrt(x_amplitude[i] ** 2 + y_amplitude[i] ** 2) for i in range(len(x_amplitude))]
            vel = [amplitude[i] / durations[i] for i in range(len(amplitude))]
            intermic_dur.extend(durations)
            amp.extend(amplitude)
            micsac_vel.extend(vel)
        if return_data:
            return intermic_dur, amp, micsac_vel
        plt.hist(intermic_dur, bins=200, edgecolor='black')

        # Titel und Beschriftungen hinzufügen
        plt.title('Histogramm der intermicrosakkadischen Intervalle - Roorda Lab')
        plt.xlabel('Dauer in Sekunden [s]')
        plt.ylabel('Häufigkeit')

        # Diagramm speichern
        plt.savefig(f'{speicherpfad}\histogram_Roorda_intermicsac.jpeg', dpi=600)
        plt.close()
        Evaluation.get_histogram_log_from_list(intermic_dur, speicherpfad, const_dict=const_dict)

    @staticmethod
    def get_intermicsac_roorda(dataframe):
        # Diese Funktion nimmt als Input einen Dataframe und gibt eine LIste aus Listen zurück, mit ONset und Offset der INtermikrosakkadischen INtervalle gegeben als Indizes
        micsacs = Microsaccades.get_roorda_micsac(dataframe)
        intermic_dist = []
        for i in range(0, len(micsacs)):
            if i == 0:
                intermic_dist.append([0, micsacs[i][0]])
            else:
                intermic_dist.append([micsacs[i - 1][1], micsacs[i][0]])
        return intermic_dist

def dual_hist_subplot_w_histdiff(varlist1, varlist2_1, varlist2_2, label1, label2_1, label2_2, savefigpath, xlabel, range_limits=(0, 2.5), normalize_01=False):
    if not (isinstance(varlist1, list) and isinstance(varlist2_1, list) and isinstance(varlist2_2, list)):
        return None

    intermic_dur1 = varlist1
    intermic_dur2_1 = varlist2_1
    intermic_dur2_2 = varlist2_2

    # Berechnung der Statistiken für beide Datensätze
    mean1, median1, stdev1 = np.mean(intermic_dur1), np.median(intermic_dur1), np.std(intermic_dur1)
    mean2_1, median2_1, stdev2_1 = np.mean(intermic_dur2_1), np.median(intermic_dur2_1), np.std(intermic_dur2_1)
    mean2_2, median2_2, stdev2_2 = np.mean(intermic_dur2_2), np.median(intermic_dur2_2), np.std(intermic_dur2_2)

    # Figure und Axes für das Haupt-Histogramm erstellen
    fig, axs  = plt.subplots(2, 1, figsize=(15, 12))


    # Subplot 1
    ax1 = axs[0]
    hist_bins = 50
    histdiff_1 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2_1, hist_bins, normalize_01)
    ax1.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    ax1.hist(intermic_dur2_1, label=label2_1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Wahrscheinlichkeitsdichte")
    ax1.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2_1}", fontweight='bold', fontsize=14)
    ax1.legend()

    # Subplot 2
    ax2 = axs[1]
    histdiff_2 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2_2, hist_bins, normalize_01)
    ax2.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    ax2.hist(intermic_dur2_2, label=label2_2, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Wahrscheinlichkeitsdichte")
    ax2.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2_2}", fontweight='bold', fontsize=14)
    ax2.legend()

    # Tabelle erstellen
    table_data = [["", "Mean", "Median", "Stdev"],
                  [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
                  [label2_1, round(mean2_1, 3), round(median2_1, 3), round(stdev2_1, 3)],
                  [f"HD={round(histdiff_1, 3)}", "", "", ""]
                  ]
    table1 = ax1.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(0.6, 1.2)

    table_data = [["", "Mean", "Median", "Stdev"],
                  [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
                  [label2_2, round(mean2_2, 3), round(median2_2, 3), round(stdev2_2, 3)],
                  [f"HD={round(histdiff_2, 3)}", "", "", ""]
                  ]
    table2 = ax2.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(0.6, 1.2)

    plt.tight_layout()
    plt.savefig(f"{savefigpath[:-5]}_histogram_intermicsac_subplot.jpeg", dpi=600)
    plt.close()

def dual_hist_subplot_w_histdiff_log(varlist1, varlist2_1, varlist2_2, label1, label2_1, label2_2, savefigpath, xlabel, range_limits=(0, 2.5), normalize_01=False):
    if not (isinstance(varlist1, list) and isinstance(varlist2_1, list) and isinstance(varlist2_2, list)):
        return None

    intermic_dur1 = varlist1
    intermic_dur2_1 = varlist2_1
    intermic_dur2_2 = varlist2_2

    # Berechnung der Statistiken für beide Datensätze
    mean1, median1, stdev1 = np.mean(intermic_dur1), np.median(intermic_dur1), np.std(intermic_dur1)
    mean2_1, median2_1, stdev2_1 = np.mean(intermic_dur2_1), np.median(intermic_dur2_1), np.std(intermic_dur2_1)
    mean2_2, median2_2, stdev2_2 = np.mean(intermic_dur2_2), np.median(intermic_dur2_2), np.std(intermic_dur2_2)

    # Figure und Axes für das Haupt-Histogramm erstellen
    fig, axs  = plt.subplots(2, 2, figsize=(15, 12), gridspec_kw={'width_ratios': [2, 1]})

    # Subplot 1
    ax1 = axs[0, 0]
    hist_bins = 50
    histdiff_1 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2_1, hist_bins, normalize_01)
    hist_vals_roorda, hist_edges_roorda , _ = ax1.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    hist_vals1, hist_edges1, _ =ax1.hist(intermic_dur2_1, label=label2_1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    ax1.set_ylabel("Wahrscheinlichkeitsdichte", fontsize=12)
    ax1.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2_1}", fontweight='bold', fontsize=14)
    ax1.set_ylim(0, max(hist_vals1)*1.1)
    ax1.legend()

    def line_function(params, x, y):
        slope, intercept = params
        y_fit = slope * x + intercept
        return np.sum((y - y_fit) ** 2)

    #Scatter für Roorda
    x_fit = np.linspace(range_limits[0], range_limits[1], 100)
    bin_centers_roorda = (hist_edges_roorda[:-1] + hist_edges_roorda[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_roorda = len(intermic_dur1)
    relative_freq_roorda= hist_vals_roorda / total_samples_roorda
    relative_freq_log_roorda = np.log10(hist_vals_roorda / total_samples_roorda)
    result_roorda = minimize(line_function, initial_params, args=(bin_centers_roorda, relative_freq_log_roorda))
    slope_fit_roorda, intercept_fit_roorda = result_roorda.x
    y_fit_roorda = slope_fit_roorda * x_fit + intercept_fit_roorda

    # Scatterplot für Subplot 1
    bin_centers_1 = (hist_edges1[:-1] + hist_edges1[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_1 = len(intermic_dur2_1)
    relative_freq_1 = hist_vals1 / total_samples_1
    hist_vals1[46] = 1
    relative_freq_log_1 = np.log10(hist_vals1 / (total_samples_1+1))
    result_1= minimize(line_function, initial_params, args=(bin_centers_1, relative_freq_log_1))
    slope_fit_1, intercept_fit_1 = result_1.x
    y_fit_1= slope_fit_1 * x_fit + intercept_fit_1


    ax1_scatter = axs[0, 1]
    ax1_scatter.plot(bin_centers_roorda, relative_freq_roorda, 'o', color='red')
    ax1_scatter.plot(bin_centers_1, relative_freq_1, 'o', color="blue")
    ax1_scatter.plot(x_fit, 10 ** y_fit_roorda, color='red',
                 label="Anpassende Gerade (log) Roorda")  # Gerade auf logarithmischer Skala
    ax1_scatter.plot(x_fit, 10 ** y_fit_1, color='blue',
                     label="Anpassende Gerade (log) math. Modell")
    ax1_scatter.set_xscale('linear')
    ax1_scatter.set_yscale('log')
    ax1_scatter.set_xlim(range_limits)
    ax1_scatter.set_ylim(10 ** np.floor(np.log10(min(relative_freq_roorda))),
                     10 ** np.ceil(np.log10(max(relative_freq_roorda))))  # Anpassung der y-Achsenbegrenzung
    ax1_scatter.set_xlabel(xlabel,fontsize=12)
    ax1_scatter.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=12)
    ax1_scatter.set_title("Logarithmischer Histogramm-Vergleich",  fontweight='bold', fontsize=14)
    ax1_scatter.legend(loc='upper right')
    ax1_scatter.grid(True)

    # Subplot 2
    ax2 = axs[1, 0]
    histdiff_2 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2_2, hist_bins, normalize_01)
    ax2.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    hist_vals2, hist_edges2, _ = ax2.hist(intermic_dur2_2, label=label2_2, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Wahrscheinlichkeitsdichte",fontsize=12)
    ax2.set_title(f"Vergleich {label1} / {label2_2}", fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.set_ylim(0, max(hist_vals1)*1.1)

    # Scatterplot für Subplot 2
    bin_centers_2 = (hist_edges2[:-1] + hist_edges2[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_2 = len(intermic_dur2_2)
    relative_freq_2 = hist_vals2 / total_samples_2
    relative_freq_log_2 = np.log10(hist_vals2 / total_samples_2)
    result_2 = minimize(line_function, initial_params, args=(bin_centers_2, relative_freq_log_2))
    slope_fit_2, intercept_fit_2 = result_2.x
    y_fit_2 = slope_fit_2 * x_fit + intercept_fit_2


    # Scatterplot für Subplot 2
    ax2_scatter = axs[1, 1]
    ax2_scatter.plot(bin_centers_roorda, relative_freq_roorda, 'o', color="blue")
    ax2_scatter.plot(bin_centers_2, relative_freq_2, 'o',color='red')
    ax2_scatter.plot(x_fit, 10 ** y_fit_roorda, color="blue",
                     label="Anpassende Gerade (log) Roorda")  # Gerade auf logarithmischer Skala
    ax2_scatter.plot(x_fit, 10 ** y_fit_2, color='red',
                     label="Anpassende Gerade (log) math. Modell")
    ax2_scatter.set_xscale('linear')
    ax2_scatter.set_yscale('log')
    ax2_scatter.set_xlim(range_limits)
    ax2_scatter.set_ylim(10 ** np.floor(np.log10(min(relative_freq_roorda))),
                          10 ** np.ceil(np.log10(max(relative_freq_roorda))))  # Anpassung der y-Achsenbegrenzung
    ax2_scatter.set_xlabel(xlabel,fontsize=12)
    ax2_scatter.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=12)
    ax2_scatter.set_title(f"Logarithmischer Histogramm-Vergleich", fontweight='bold', fontsize=14)
    ax2_scatter.legend(loc='upper right')
    ax2_scatter.grid(True)
    # Tabelle erstellen
    table_data = [[f"HD={round(histdiff_1, 3)}", "Mittelwert", "Median", "Standardabw."],
                  [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
                  ['Math. Modell', round(mean2_1, 3), round(median2_1, 3), round(stdev2_1, 3)]
                  ]
    table1 = ax1.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(0.6, 1.2)
    # Fette Schrift für HD-Zelle
    table1[(0, 0)].set_text_props(fontweight='bold')

    table_data = [[f"HD={round(histdiff_2, 3)}",  "Mittelwert", "Median", "Standardabw."],
                  [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
                  ['Math. Modell', round(mean2_2, 3), round(median2_2, 3), round(stdev2_2, 3)]
                  ]
    table2 = ax2.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table2[(0, 0)].set_text_props(fontweight='bold')
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(0.6, 1.2)

    plt.tight_layout()
    plt.savefig(f"{savefigpath[:-5]}_histogram_intermicsac_subplot.jpeg", dpi=600)
    plt.close()

def hist_subplot_w_histdiff_log(varlist1, varlist2,  label1, label2, savefigpath, xlabel, range_limits=(0, 2.5), normalize_01=False, title=None):
    if not isinstance(varlist1, list) and isinstance(varlist2, list):
        return None

    intermic_dur1 = varlist1
    intermic_dur2 = varlist2

    # Berechnung der Statistiken für beide Datensätze
    mean1, median1, stdev1 = np.mean(intermic_dur1), np.median(intermic_dur1), np.std(intermic_dur1)
    mean2_1, median2_1, stdev2_1 = np.mean(intermic_dur2), np.median(intermic_dur2), np.std(intermic_dur2)

    # Figure und Axes für das Haupt-Histogramm erstellen
    fig, axs  = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [2, 1]})

    # Subplot 1
    ax1 = axs[0]
    hist_bins = 50
    histdiff_1 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2, hist_bins, normalize_01)
    hist_vals_roorda, hist_edges_roorda , _ = ax1.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True)
    hist_vals1, hist_edges1, _ =ax1.hist(intermic_dur2, label=label2, bins=hist_bins, range=range_limits, edgecolor='black',
                                         alpha=0.5, density=True)
    ax1.set_ylabel("Wahrscheinlichkeitsdichte", fontsize=12)
    if title== None:
        title = f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2}"
    ax1.set_title(title, fontweight='bold', fontsize=14)
    ax1.set_ylim(0, max(hist_vals1)*1.1)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.legend()

    def line_function(params, x, y):
        slope, intercept = params
        y_fit = slope * x + intercept
        return np.sum((y - y_fit) ** 2)

    #Scatter für Roorda
    x_fit = np.linspace(range_limits[0], range_limits[1], 100)
    bin_centers_roorda = (hist_edges_roorda[:-1] + hist_edges_roorda[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_roorda = len(intermic_dur1)
    relative_freq_roorda= hist_vals_roorda / total_samples_roorda
    relative_freq_log_roorda = np.log10(hist_vals_roorda / total_samples_roorda)
    result_roorda = minimize(line_function, initial_params, args=(bin_centers_roorda, relative_freq_log_roorda))
    slope_fit_roorda, intercept_fit_roorda = result_roorda.x
    y_fit_roorda = slope_fit_roorda * x_fit + intercept_fit_roorda

    # Scatterplot für Subplot 1
    bin_centers_1 = (hist_edges1[:-1] + hist_edges1[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_1 = len(intermic_dur2)
    relative_freq_1 = hist_vals1 / total_samples_1
    hist_vals1[46] = 1
    relative_freq_log_1 = np.log10(hist_vals1 / (total_samples_1+1))
    result_1= minimize(line_function, initial_params, args=(bin_centers_1, relative_freq_log_1))
    slope_fit_1, intercept_fit_1 = result_1.x
    y_fit_1= slope_fit_1 * x_fit + intercept_fit_1


    ax1_scatter = axs[1]
    ax1_scatter.plot(bin_centers_roorda, relative_freq_roorda, 'o', color='red')
    ax1_scatter.plot(bin_centers_1, relative_freq_1, 'o', color="blue")
    ax1_scatter.plot(x_fit, 10 ** y_fit_roorda, color='red',
                 label="Anpassende Gerade (log) Roorda")  # Gerade auf logarithmischer Skala
    ax1_scatter.plot(x_fit, 10 ** y_fit_1, color='blue',
                     label="Anpassende Gerade (log) math. Modell")
    ax1_scatter.set_xscale('linear')
    ax1_scatter.set_yscale('log')
    ax1_scatter.set_xlim(range_limits)
    ax1_scatter.set_ylim(10 ** np.floor(np.log10(min(relative_freq_roorda))),
                     10 ** np.ceil(np.log10(max(relative_freq_roorda))))  # Anpassung der y-Achsenbegrenzung
    ax1_scatter.set_xlabel(xlabel,fontsize=12)
    ax1_scatter.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=12)
    ax1_scatter.set_title("Logarithmischer Histogramm-Vergleich",  fontweight='bold', fontsize=14)
    ax1_scatter.legend(loc='upper right')
    ax1_scatter.grid(True)


    # Tabelle erstellen
    table_data = [[f"HD={round(histdiff_1, 3)}", "Mittelwert", "Median", "Standardabw."],
                  [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
                  [label2, round(mean2_1, 3), round(median2_1, 3), round(stdev2_1, 3)]
                  ]
    table1 = ax1.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(0.6, 1.2)
    # Fette Schrift für HD-Zelle
    table1[(0, 0)].set_text_props(fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{savefigpath[:-5]}_singlehistogram_intermicsac_subplot.jpeg", dpi=600)
    plt.close()

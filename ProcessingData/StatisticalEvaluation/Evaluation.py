'''
This file implements different functions for statistical Evaluation and visualization of those properties.
It contains Evaluation of statistics as mean, median and stdev as well as Histogramdifference. Different plotting
funcitons are implemented.
Within this file there is also a specified class Roorda for evaluation of the Roorda Dataset.

Author: Fabian Anzlinger
Date 04.01.2024
'''

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox.microsac_detection import \
    vecvel
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades

def dual_hist_subplot_w_histdiff_log(varlist1, varlist2_1, varlist2_2, label1, label2_1, label2_2, savefigpath, xlabel,
                                     range_limits=(0, 2.5), normalize_01=False):
    """
    Generates a dual histogram subplot with additional scatter plots and logarithmic histogram comparison, including statistical values and fitting lines.

    Parameters:
    - varlist1 (list): List of values for the first dataset.
    - varlist2_1 (list): List of values for the second dataset (first comparison).
    - varlist2_2 (list): List of values for the second dataset (second comparison).
    - label1 (str): Label for the first dataset.
    - label2_1 (str): Label for the second dataset (first comparison).
    - label2_2 (str): Label for the second dataset (second comparison).
    - savefigpath (str): Path to save the generated plots.
    - xlabel (str): Label for the x-axis.
    - range_limits (tuple): Limits for the histogram x-axis (default is (0, 2.5)).
    - normalize_01 (bool): Whether to normalize the histograms to the range [0, 1].

    Returns:
    - None

    Notes:
    - The function generates a dual histogram subplot with additional scatter plots and logarithmic histogram comparisons.
    - Fitting lines are added to the scatter plots based on logarithmic transformations.
    - Statistical values, such as mean, median, and standard deviation, are displayed in tables for both comparisons.
    - The resulting plots are saved as a JPEG file under the specified 'savefigpath'.
    """
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
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [2, 1]})

    # Subplot 1
    ax1 = axs[0, 0]
    hist_bins = 50
    histdiff_1 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2_1, hist_bins, normalize_01)
    hist_vals_roorda, hist_edges_roorda, _ = ax1.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits,
                                                      edgecolor='black',
                                                      alpha=0.5, density=True, color='blue')
    hist_vals1, hist_edges1, _ = ax1.hist(intermic_dur2_1, label=label2_1, bins=hist_bins, range=range_limits,
                                          edgecolor='black',
                                          alpha=0.5, density=True, color='red')
    ax1.set_ylabel("Wahrscheinlichkeitsdichte", fontsize=14)
    ax1.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2_1}",
                  fontweight='bold', fontsize=15)
    ax1.set_ylim(0, max(hist_vals1) * 1.1)
    ax1.legend()

    def line_function(params, x, y):
        slope, intercept = params
        y_fit = slope * x + intercept
        return np.sum((y - y_fit) ** 2)

    # Scatter für Roorda
    x_fit = np.linspace(range_limits[0], range_limits[1], 100)
    bin_centers_roorda = (hist_edges_roorda[:-1] + hist_edges_roorda[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_roorda = len(intermic_dur1)
    relative_freq_roorda = hist_vals_roorda / total_samples_roorda
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
    relative_freq_log_1 = np.log10(hist_vals1 / (total_samples_1 + 1))
    result_1 = minimize(line_function, initial_params, args=(bin_centers_1, relative_freq_log_1))
    slope_fit_1, intercept_fit_1 = result_1.x
    y_fit_1 = slope_fit_1 * x_fit + intercept_fit_1

    ax1_scatter = axs[0, 1]
    ax1_scatter.plot(bin_centers_roorda, relative_freq_roorda, 'o', color='blue')
    ax1_scatter.plot(bin_centers_1, relative_freq_1, 'o', color="red")
    ax1_scatter.plot(x_fit, 10 ** y_fit_roorda, color='red',
                     label="Anpassende Gerade (log) Roorda")  # Gerade auf logarithmischer Skala
    ax1_scatter.plot(x_fit, 10 ** y_fit_1, color='blue',
                     label="Anpassende Gerade (log) math. Modell")
    ax1_scatter.set_xscale('linear')
    ax1_scatter.set_yscale('log')
    ax1_scatter.set_xlim(range_limits)
    ax1_scatter.set_ylim(10 ** np.floor(np.log10(min(relative_freq_roorda))),
                         10 ** np.ceil(np.log10(max(relative_freq_roorda))))  # Anpassung der y-Achsenbegrenzung
    ax1_scatter.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=14)
    ax1_scatter.set_title("Logarithmischer Histogramm-Vergleich", fontweight='bold', fontsize=15)
    ax1_scatter.legend(loc='upper right')
    ax1_scatter.grid(True)

    # Subplot 2
    ax2 = axs[1, 0]
    histdiff_2 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2_2, hist_bins, normalize_01)
    ax2.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
             alpha=0.5, density=True, color='blue')
    hist_vals2, hist_edges2, _ = ax2.hist(intermic_dur2_2, label=label2_2, bins=hist_bins, range=range_limits,
                                          edgecolor='black',
                                          alpha=0.5, density=True, color='red')
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_ylabel("Wahrscheinlichkeitsdichte", fontsize=14)
    ax2.set_title(f"Vergleich {label1} / {label2_2}", fontweight='bold', fontsize=15)
    ax2.legend()
    ax2.set_ylim(0, max(hist_vals1) * 1.1)

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
    ax2_scatter.plot(bin_centers_2, relative_freq_2, 'o', color='red')
    ax2_scatter.plot(x_fit, 10 ** y_fit_roorda, color="blue",
                     label="Anpassende Gerade (log) Roorda")  # Gerade auf logarithmischer Skala
    ax2_scatter.plot(x_fit, 10 ** y_fit_2, color='red',
                     label="Anpassende Gerade (log) math. Modell")
    ax2_scatter.set_xscale('linear')
    ax2_scatter.set_yscale('log')
    ax2_scatter.set_xlim(range_limits)
    ax2_scatter.set_ylim(10 ** np.floor(np.log10(min(relative_freq_roorda))),
                         10 ** np.ceil(np.log10(max(relative_freq_roorda))))  # Anpassung der y-Achsenbegrenzung
    ax2_scatter.set_xlabel(xlabel, fontsize=14)
    ax2_scatter.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=14)
    ax2_scatter.set_title(f"Logarithmischer Histogramm-Vergleich", fontweight='bold', fontsize=15)
    ax2_scatter.legend(loc='upper right')
    ax2_scatter.grid(True)
    # Tabelle erstellen
    hd1 = str(round(histdiff_1, 3)).replace('.', ',')
    mean1 = str(round(mean1, 3)).replace('.', ',')
    median1 = str(round(median1, 3)).replace('.', ',')
    std1 = str(round(stdev1, 3)).replace('.', ',')
    mean2 = str(round(mean2_1, 3)).replace('.', ',')
    median2 = str(round(median2_1, 3)).replace('.', ',')
    std2 = str(round(stdev2_1, 3)).replace('.', ',')
    table_data = [[f"HD={hd1}", "Mittelwert", "Median", "Standardabw."],
                  [label1, mean1, median1, std1],
                  ['Math. Modell', mean2, median2, std2]
                  ]
    table1 = ax1.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(13)
    table1.scale(0.6, 1.5)
    # Fette Schrift für HD-Zelle
    table1[(0, 0)].set_text_props(fontweight='bold')
    hd2 = str(round(histdiff_2, 3)).replace('.', ',')
    mean3 = str(round(mean2_2, 3)).replace('.', ',')
    median3 = str(round(median2_2, 3)).replace('.', ',')
    std3 = str(round(stdev2_2, 3)).replace('.', ',')
    table_data = [[f"HD={hd2}", "Mittelwert", "Median", "Standardabw."],
                  [label1, mean1, median1, std1],
                  ['Math. Modell', mean3, median3, std3]
                  ]
    table2 = ax2.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table2[(0, 0)].set_text_props(fontweight='bold')
    table2.auto_set_font_size(False)
    table2.set_fontsize(13)
    table2.scale(0.6, 1.5)

    plt.tight_layout()
    plt.savefig(f"{savefigpath[:-5]}_histogram_intermicsac_subplot.jpeg", dpi=600)
    plt.close()


def hist_subplot_w_histdiff_log(varlist1, varlist2, label1, label2, savefigpath, xlabel, range_limits=(0, 2.5),
                                normalize_01=False, title=None):
    """
    Generates a histogram subplot with additional scatter plot and logarithmic histogram comparison, including statistical values and fitting lines.

    Parameters:
    - varlist1 (list): List of values for the first dataset.
    - varlist2 (list): List of values for the second dataset.
    - label1 (str): Label for the first dataset.
    - label2 (str): Label for the second dataset.
    - savefigpath (str): Path to save the generated plots.
    - xlabel (str): Label for the x-axis.
    - range_limits (tuple): Limits for the histogram x-axis (default is (0, 2.5)).
    - normalize_01 (bool): Whether to normalize the histograms to the range [0, 1].
    - title (str): Optional title for the subplot.

    Returns:
    - None

    Notes:
    - The function generates a histogram subplot with an additional scatter plot and logarithmic histogram comparison.
    - A fitting line is added to the scatter plot based on logarithmic transformations.
    - Statistical values, such as mean, median, and standard deviation, are displayed in a table.
    - The resulting plot is saved as a JPEG file under the specified 'savefigpath'.
    """
    if not isinstance(varlist1, list) and isinstance(varlist2, list):
        return None

    intermic_dur1 = varlist1
    intermic_dur2 = varlist2

    # Berechnung der Statistiken für beide Datensätze
    mean1, median1, stdev1 = np.mean(intermic_dur1), np.median(intermic_dur1), np.std(intermic_dur1)
    mean2_1, median2_1, stdev2_1 = np.mean(intermic_dur2), np.median(intermic_dur2), np.std(intermic_dur2)

    # Figure und Axes für das Haupt-Histogramm erstellen
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})

    # Subplot 1
    ax1 = axs[0]
    hist_bins = 50
    histdiff_1 = Evaluation.normalized_histogram_difference(intermic_dur1, intermic_dur2, hist_bins, normalize_01)
    hist_vals_roorda, hist_edges_roorda, _ = ax1.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits,
                                                      edgecolor='black',
                                                      alpha=0.5, density=True, color='blue')
    hist_vals1, hist_edges1, _ = ax1.hist(intermic_dur2, label=label2, bins=hist_bins, range=range_limits,
                                          edgecolor='black',
                                          alpha=0.5, density=True, color='red')
    ax1.set_ylabel("Wahrscheinlichkeitsdichte", fontsize=14)
    if title == None:
        title = f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2}"
    ax1.set_title(title, fontweight='bold', fontsize=15)
    ax1.set_ylim(0, max(hist_vals1) * 1.1)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.legend()

    def line_function(params, x, y):
        slope, intercept = params
        y_fit = slope * x + intercept
        return np.sum((y - y_fit) ** 2)

    # Scatter für Roorda
    x_fit = np.linspace(range_limits[0], range_limits[1], 100)
    bin_centers_roorda = (hist_edges_roorda[:-1] + hist_edges_roorda[1:]) / 2
    initial_params = [0.1, 0.1]
    total_samples_roorda = len(intermic_dur1)
    relative_freq_roorda = hist_vals_roorda / total_samples_roorda
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
    relative_freq_log_1 = np.log10(hist_vals1 / (total_samples_1 + 1))
    result_1 = minimize(line_function, initial_params, args=(bin_centers_1, relative_freq_log_1))
    slope_fit_1, intercept_fit_1 = result_1.x
    y_fit_1 = slope_fit_1 * x_fit + intercept_fit_1

    ax1_scatter = axs[1]
    ax1_scatter.plot(bin_centers_roorda, relative_freq_roorda, 'o', color='blue')
    ax1_scatter.plot(bin_centers_1, relative_freq_1, 'o', color="red")
    ax1_scatter.plot(x_fit, 10 ** y_fit_roorda, color='blue',
                     label=f"Anpassende Gerade (log) {label1}")  # Gerade auf logarithmischer Skala
    ax1_scatter.plot(x_fit, 10 ** y_fit_1, color='red',
                     label=f"Anpassende Gerade (log) {label2}")
    ax1_scatter.set_xscale('linear')
    ax1_scatter.set_yscale('log')
    ax1_scatter.set_xlim(range_limits)
    ax1_scatter.set_ylim(10 ** np.floor(np.log10(min(relative_freq_roorda))),
                         10 ** np.ceil(np.log10(max(relative_freq_roorda))))  # Anpassung der y-Achsenbegrenzung
    ax1_scatter.set_xlabel(xlabel, fontsize=14)
    ax1_scatter.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=14)
    ax1_scatter.set_title("Logarithmischer Histogramm-Vergleich", fontweight='bold', fontsize=15)
    ax1_scatter.legend(loc='upper right')
    ax1_scatter.grid(True)

    # Tabelle erstellen
    hd1 = round(histdiff_1, 3)
    hd1 = str(hd1).replace('.', ',')
    mean1 = "{:,.3f}".format(round(mean1, 3)).replace('.', ',')
    median1 = "{:,.3f}".format(round(median1, 3)).replace('.', ',')
    std1 = "{:,.3f}".format(round(stdev1, 3)).replace('.', ',')
    mean2 = "{:,.3f}".format(round(mean2_1, 3)).replace('.', ',')
    median2 = "{:,.3f}".format(round(median2_1, 3)).replace('.', ',')
    std2 = "{:,.3f}".format(round(stdev2_1, 3)).replace('.', ',')
    table_data = [[f"HD={hd1}", "Mittelwert", "Median", "Standardabw."],
                  [label1, mean1, median1, std1],
                  [label2, mean2, median2, std2]
                  ]
    table1 = ax1.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(13)
    table1.scale(0.6, 1.2)
    # Fette Schrift für HD-Zelle
    table1[(0, 0)].set_text_props(fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{savefigpath[:-5]}_singlehistogram_intermicsac_subplot.jpeg", dpi=600)
    plt.close()
class Evaluation():

    @staticmethod
    def get_statistics(data):
        """
        This function calculates statistical metrics for a given dataset.

        Parameters:
            data (list or numpy.ndarray): A list or NumPy array of numerical values.

        Returns:
            tuple: A tuple containing the mean, median, and standard deviation of the data, in that order.

            - Mean (Average): Indicates the average distribution of the data.
            - Median: The value that divides the data into two equal halves.
            - Standard Deviation: A measure of the spread of the data.
        """
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        return mean, median, std

    @staticmethod
    def plot_histogram(data, bins):
        mean, _, std = Evaluation.get_statistics(data)
        plt.hist(data, bins=bins, density=True, alpha=0.6, label='Data')
        x = np.linspace(np.min(data), np.max(data), 100)
        norm_dist = stats.norm(loc=mean, scale=std)
        plt.plot(x, norm_dist.pdf(x), 'r', label='Normaldistribution')

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
        Calculate the intermicrosaccadic interval for a given list of microsaccades.

        Parameters:
            const_dict (dict): A dictionary containing constants used in the analysis.
            micsac_list (list or tuple): A list or tuple of microsaccades, where each entry is a tuple (onset, offset).

        Returns:
            list: A list containing the intermicrosaccadic intervals in seconds.
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
        return timedist

    @staticmethod
    def intermicsac_distribution(const_dict, micsac_list):
        intermicsac_list = Evaluation.intermicrosacc_len(const_dict, micsac_list)
        mean, median, stabw = Evaluation.get_statistics(intermicsac_list)
        return mean, median, stabw

    @staticmethod
    def get_micsac_statistics(df, const_dict, micsac_list):
        '''
        Calculate statistics for a given list of microsaccades based on a DataFrame.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing eye-tracking data.
            const_dict (dict): A dictionary containing constants used in the analysis.
            micsac_list (list or tuple): A list or tuple of microsaccades, where each entry is a tuple (onset, offset).

        Returns:
            tuple: A tuple containing intermicrosaccadic intervals, durations, amplitudes, and peak velocities.
                   - intermic_dist (list): Inter-microsaccadic intervals in seconds.
                   - duration (list): Durations of individual microsaccades in seconds.
                   - amplitudes (list): Amplitudes of individual microsaccades.
                   - velocity (list): Peak velocities of individual microsaccades.
        '''
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
            duration.append((end - start) / const_dict['f'])
            amplitudes.append(amp)
            velocity.append(v)
        return intermic_dist, duration, amplitudes, velocity

    @staticmethod
    def get_drift_statistics(df, const_dict):  # df has to be without blinks
        '''
        Calculate statistics for drift segments in a DataFrame without blinks.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing eye-tracking data without blinks.
            const_dict (dict): A dictionary containing constants used in the analysis.

        Returns:
            tuple: A tuple containing amplitudes and peak velocities of drift segments.
                   - amp (list): Amplitudes of individual drift segments.
                   - velocity (list): Peak velocities of individual drift segments.
        '''
        drift_indexes = EventDetection.drift_indexes_only(df, const_dict)
        amp = []
        velocity = []
        x = df[[const_dict['x_col'], const_dict['y_col']]].to_numpy()
        vel = vecvel(x, sampling=const_dict['f'])
        for drift_segment in drift_indexes:
            start = drift_segment[0]
            end = drift_segment[1]
            amplituden = np.sqrt(df[const_dict['x_col']][start:end] ** 2 + df[const_dict['y_col']][start:end] ** 2)
            amp.append(np.max(amplituden) - np.min(amplituden))
            v = np.max(np.sqrt(vel[start:end, 0] ** 2 + vel[start:end, 1] ** 2))  # Caluclating max speed in segment
            velocity.append(v)
        return amp, velocity

    @staticmethod
    def get_tremor_statistics(df, const_dict):  # df has to be without blinks
        '''
            Calculate statistics for tremor segments in a DataFrame without blinks.

            Parameters:
                df (pandas.DataFrame): The DataFrame containing eye-tracking data without blinks.
                const_dict (dict): A dictionary containing constants used in the analysis.

            Returns:
                tuple: A tuple containing amplitudes and peak velocities of tremor segments.
                       - amp (list): Amplitudes of individual tremor segments.
                       - velocity (list): Peak velocities of individual tremor segments.
        '''
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

    @staticmethod
    def evaluate_json_hist(json_filepath,
                           suptitle='Histograms of Microsaccade Amplitudes, Intermicrosaccadic Intervals, and Number of Microsaccades',
                           label_amplitude="Microsaccade Amplitudes in Arcminutes [arcmin]",
                           label_micsacdur="Intermicorsaccadic Intervals in seconds",
                           label_num_micsac="Number of Microsaccades per Simulation"):
        '''
        Generate and save histograms based on data from a JSON file.

        Parameters:
            json_filepath (str): The path to the JSON file containing microsaccade data.
            suptitle (str, optional): The main title for the entire plot. Defaults to 'Histograms of Microsaccade Amplitudes, Intermicrosaccadic Intervals, and Number of Microsaccades'.
            label_amplitude (str, optional): The label for the amplitude histogram. Defaults to 'Microsaccade Amplitudes in Arcminutes [arcmin]'.
            label_micsacdur (str, optional): The label for the intermicorsaccadic interval histogram. Defaults to 'Intermicorsaccadic Intervals in seconds'.
            label_num_micsac (str, optional): The label for the number of microsaccades histogram. Defaults to 'Number of Microsaccades per Simulation'.
        '''
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
        micsac_amp = data["MicsacAmplitudes"]
        intermic_dur = data["IntermicDur"]
        num_micsac = data["NumMicsac"]

        # Figure subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        axs[0].hist(micsac_amp, bins=50, alpha=0.5, edgecolor='black',
                    label=f"{label_amplitude}", density=True)
        axs[1].hist(intermic_dur, bins=50, alpha=0.5, edgecolor='black', range=(0, 2),
                    label=f"{label_micsacdur}", density=True)
        axs[2].hist(num_micsac, bins=50, alpha=0.5, edgecolor='black',
                    label=f"{label_num_micsac}", density=True)

        fig.suptitle(f"{suptitle}", fontsize=16)
        plt.legend(loc='upper right')

        for ax in axs:
            ax.set_ylabel('Frequency')

        # Customize each histogram
        axs[0].set_title(label_amplitude)
        axs[0].set_xlabel(f'{label_amplitude}')
        axs[0].set_xlim(0, max(micsac_amp))

        axs[1].set_title(label_micsacdur)
        axs[1].set_xlabel(f'{label_micsacdur}')

        axs[2].set_title(label_num_micsac)
        axs[2].set_xlabel(f'{label_num_micsac}')
        axs[2].set_xlim(min(num_micsac), max(num_micsac))

        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f"{str(json_filepath)[:-5]}_histogram.jpeg", dpi=350)
        plt.close()

    @staticmethod
    def generate_histogram_with_logfit(json_filepath, range=(0, 4)):
        """
        Create a histogram from data in a JSON file and add a fit line on a logarithmic scale.

        Args:
            json_filepath (str): The file path to the JSON file containing the data.
            range (tuple): The range limit for the histogram (e.g., (0, 10)).

        Returns:
            None
        """
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
        intermic_dur = data["IntermicDur"]
        mean, median, stdev = np.mean(intermic_dur), np.median(intermic_dur), np.std(intermic_dur)
        # Figure subplots
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
        axs.set_title(
            f"Histogramm der intermikrosakkadischen Intervalle\n Mean={round(mean, 3)}, Median={round(median, 3)}, Stdev={round(stdev, 3)}")
        axs.legend()

        plt.savefig(f"{str(json_filepath)[:-5]}_{range}_histogram.jpeg", dpi=600)
        plt.close()

    @staticmethod
    def get_histogram_log_from_list(liste, savefigpath, const_dict):
        """
        Create a histogram from a given list of values and add a fit line on a logarithmic scale.

        Args:
            liste (list): The list of values for which the histogram should be created.
            savefigpath (str): The file path to save the histogram as a JPEG file.
            const_dict (dict): A dictionary with constants and information for labeling the histogram.

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
        axs.set_title(
            f"Histogramm der intermikrosakkadischen Intervalle - {const_dict['Name']}\n Mean={round(mean, 3)}, Median={round(median, 3)}, Stdev={round(stdev, 3)}")
        axs.legend()

        plt.savefig(f"{savefigpath}\histogram_intermicsac.jpeg", dpi=600)
        plt.close()

    @staticmethod
    def normalized_histogram_difference(list1, list2, num_bins=10, normalize_01=False):
        """
        Calculate the normalized histogram difference between two lists.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.
            num_bins (int): The number of bins in the histogram.

        Returns:
            float: The normalized histogram difference between the two lists.
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
    def dual_hist_w_histdiff(varlist1, varlist2, label1, label2, savefigpath, xlabel, range_limits=(0, 4),
                             normalize_01=False):
        """
        Create and save a histogram comparison of intermicrosaccadic intervals between two datasets.

        Args:
            varlist1 (list): The intermicrosaccadic intervals for the first dataset.
            varlist2 (list): The intermicrosaccadic intervals for the second dataset.
            label1 (str): Label for the first dataset.
            label2 (str): Label for the second dataset.
            savefigpath (str): The file path to save the histogram comparison as a JPEG file.
            xlabel (str): Label for the x-axis.
            range_limits (tuple, optional): The range limit for the histogram (e.g., (0, 2)). Defaults to None.
            normalize_01 (bool, optional): Whether to normalize the histogram difference to the range [0, 1]. Defaults to False.

        Returns:
            None
        """
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
        axs.hist(intermic_dur1, label=label1, bins=hist_bins, range=range_limits, edgecolor='black',
                 alpha=0.5, density=True)

        # Histogramm für den zweiten Datensatz plotten
        axs.hist(intermic_dur2, label=label2, bins=hist_bins, range=range_limits, edgecolor='black',
                 alpha=0.5, density=True)

        # Beschriftungen und Legende hinzufügen
        axs.set_xlabel(xlabel)
        axs.set_ylabel("Wahrscheinlichkeitsdichte")
        axs.set_title(f"Histogramm der intermikrosakkadischen Intervalle\n Vergleich {label1} / {label2}",
                      fontweight='bold', fontsize=14)
        # axs.text(0.95, 0.85, f"HD = {histdiff}", transform=axs.transAxes, ha='right', va='top', fontsize=12)

        axs.legend()
        # Tabelle erstellen
        table_data = [["", "Mean", "Median", "Stdev"],
                      [label1, round(mean1, 3), round(median1, 3), round(stdev1, 3)],
                      [label2, round(mean2, 3), round(median2, 3), round(stdev2, 3)],
                      [f"HD={round(histdiff, 3)}", "", "", ""]
                      ]
        table = axs.table(cellText=table_data, colLabels=None, loc='center right', cellLoc='center')
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
        Generate a histogram and calculate statistical values for intermicrosaccadic intervals based on
        the given CSV files in the specified folder.

        Parameters:
        - folderpath (str): The path to the folder containing the CSV files.
        - savefigpath (str): The path under which the histogram and statistical values will be saved.
        - const_dict (dict): A dictionary with constants containing column names and other values.
        - return_data (bool): An optional parameter indicating whether the calculated data should be returned.

        Returns (if return_data=True):
        - intermic_dur (list): A list of intermicrosaccadic intervals in seconds.
        - amp (list): A list of the amplitudes of the intervals.
        - micsac_vel (list): A list of the velocities of the intervals.

        Notes:
        - The function reads CSV files from the specified folder and performs various calculations on the data.
        - It removes blink annotations from the data and calculates intermicrosaccadic intervals, amplitudes, and velocities.
        - A histogram of the intervals is created and saved as a JPEG file under 'savefigpath'.
        - Additionally, logarithmic histogram data is saved.
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

        plt.title('Histogramm der intermicrosakkadischen Intervalle - Roorda Lab')
        plt.xlabel('Dauer in Sekunden [s]')
        plt.ylabel('Häufigkeit')

        # Diagramm speichern
        plt.savefig(f'{speicherpfad}\histogram_Roorda_intermicsac.jpeg', dpi=600)
        plt.close()
        Evaluation.get_histogram_log_from_list(intermic_dur, speicherpfad, const_dict=const_dict)

    @staticmethod
    def get_intermicsac_roorda(dataframe):
        micsacs = Microsaccades.get_roorda_micsac(dataframe)
        intermic_dist = []
        for i in range(0, len(micsacs)):
            if i == 0:
                intermic_dist.append([0, micsacs[i][0]])
            else:
                intermic_dist.append([micsacs[i - 1][1], micsacs[i][0]])
        return intermic_dist



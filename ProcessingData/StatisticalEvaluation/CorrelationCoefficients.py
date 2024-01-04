"""
File: analyze_gaze_data.py
Author: Fabian Anzlinger
Date: 04.02.2024
Description: This script reads gaze data from a CSV file, calculates correlation coefficients,
and generates a scatterplot with a linear fit.
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr, ks_2samp, linregress
import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

def calculate_correlation_coefficients(drifts, ms):
    # Pearson's correlation coefficient
    pearson_corr, _ = pearsonr(drifts, ms)

    # Spearman's rank correlation coefficient
    spearman_corr, _ = spearmanr(drifts, ms)

    # Kolmogorov-Smirnov test
    ks_statistic, ks_p_value = ks_2samp(drifts, ms)

    return pearson_corr, spearman_corr, ks_p_value

def plot_scatter_and_fit(drifts, ms, pearson_corr, spearman_corr, ks_p_value, save_path):
    # Create scatterplot
    plt.scatter(drifts, ms, label='Data Points')
    plt.xlabel('Drift Duration in [ms]', fontsize=14)
    plt.ylabel('Microsaccade Duration in [ms]', fontsize=14)
    plt.title('Scatterplot of Drift Duration and Microsaccade Duration', fontsize=16)

    # Linear fit
    slope, intercept, r_value, p_value, std_err = linregress(drifts, ms)
    line = slope * drifts + intercept

    # Add linear fit to the plot
    plt.plot(drifts, line, color='red', label=f'Linear Fit (p-Value: {p_value:.4f})')

    # Add text with correlation coefficients and p-value
    text = f'Pearson Correlation: {pearson_corr:.4f}\nSpearman Correlation: {spearman_corr:.4f}\nKolmogorov-Smirnov p-Value: {ks_p_value:.3f}'

    # Display the plot
    legend = plt.legend(loc='upper right')
    plt.text(12500, 150, text, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.5'))
    plt.gca().add_artist(legend)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

def main(file_path, save_path):
    data = read_data(file_path)
    drifts = np.array(list(data['drift']))
    ms = np.array(list(data['ms']))

    pearson_corr, spearman_corr, ks_p_value = calculate_correlation_coefficients(drifts, ms)

    print(f'Statistics for the length of drifts with corresponding microsaccades:\n\nPearson Correlation: {pearson_corr}\nSpearman Correlation: {spearman_corr}\nKolmogorov-Smirnov p-Value: {ks_p_value} ')

    plot_scatter_and_fit(drifts, ms, pearson_corr, spearman_corr, ks_p_value, save_path)

# Example usage
data_file_path = r"C:\Users\uvuik\Documents\Code\MasterarbeitIAI\GeneratingTraces_RGANtorch\FEM\GazeBaseLabels.csv"
save_plot_path = fr"C:\Users\uvuik\bwSyncShare\Documents\ScatterplotGazeBaseDriftMS.jpg"

main(data_file_path, save_plot_path)
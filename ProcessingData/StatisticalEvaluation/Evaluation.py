import json

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox.microsac_detection import \
    vecvel
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation.FixationalEyeMovementDetection import EventDetection
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades


class Evaluation():

    @staticmethod
    def get_statistics(data):
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        return mean, median, std

    @staticmethod
    def plot_histogram(data):
        mean, _, std = Evaluation.get_statistics(data)
        plt.hist(data, bins=20, density=True, alpha=0.6, label='Daten')
        x = np.linspace(np.min(data), np.max(data), 100)
        norm_dist = stats.norm(loc=mean, scale=std)
        plt.plot(x, norm_dist.pdf(x), 'r', label='Normalverteilung')

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

        axs[0].hist(micsac_amp, bins=50, alpha=0.5,
                 label="Amplituden der Mikrosakkaden in Bogenminuten [arcmin]", density=True)
        axs[1].hist(intermic_dur, bins=50, alpha=0.5,
                 label="Intermikrosakkadische Intervalle in s",density=True)
        axs[2].hist(num_micsac, bins=50, alpha=0.5,
                 label="Anzahl an Mikrosakkaden pro Simulation", density=True)

        fig.suptitle(
            'Histogramme der Mikrosakkadenamplitude, Intermikrosakkadischen Intervalle und Anzahl der Mikrosakkaden', fontsize=16)
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

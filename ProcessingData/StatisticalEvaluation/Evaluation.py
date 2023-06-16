import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm

from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox.microsac_detection import \
    vecvel


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

            v = np.max(np.sqrt(vel[start:end, 0]**2 + vel[start:end, 1]**2))
            #v = amp / ((micsac_offset - micsac_onset) / const_dict['f'])
            # Gibt Geschwindigkeit in der Einheit der Eingabe zurück
            duration.append((end - start) / const_dict['f'])
            amplitudes.append(amp)
            velocity.append(v)

            # Hier ist ein Unterschied in der AMplitude detetiert worden: Entsteht durch Filtern bei Detektion der MS
        return intermic_dist, duration, amplitudes, velocity

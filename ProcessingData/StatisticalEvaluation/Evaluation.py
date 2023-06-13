import FixationalEyeMovementDetection
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class Evaluation():

    @staticmethod
    def get_statistics(data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std
    @staticmethod
    def plot_distributions(data, mean, std):
        plt.hist(data, bins=20, density=True, alpha=0.6, label='Daten')
        x = np.linspace(np.min(data), np.max(data), 100)
        norm_dist = stats.norm(loc=mean, scale=std)
        plt.plot(x, norm_dist.pdf(x), 'r', label='Normalverteilung')

    @staticmethod
    def intermicrosacc_dist(df, const_dict, micsac_list):
        timedist = []
        for i in range(len(micsac_list)-1):
            timedist.append(df[const_dict['time_col']][micsac_list[i][0]])



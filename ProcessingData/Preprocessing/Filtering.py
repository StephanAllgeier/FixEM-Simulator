import scipy
import numpy as np
import statsmodels.datasets

class Filtering():
    @staticmethod
    def fft_transform(df, const_dict, coordinate):
        fft = scipy.fft.fft(df[const_dict[coordinate]].values)
        fftfreq = scipy.fft.fftfreq(len(fft), 1/const_dict['f'])
        return fft, fftfreq
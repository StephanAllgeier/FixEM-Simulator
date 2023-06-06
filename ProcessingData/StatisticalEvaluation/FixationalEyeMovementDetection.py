import copy

import numpy as np
import scipy
import pandas as pd
from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox import microsac_detection
from ProcessingData.Preprocessing.Filtering import Filtering


class EventDetection():
    # -------------------------------------------------------------------------------------------------------------------
    # Drift and Tremor, thus filtered signal necessary
    # Tremor: f =[50Hz, 100Hz]
    @staticmethod
    def filter_tremor(df, constant_dict, lowcut=70, highcut=103, order=5):
        # [70,103] laut "Eye Movement Analysis in Simple Visual Tasks"
        return_frame = copy.deepcopy(df)
        return_frame[constant_dict['x_col']] = Filtering.butter_bandpass_filter(df[constant_dict['x_col']],
                                                                                lowcut=lowcut, highcut=highcut,
                                                                                fs=constant_dict['f'], order=order)
        return_frame[constant_dict['y_col']] = pd.Series(Filtering.butter_bandpass_filter(df[constant_dict['y_col']],
                                                                                          lowcut=lowcut,
                                                                                          highcut=highcut,
                                                                                          fs=constant_dict['f'], order=order))
        return return_frame

    @staticmethod
    def filter_drift(df, constant_dict, highcut=40, order=5): #laut "Eye Movement Analysis in Simple Visual Tasks"
        return_frame = copy.deepcopy(df)
        return_frame[constant_dict['x_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['x_col']],
                                                                                highcut=highcut,
                                                                                fs=constant_dict['f'], order=order))
        return_frame[constant_dict['y_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['y_col']],
                                                                                          highcut=highcut,
                                                                                          fs=constant_dict['f'], order=order))
        return return_frame




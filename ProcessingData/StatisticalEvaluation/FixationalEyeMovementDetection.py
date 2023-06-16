import copy

import numpy as np
import scipy
import pandas as pd
from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox import microsac_detection
from ProcessingData.Preprocessing.Filtering import Filtering
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades
from ProcessingData.Visualize import Visualize


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
        df[constant_dict['x_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['x_col']],
                                                                                highcut=highcut,
                                                                                fs=constant_dict['f'], order=order))
        df[constant_dict['y_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['y_col']],
                                                                                          highcut=highcut,
                                                                                          fs=constant_dict['f'], order=order))
        return df

    @staticmethod
    def drift_only(df, const_dict, micsac_mindur=10, micsac_vfac=21):
        #remove blinks
        blink_rm = Interpolation.remove_blink_annot(df, const_dict) #Noch alle Frequenzbänder vorhanden
        fft, fftfreq = Filtering.fft_transform(blink_rm, const_dict, 'x_col')
        Visualize.plot_fft(fft, fftfreq)
        #remove micsacs
        removed_micsac, drift_segment_indexes = Microsaccades.remove_micsac(blink_rm, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)
        fft, fftfreq = Filtering.fft_transform(removed_micsac, const_dict, 'x_col')# Noch alle Frequenzbänder vorhanden
        Visualize.plot_fft(fft, fftfreq)
        #Lowpassfiltering Signal
        filtered = EventDetection.filter_drift(removed_micsac, const_dict)
        return filtered, drift_segment_indexes

    @staticmethod
    def tremor_only(df, const_dict, micsac_mindur=25, micsac_vfac=21):
        # remove blinks
        blink_rm = Interpolation.remove_blink_annot(df, const_dict)  # Noch alle Frequenzbänder vorhanden

        # remove micsacs
        removed_micsac, _ = Microsaccades.remove_micsac(blink_rm, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)

        # Passbandfiltering Signal
        filtered = EventDetection.filter_tremor(removed_micsac, const_dict)
        return filtered




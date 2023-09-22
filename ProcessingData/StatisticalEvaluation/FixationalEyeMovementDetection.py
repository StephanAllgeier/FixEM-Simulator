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
    def filter_tremor(df, constant_dict, lowcut=40, highcut=103, order=5):
        # [70,103] laut "Eye Movement Analysis in Simple Visual Tasks"
        return_frame = copy.deepcopy(df)
        return_frame[constant_dict['x_col']] = Filtering.butter_bandpass_filter(df[constant_dict['x_col']],
                                                                                lowcut=lowcut, highcut=highcut,
                                                                                fs=constant_dict['f'], order=order)
        return_frame[constant_dict['y_col']] = pd.Series(Filtering.butter_bandpass_filter(df[constant_dict['y_col']],
                                                                                          lowcut=lowcut,
                                                                                          highcut=highcut,
                                                                                          fs=constant_dict['f'],
                                                                                          order=order))
        return return_frame

    @staticmethod
    def filter_drift(df, constant_dict, highcut=40, order=5):  # laut "Eye Movement Analysis in Simple Visual Tasks"
        df[constant_dict['x_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['x_col']],
                                                                               highcut=highcut,
                                                                               fs=constant_dict['f'], order=order))
        df[constant_dict['y_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['y_col']],
                                                                               highcut=highcut,
                                                                               fs=constant_dict['f'], order=order))
        return df

    @staticmethod
    def drift_only_wo_micsac(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        if const_dict['rm_blink'] == False:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)  # Noch alle Frequenzbänder vorhanden
        # fft, fftfreq = Filtering.fft_transform(df, const_dict, 'x_col')
        # Visualize.plot_fft(fft, fftfreq) #HIER SIND NOCH ALLE FREQUENZBÄNDER VORHANDEN
        # remove micsacs
        removed_micsac = Microsaccades.remove_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)
        # fft, fftfreq = Filtering.fft_transform(removed_micsac, const_dict,
        #   'x_col')  # Noch alle Frequenzbänder vorhanden
        # Visualize.plot_fft(fft, fftfreq) #TODO: Wieso sieht das Signal hier so anders aus? Entsteht durch das Splicen des Signals an Stellen der Mikrosakkaden
        # Lowpassfiltering Signal
        filtered = EventDetection.filter_drift(removed_micsac, const_dict)
        return filtered

    @staticmethod
    def drift_interpolated(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        if const_dict['rm_blink'] == False:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)  # Noch alle Frequenzbänder vorhanden
        # interpolate micsacs
        interpol_micsac = Microsaccades.interpolate_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)
        # Lowpassfiltering Signal
        filtered = EventDetection.filter_drift(interpol_micsac, const_dict)
        return filtered

    @staticmethod
    def drift_only(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        if const_dict['rm_blink'] == False:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # Lowpassfiltering Signal
        filtered = EventDetection.filter_drift(df, const_dict)
        return filtered

    @staticmethod
    def tremor_only_wo_micsac(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        if const_dict['rm_blink'] == False:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # remove micsacs
        removed_micsac = Microsaccades.remove_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)

        # Passbandfiltering Signal
        filtered = EventDetection.filter_tremor(removed_micsac, const_dict)
        return filtered

    @staticmethod
    def tremor_only(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        if const_dict['rm_blink'] == False:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # Passbandfiltering Signal
        filtered = EventDetection.filter_tremor(df, const_dict)
        return filtered

    @staticmethod
    def drift_indexes_only(df, const_dict, mindur=6, vfac=21):
        dataframe = copy.deepcopy(df)
        if const_dict['rm_blink'] == False:
            # remove blinks
            dataframe, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        micsac = Microsaccades.find_micsac(dataframe, const_dict, mindur=mindur, vfac=vfac)
        micsac_list = [[micsac[0][i][0], micsac[0][i][1]] for i in range(len(micsac[0]))]
        i = 0
        drift_segment_indexes = []
        # Cutting 20ms before and after MS to exclude acceleration and deceleration events
        for start_index, end_index in micsac_list:
            if i == 0:
                drift_segment_indexes.append([0, start_index - 1 - round(20 / 1000 * const_dict['f'])])
            else:
                drift_start = micsac_list[i - 1][1] + 1 + round(20 / 1000 * const_dict['f'])
                drift_end = micsac_list[i][0] - 1 - round(20 / 1000 * const_dict['f'])
                if not drift_start > drift_end:
                    drift_segment_indexes.append([drift_start, drift_end])
            i += 1
        return drift_segment_indexes

    @staticmethod
    def micsac_only(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        new_dataframe = df.copy()
        if const_dict['rm_blink'] == False:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)  # Noch alle Frequenzbänder vorhanden
        # get micsacs
        micsac = Microsaccades.find_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)[0]

        # Füge den Startwert 0 für den Anfang des DataFrames hinzu
        first_start = micsac[0][0]
        # new_dataframe.loc[0:first_start, [const_dict['x_col'], const_dict['y_col']]] = 0

        # Iteriere über die Segmente und setze die Werte entsprechend
        for i in range(len(micsac) - 1):
            curr_end = micsac[i][1]
            next_start = micsac[i + 1][0]
            curr_x_value = df.loc[curr_end, const_dict['x_col']]  # Wert der aktuellen End-x-Koordinate
            curr_y_value = df.loc[curr_end, const_dict['y_col']]

            # Setze die Werte zwischen dem aktuellen Endpunkt und dem nächsten Startpunkt auf den aktuellen Wert
            new_dataframe.loc[curr_end + 1:next_start, const_dict['x_col']] = curr_x_value
            new_dataframe.loc[curr_end + 1:next_start, const_dict['y_col']] = curr_y_value

        # Setze den letzten Bereich ab dem letzten Endpunkt auf den letzten Wert des DataFrames
        last_end = micsac[-1][1]
        last_x_value = df.loc[last_end, const_dict['x_col']]
        last_y_value = df.loc[last_end, const_dict['y_col']]
        new_dataframe.loc[last_end + 1:, const_dict['x_col']] = last_x_value
        new_dataframe.loc[last_end + 1:, const_dict['y_col']] = last_y_value
        return new_dataframe

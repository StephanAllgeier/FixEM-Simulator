import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox import microsac_detection
from ProcessingData.Preprocessing.Filtering import Filtering
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.Visualize import Visualize


class Microsaccades():

    @staticmethod
    def count_micsac_annot(df):
        '''
        return number of annotated microsaccades
        '''
        micsac = Microsaccades.get_roorda_micsac(df)
        count = len(micsac)
        return count

    @staticmethod
    def lp_filter(df, constant_dict, highcut=40, order=5):  # laut "Eye Movement Analysis in Simple Visual Tasks"
        df[constant_dict['x_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['x_col']],
                                                                         highcut=highcut,
                                                                         fs=constant_dict['f'], order=order))
        df[constant_dict['y_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['y_col']],
                                                                         highcut=highcut,
                                                                         fs=constant_dict['f'], order=order))
        return df
    @staticmethod
    def find_micsac(df, constant_dict, mindur=3, vfac=5, highcut=40):
        '''
        parameters:
        df=dataframe to work with, units of the traces is in degrees of visual angle
        constant_dict = dictionary belongig to the dataset with information about the structure of the dataset
        coordinate = which coordinate to evaluate
        returns tuple with tuple[0] = microsaccades
        '''
        #Filtering Signal like in Paper "Eye Movement Analysis in Simple Visual Tasks"
        dataframe = copy.deepcopy(df)
        df_2 = Microsaccades.lp_filter(dataframe, constant_dict=constant_dict, highcut=highcut, order=5)
        input_array = df_2[[constant_dict['x_col'], constant_dict['y_col']]].to_numpy()
        micsac = microsac_detection.microsacc(input_array, sampling=constant_dict['f'], mindur=mindur, vfac=vfac)
        return micsac

    @staticmethod
    def get_all_micsac(df, const_dict):
        data = Interpolation.remove_blink_annot(df, const_dict)
        if const_dict['Name'] == "Roorda":
            data = Interpolation.convert_arcmin_to_dva(data, const_dict)
        micsac_detec = Microsaccades.find_micsac(data, const_dict, mindur=mindur, vfac=vfac)
        return data, micsac_detec

    @staticmethod
    def get_roorda_micsac(df):
        # Input is dataframe from Roorda_Database. It Returns list of lists containing onset and offset of microsaccades
        mic_sac_idx = df[df['Flags'] == 1].index
        current_sublist = []
        indexes = []
        for i in range(len(mic_sac_idx)):
            if i == 0 or mic_sac_idx[i] != mic_sac_idx[i - 1] + 1:
                if current_sublist:
                    indexes.append(current_sublist)
                current_sublist = [mic_sac_idx[i]]
            else:
                current_sublist.append(mic_sac_idx[i])

        # Füge die letzte Teil-Liste hinzu, falls vorhanden
        if current_sublist:
            indexes.append(current_sublist)
        micsac_onoff = []
        for liste in indexes:
            micsac_onoff.append([liste[0], liste[-1]])
        return micsac_onoff

    @staticmethod
    def remove_micsac(df, const_dict, mindur=25, vfac=21):
        dataframe = copy.deepcopy(df)
        micsac = Microsaccades.find_micsac(dataframe, const_dict, mindur=mindur, vfac=vfac)
        micsac_annot = Microsaccades.get_roorda_micsac(Interpolation.remove_blink_annot(df, const_dict))
        micsac_list = [[micsac[0][i][0], micsac[0][i][1]] for i in range(len(micsac[0]))]
        xdiff = [[dataframe[const_dict['x_col']].iloc[end_index] - dataframe[const_dict['x_col']].iloc[start_index]] for start_index, end_index in micsac_list]
        ydiff = [[dataframe[const_dict['y_col']].iloc[end_index] - dataframe[const_dict['y_col']].iloc[start_index]] for
                 start_index, end_index in micsac_list]
        i=0
        for start_index, end_index in micsac_list:
            #Visualize.plot_xy(dataframe, const_dict)
            dataframe = dataframe.drop(dataframe.index[start_index:end_index + 1])
            dataframe.loc[start_index:, const_dict['x_col']] -= xdiff[i]
            dataframe.loc[start_index:, const_dict['y_col']] -= ydiff[i]
            i+=1
        dataframe = dataframe.reset_index(drop=True)
        dataframe[const_dict['time_col']] = dataframe.index / const_dict['f']
        return dataframe
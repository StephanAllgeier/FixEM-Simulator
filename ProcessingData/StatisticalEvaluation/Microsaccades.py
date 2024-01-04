"""
Microsaccades Module

This module provides functions for detecting, interpolating, and removing microsaccades in eye-tracking data.
It includes methods for lowpass filtering, microsaccade detection, Roorda Database-specific microsaccade extraction,
interpolation between microsaccades, and removal of microsaccade segments.

Author: Fabian Anzlinger
Date: 04.01.2024

"""
import copy

import numpy as np
import pandas as pd

from ProcessingData.ExternalCode.EngbertMicrosaccadeToolboxmaster.EngbertMicrosaccadeToolbox import microsac_detection
from ProcessingData.Preprocessing.Filtering import Filtering
from ProcessingData.Preprocessing.Interpolation import Interpolation


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
    def lp_filter(df, constant_dict, highcut=40, order=5):
        df[constant_dict['x_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['x_col']],
                                                                               highcut=highcut,
                                                                               fs=constant_dict['f'], order=order))
        df[constant_dict['y_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['y_col']],
                                                                               highcut=highcut,
                                                                               fs=constant_dict['f'], order=order))
        return df

    @staticmethod
    def find_micsac(df, constant_dict, mindur=6, vfac=21, highcut=40, threshold=15):  # threshold in ms
        """
        Detect microsaccades in eye-tracking data.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - constant_dict (dict): Dictionary containing constants for the dataset.
        - mindur (float): Minimum duration for microsaccades in milliseconds.
        - vfac (float): Velocity factor for microsaccade detection.
        - highcut (float): Upper cutoff frequency for lowpass filter.
        - threshold (float): Time threshold in milliseconds for merging adjacent microsaccades.

        Returns:
        - Tuple: Tuple[0] contains a list of microsaccades represented by start and end indexes,
        and Tuple[1] contains the corresponding velocities.

        This function applies a lowpass filter to the eye-tracking data, detects microsaccades based on the specified
        minimum duration and velocity factor, and optionally merges adjacent microsaccades based on a time threshold.
        The result is a tuple containing microsaccades and their corresponding velocities.
        """
        mindur = round(mindur / 1000 * constant_dict['f'])  # Calculation of mindur to indices
        # Filtering Signal like in Paper "Eye Movement Analysis in Simple Visual Tasks"
        dataframe = copy.deepcopy(df)
        df_2 = Microsaccades.lp_filter(dataframe, constant_dict=constant_dict, highcut=highcut, order=5)
        input_array = df_2[[constant_dict['x_col'], constant_dict['y_col']]].to_numpy()
        micsac = microsac_detection.microsacc(input_array, sampling=constant_dict['f'], mindur=mindur, vfac=vfac)
        # Threshold
        if threshold > 0:
            micsac_list = []
            for i in range(len(micsac[0]) - 1):
                if (micsac[0][i + 1][0] - micsac[0][i][1]) / constant_dict['f'] * 1000 < threshold:
                    micsac_list.append([micsac[0][i][0], micsac[0][i + 1][1]])
                elif i != 0 and micsac_list[-1][1] == micsac[0][i][1]:
                    continue
                else:
                    micsac_list.append(micsac[0][i])
                if i != 0 and micsac_list[-1][0] < micsac_list[-2][1]:
                    end_val = micsac_list[-1][1]
                    micsac_list.pop(-1)
                    micsac_list[-1][1] = end_val
            micsac = (micsac_list, micsac[1])
        return micsac

    @staticmethod
    def get_roorda_micsac(df):
        """
        Get microsaccade onset and offset indexes from Roorda Database.

        Args:
        - df (pd.DataFrame): DataFrame from the Roorda Database.

        Returns:
        - List[List[int]]: List of lists containing microsaccade onset and offset indexes.

        This function takes a DataFrame from the Roorda Database and extracts microsaccade onset and offset indexes.
        The result is a list of lists, where each sublist represents the onset and offset indexes of a microsaccade.
        """
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
    def interpolate_micsac(df, const_dict, mindur=6, vfac=21):
        """
        Interpolate microsaccade segments in eye-tracking data.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - mindur (float): Minimum duration for microsaccades.
        - vfac (float): Velocity factor for microsaccade detection.

        Returns:
        - pd.DataFrame: DataFrame with interpolated microsaccade segments.

        This function takes the input DataFrame, identifies microsaccades, and interpolates the values between
        microsaccades to maintain continuity in the signal. It removes blinks (if specified) before processing.
        The resulting DataFrame contains the signals with interpolated values between microsaccades.
        """
        dataframe = df.copy()
        micsac = Microsaccades.find_micsac(dataframe, const_dict, mindur=mindur, vfac=vfac)
        if not const_dict['rm_blink']:
            dataframe, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        micsac_list = [[micsac[0][i][0], micsac[0][i][1]] for i in range(len(micsac[0]))]
        i = 0
        xdiff = [[dataframe[const_dict['x_col']].iloc[end_index] - dataframe[const_dict['x_col']].iloc[start_index]] for
                 start_index, end_index in micsac_list]
        ydiff = [[dataframe[const_dict['y_col']].iloc[end_index] - dataframe[const_dict['y_col']].iloc[start_index]] for
                 start_index, end_index in micsac_list]

        for start_index, end_index in micsac_list:
            dataframe.loc[start_index:, const_dict['x_col']] -= xdiff[i]
            dataframe.loc[start_index:, const_dict['y_col']] -= ydiff[i]
            num_points = end_index - start_index + 1
            x_values = np.linspace(dataframe[const_dict['x_col']].iloc[start_index - 1],
                                   dataframe[const_dict['x_col']].iloc[end_index + 1], num_points)
            y_values = np.linspace(dataframe[const_dict['y_col']].iloc[start_index - 1],
                                   dataframe[const_dict['y_col']].iloc[end_index + 1], num_points)
            dataframe.loc[start_index:end_index, const_dict['x_col']] = x_values
            dataframe.loc[start_index:end_index, const_dict['y_col']] = y_values
            i += 1
        return dataframe

    @staticmethod
    def remove_micsac(df, const_dict, mindur=6, vfac=21):
        """
        Remove microsaccade segments from eye-tracking data.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - mindur (float): Minimum duration for microsaccades.
        - vfac (float): Velocity factor for microsaccade detection.

        Returns:
        - pd.DataFrame: DataFrame with microsaccade segments removed.

        This function takes the input DataFrame, identifies microsaccades, and removes the segments corresponding to
        microsaccades from the DataFrame. It adjusts the x and y values to maintain continuity in the signal.
        Blinks are removed (if specified) before processing. The resulting DataFrame has microsaccade segments removed.
        """
        dataframe = df.copy()
        micsac = Microsaccades.find_micsac(dataframe, const_dict, mindur=mindur, vfac=vfac)
        if not const_dict['rm_blink']:
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)

        micsac_list = [[micsac[0][i][0], micsac[0][i][1]] for i in range(len(micsac[0]))]
        xdiff = [[dataframe[const_dict['x_col']].iloc[end_index] - dataframe[const_dict['x_col']].iloc[start_index]] for
                 start_index, end_index in micsac_list]
        ydiff = [[dataframe[const_dict['y_col']].iloc[end_index] - dataframe[const_dict['y_col']].iloc[start_index]] for
                 start_index, end_index in micsac_list]
        i = 0
        drift_segment_indexes = []

        for start_index, end_index in micsac_list:
            if i == 0:
                drift_segment_indexes.append([0, start_index - 1])
            else:
                drift_segment_indexes.append([micsac_list[i - 1][1] + 1, micsac_list[i][0] - 1])

            # Segmente aus dem DataFrame entfernen
            dataframe.drop(dataframe.index[start_index:end_index + 1], inplace=True)

            # Anpassung der x- und y-Werte um xdiff und ydiff
            dataframe.loc[start_index:, const_dict['x_col']] -= xdiff[i]
            dataframe.loc[start_index:, const_dict['y_col']] -= ydiff[i]

            i += 1

        dataframe.reset_index(drop=True, inplace=True)
        dataframe[const_dict['time_col']] = dataframe.index / const_dict['f']
        return dataframe

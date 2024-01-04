"""
EventDetection Module

This module provides functions for event detection and signal processing in eye-tracking data.

Author: Fabian Anzlinger

Date: 04.01.2024
"""

import copy
import pandas as pd
from ProcessingData.Preprocessing.Filtering import Filtering
from ProcessingData.Preprocessing.Interpolation import Interpolation
from ProcessingData.StatisticalEvaluation.Microsaccades import Microsaccades


class EventDetection():
    @staticmethod
    def filter_tremor(df, constant_dict, lowcut=40, highcut=103, order=5):
        """
        Apply bandpass filtering to remove tremor from eye-tracking data.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - constant_dict (dict): Dictionary containing constants for the dataset.
        - lowcut (float): Lower cutoff frequency for bandpass filter.
        - highcut (float): Upper cutoff frequency for bandpass filter.
        - order (int): Order of the bandpass filter.

        Returns:
        - pd.DataFrame: DataFrame with tremor-filtered signals.

        This function applies a bandpass filter to the eye-tracking data to remove tremor artifacts.
        The bandpass filter is applied to both the x and y coordinates of the eye movements based on the specified
        cutoff frequencies and filter order. The resulting DataFrame contains the tremor-filtered signals.
        """
        # [70,103] following "Eye Movement Analysis in Simple Visual Tasks"
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
        """
        Apply lowpass filtering to remove drift from eye-tracking data.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - constant_dict (dict): Dictionary containing constants for the dataset.
        - highcut (float): Upper cutoff frequency for lowpass filter.
        - order (int): Order of the lowpass filter.

        Returns:
        - pd.DataFrame: DataFrame with drift-filtered signals.

        This function applies a lowpass filter to the eye-tracking data to remove drift artifacts.
        The lowpass filter is applied to both the x and y coordinates of the eye movements based on the specified
        cutoff frequency and filter order. The resulting DataFrame contains the drift-filtered signals.
        """
        df[constant_dict['x_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['x_col']],
                                                                               highcut=highcut,
                                                                               fs=constant_dict['f'], order=order))
        df[constant_dict['y_col']] = pd.Series(Filtering.butter_lowpass_filter(df[constant_dict['y_col']],
                                                                               highcut=highcut,
                                                                               fs=constant_dict['f'], order=order))
        return df

    @staticmethod
    def drift_only_wo_micsac(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        """
        Filter eye-tracking data to isolate drift components without microsaccades.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - micsac_mindur (int): Minimum duration for microsaccades.
        - micsac_vfac (int): Velocity factor for microsaccade detection.

        Returns:
        - pd.DataFrame: DataFrame with isolated drift components after removing microsaccades.

        This function removes blinks, extracts microsaccades, and filters the eye-tracking data to isolate
        the drift components. The resulting DataFrame contains the signals with microsaccades removed,
        focusing on the drift components only.
        """
        if not const_dict['rm_blink']:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # remove micsacs
        removed_micsac = Microsaccades.remove_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)

        # Lowpassfiltering signal
        filtered = EventDetection.filter_drift(removed_micsac, const_dict)
        return filtered

    @staticmethod
    def drift_interpolated(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        """
        Filter eye-tracking data to isolate drift components with interpolated microsaccades.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - micsac_mindur (int): Minimum duration for microsaccades.
        - micsac_vfac (int): Velocity factor for microsaccade detection.

        Returns:
        - pd.DataFrame: DataFrame with isolated drift components after interpolating microsaccades.

        This function removes blinks, interpolates microsaccades, and filters the eye-tracking data to isolate
        the drift components. The resulting DataFrame contains the signals with interpolated microsaccades,
        focusing on the drift components only.
        """
        if not const_dict['rm_blink']:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # interpolate micsacs
        interpol_micsac = Microsaccades.interpolate_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)
        # Lowpassfiltering Signal
        filtered = EventDetection.filter_drift(interpol_micsac, const_dict)
        return filtered

    @staticmethod
    def drift_only(df, const_dict):
        """
        Filter eye-tracking data to isolate drift components.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - micsac_mindur (int): Minimum duration for microsaccades.
        - micsac_vfac (int): Velocity factor for microsaccade detection.

        Returns:
        - pd.DataFrame: DataFrame with isolated drift components.

        This function removes blinks and filters the eye-tracking data to isolate the drift components.
        The resulting DataFrame contains the signals focusing on the drift components only.
        """
        if not const_dict['rm_blink']:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # Lowpassfiltering signal
        filtered = EventDetection.filter_drift(df, const_dict)
        return filtered

    @staticmethod
    def tremor_only_wo_micsac(df, const_dict, micsac_mindur=6, micsac_vfac=21):
        """
        Extract tremor signal excluding microsaccades.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - micsac_mindur (int): Minimum duration for microsaccades.
        - micsac_vfac (int): Velocity factor for microsaccade detection.

        Returns:
        - pd.DataFrame: DataFrame with tremor signal after excluding microsaccades.

        This function removes blinks, then removes microsaccades, and finally applies passband filtering
        to extract the tremor signal from the eye-tracking data.
        """
        if not const_dict['rm_blink']:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # remove micsacs
        removed_micsac = Microsaccades.remove_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)

        # Passbandfiltering Signal
        filtered = EventDetection.filter_tremor(removed_micsac, const_dict)
        return filtered

    @staticmethod
    def tremor_only(df, const_dict):
        """
        Extract tremor signal.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.

        Returns:
        - pd.DataFrame: DataFrame with tremor signal.

        This function removes blinks (if specified), and applies passband filtering
        to extract the tremor signal from the eye-tracking data.
        """
        if not const_dict['rm_blink']:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)
        # Passbandfiltering Signal
        filtered = EventDetection.filter_tremor(df, const_dict)
        return filtered

    @staticmethod
    def drift_indexes_only(df, const_dict, mindur=6, vfac=21):
        """
        Extract indexes corresponding to drift segments.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - mindur (int): Minimum duration for microsaccades.
        - vfac (int): Velocity factor for microsaccade detection.

        Returns:
        - List[List[int]]: List of index pairs representing drift segments.

        This function identifies microsaccades, excludes the acceleration and deceleration events,
        and returns the start and end indexes of drift segments.
        """
        dataframe = copy.deepcopy(df)
        if not const_dict['rm_blink']:
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
        """
        Extract microsaccade-only segments from the eye-tracking data.

        Args:
        - df (pd.DataFrame): Eye-tracking data DataFrame.
        - const_dict (dict): Dictionary containing constants for the dataset.
        - micsac_mindur (float): Minimum duration for microsaccades.
        - micsac_vfac (float): Velocity factor for microsaccades.

        Returns:
        - pd.DataFrame: DataFrame with microsaccade-only segments.

        This function takes the input DataFrame and extracts segments where only microsaccades are present,
        while interpolating the values between microsaccades to maintain continuity in the signal. It removes
        blinks and then identifies microsaccades based on the specified minimum duration and velocity factor.
        The resulting DataFrame contains only microsaccade segments with interpolated values between them.
        """
        new_dataframe = df.copy()
        if not const_dict['rm_blink']:
            # remove blinks
            df, const_dict = Interpolation.remove_blink_annot(df, const_dict)  # Noch alle Frequenzbänder vorhanden
        # get micsacs
        micsac = Microsaccades.find_micsac(df, const_dict, mindur=micsac_mindur, vfac=micsac_vfac)[0]

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

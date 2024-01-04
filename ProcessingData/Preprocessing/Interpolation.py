"""
Interpolation Module

This module provides methods for data interpolation, manipulation, and conversion related to eye-tracking data.

Classes:
    Interpolation: A class containing static methods for various interpolation and data processing operations.

Methods:
    interp_monocub(df, const_dict):
    interp_akima(df, const_dict):
    interp_cubic(df, const_dict):
    splice_together(df, const_dict):
    remove_blink(df, const_dict, time_cutoff, remove_start=False, remove_end=False, remove_start_time=0, remove_end_time=0):
    remove_blink_annot(df, const_dict, cutoff=0):
    remove_sacc_annot(df, const_dict):
    convert_arcmin_to_dva(df, const_dict):
    arcmin_to_µm(df, const_dict, r_eye=12.5):
    dva_to_arcmin(df, const_dict):
Author: Fabian Anzlinger
Date: 04.01.2024
"""
import copy
import numpy as np


class Interpolation():
    @staticmethod
    def interp_monocub(df, const_dict):
        data_frame = copy.deepcopy(df)
        data_frame[const_dict['x_col']] = data_frame[const_dict['x_col']].interpolate(method='pchip')
        data_frame[const_dict['y_col']] = data_frame[const_dict['y_col']].interpolate(method='pchip')
        return data_frame

    @staticmethod
    def interp_akima(df, const_dict):
        data_frame = copy.deepcopy(df)
        data_frame[const_dict['x_col']] = data_frame[const_dict['x_col']].interpolate(method='akima')
        data_frame[const_dict['y_col']] = data_frame[const_dict['y_col']].interpolate(method='akima')
        return data_frame

    @staticmethod
    def interp_cubic(df, const_dict):
        data_frame = copy.deepcopy(df)
        data_frame[const_dict['x_col']] = data_frame[const_dict['x_col']].interpolate(method='cubic')
        data_frame[const_dict['y_col']] = data_frame[const_dict['y_col']].interpolate(method='cubic')
        return data_frame

    @staticmethod
    def splice_together(df, const_dict):
        data_frame = df.dropna(subset=[const_dict['x_col'], const_dict['y_col']], how='any')
        data_frame = data_frame.reset_index()
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f']
        return data_frame

    @staticmethod
    def remove_blink(df, const_dict, time_cutoff, remove_start=False, remove_end=False, remove_start_time=0,
                     remove_end_time=0):  # time_cutoff in ms
        """
       Remove blink movements from the DataFrame based on missing values in x and y coordinates.

       Parameters:
           df (pandas.DataFrame): The DataFrame containing the eye-tracking data.
           const_dict (dict): A dictionary mapping column names and constants used in the DataFrame.
           time_cutoff (float): The time duration (in milliseconds) around missing values to be considered as blink.
           remove_start (bool, optional): If True, remove data before the start time. Defaults to False.
           remove_end (bool, optional): If True, remove data after the end time. Defaults to False.
           remove_start_time (float, optional): The start time for removal (in seconds). Defaults to 0.
           remove_end_time (float, optional): The end time for removal (in seconds). Defaults to 0.

       Returns:
           tuple: A tuple containing the modified DataFrame and updated constants.
                  - data_frame (pandas.DataFrame): The DataFrame with blink movements removed.
                  - const_dict (dict): The updated constants dictionary.

       Example:
           data_frame, updated_constants = remove_blink(input_data, {'x_col': 'column_x', 'y_col': 'column_y', 'f': 100}, time_cutoff=50, remove_start=True)
       """
        data_frame = copy.deepcopy(df)

        x_nan = df[df[const_dict['x_col']].isnull()].index.to_list()
        y_nan = df[df[const_dict['y_col']].isnull()].index.to_list()
        if x_nan != y_nan:
            print('Unequal length of x and y.')
        # Splitting lists into sublists
        indexes = []
        current_sublist = []
        padding_count = round(time_cutoff * const_dict['f'] // 1000)
        for i in range(len(x_nan)):
            if i == 0 or x_nan[i] != x_nan[i - 1] + 1:
                if current_sublist:
                    padded_list = list(
                        range(current_sublist[0] - padding_count, current_sublist[0])) + current_sublist + list(
                        range(current_sublist[-1] + 1, current_sublist[-1] + padding_count + 1))
                    indexes.append(padded_list)
                current_sublist = [x_nan[i]]
            else:
                current_sublist.append(x_nan[i])

        if current_sublist:
            indexes.append(current_sublist)
        # Delete Rows from Frame
        for liste in indexes:
            try:
                data_frame = data_frame.drop(liste)
            except:
                pass
        if remove_end:
            data_frame = data_frame[0:-remove_end_time * const_dict['f'] // 1000]
        if remove_start:
            data_frame = data_frame[remove_start_time * const_dict['f'] // 1000:]
        data_frame = data_frame.reset_index(drop=True)
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f'] / const_dict['TimeScaling']
        const_dict['rm_blink'] = True
        return data_frame, const_dict

    @staticmethod
    def remove_blink_annot(df, const_dict, cutoff=0):
        data_frame = copy.deepcopy(df)
        blink_idx = df[df[const_dict['Annotations']] == const_dict['BlinkID']].index
        current_sublist = []
        indexes = []

        for i in range(len(blink_idx)):
            if i == 0 or blink_idx[i] != blink_idx[i - 1] + 1:
                if current_sublist:
                    # Remove before and after blink
                    start = max(0, current_sublist[0] - int(const_dict['f'] * cutoff))
                    end = min(len(data_frame), current_sublist[-1] + int(const_dict['f'] * cutoff)+1)
                    indexes.extend(range(start, end))
                current_sublist = [blink_idx[i]]
            else:
                current_sublist.append(blink_idx[i])

        if current_sublist:
            start = max(0, current_sublist[0] - int(const_dict['f'] * cutoff))
            end = min(len(data_frame), current_sublist[-1] + int(const_dict['f'] * cutoff)+1)
            indexes.extend(range(start, end))

        data_frame = data_frame.drop(indexes)
        data_frame = data_frame.reset_index(drop=True)
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f']
        const_dict['rm_blink'] = True
        return data_frame, const_dict

    @staticmethod
    def remove_sacc_annot(df, const_dict):
        data_frame = copy.deepcopy(df)
        sakk_idx = df[df[const_dict['Annotations']] == const_dict['SakkID']].index
        current_sublist = []
        indexes = []
        for i in range(len(sakk_idx)):
            if i == 0 or sakk_idx[i] != sakk_idx[i - 1] + 1:
                if current_sublist:
                    indexes.append(current_sublist)
                current_sublist = [sakk_idx[i]]
            else:
                current_sublist.append(sakk_idx[i])
        if current_sublist:
            indexes.append(current_sublist)
        for liste in indexes:
            data_frame = data_frame.drop(liste)
        data_frame = data_frame.reset_index(drop=True)
        data_frame[const_dict['time_col']] = data_frame.index / const_dict['f']
        const_dict['rm_sakk'] = True
        return data_frame, const_dict

    @staticmethod
    def convert_arcmin_to_dva(df, const_dict):
        return_df = copy.deepcopy(df)
        factor = 1 / 60
        return_df[const_dict['x_col']], return_df[const_dict['y_col']] = return_df[const_dict['x_col']].multiply(
            factor), return_df[const_dict['y_col']].multiply(factor)
        return return_df


    @staticmethod
    def arcmin_to_µm(df, const_dict, r_eye=12.5):
        df['x_µm'] = r_eye * np.sin((df[const_dict['x_col']] / 60) * np.pi / 180) * 1000
        df['y_µm'] = r_eye * np.sin((df[const_dict['y_col']] / 60) * np.pi / 180) * 1000
        const_dict['x_µm'] = 'x_µm'
        const_dict['y_µm'] = 'y_µm'
        return df, const_dict

    @staticmethod
    def dva_to_arcmin(df, const_dict):
        return_df = copy.deepcopy(df)
        return_df[const_dict['x_col']] = return_df[const_dict['x_col']] * 60
        return_df[const_dict['y_col']] = return_df[const_dict['y_col']] * 60
        const_dict['ValScaling'] = 1
        return return_df, const_dict

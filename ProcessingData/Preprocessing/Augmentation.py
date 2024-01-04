"""
Signal Processing Augmentation Module

This module provides functions for augmenting time-series data, particularly for signal processing applications.
The implemented functions include:
1. add_gaussian_noise(dataframe, dataframe_dict, mean=0, std=0.01)
2. slice_df(dataframe, const_dict, segment_length)
3. flip_dataframe(dataframe, const_dict)
4. resample(df, const_dict, f_target=1000)

Author: Fabian Anzlinger
Date: 04.01.2024
"""

import numpy as np
import pandas as pd
from scipy import signal


class Augmentation():
    @staticmethod
    def add_gaussian_noise(dataframe, dataframe_dict, mean=0, std=0.01):
        """
        Adds Gaussian noise to the 'x' and 'y' columns of the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame with columns 't', 'x', and 'y'.
            mean (float): Mean of the normal distribution for the noise.
            std (float): Standard deviation of the normal distribution for the noise.

        Returns:
            pd.DataFrame: The original DataFrame with added noise.
        """
        noisy_data = dataframe.copy()
        x_col = dataframe_dict['x_col']
        y_col = dataframe_dict['y_col']
        noise = np.random.normal(mean, std, size=len(noisy_data))
        noisy_data[x_col] += noise
        noisy_data[y_col] += noise
        return noisy_data

    @staticmethod
    def slice_df(dataframe, const_dict, segment_length):
        """
        Slices the DataFrame into sub-DataFrames of each {segment_length} seconds.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be sliced.
                                      It must have a 'time' column.

        Returns:
            List[pd.DataFrame]: A list of sub-DataFrames, each with {segment_length} seconds length.
        """
        f = const_dict['f']
        samples_per_segment = int(f * segment_length)
        segments = []
        start_idx = 0

        while start_idx + samples_per_segment <= len(dataframe):
            end_idx = start_idx + samples_per_segment
            segment = dataframe.iloc[start_idx:end_idx]
            segments.append(segment.values)
            start_idx = end_idx
        return segments

    @staticmethod
    def flip_dataframe(dataframe, const_dict):
        """
        Changes the sign of the 'x' and 'y' columns in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame in which to change the sign.

        Returns:
            pd.DataFrame: The modified DataFrame with changed signs in the 'x' and 'y' columns.
        """
        modified_data = dataframe.copy()
        x_col = const_dict['x_col']
        y_col = const_dict['y_col']
        modified_data[[x_col, y_col]] = -modified_data[[x_col, y_col]]
        return modified_data

    @staticmethod
    def resample(df, const_dict, f_target=1000):
        """
        Resamples a signal from the original frequency to a target frequency.

        Args:
            df (pd.DataFrame): The original DataFrame with columns 'time', 'x', 'y', and 'Annotations'.
            const_dict (dict): A dictionary containing constants, including 'time_col', 'x_col', 'y_col', 'Annotations', and 'f'.
            f_target (int, optional): Target frequency for resampling (default=1000).

        Returns:
            Tuple[pd.DataFrame, dict]: A tuple containing the resampled DataFrame and the updated constant dictionary.
        """
        interm_frame = df[[const_dict['time_col'], const_dict['x_col'], const_dict['y_col'], const_dict['Annotations']]]
        fs = const_dict['f']
        resampling_ratio = f_target / fs
        num_output_samples = int(len(interm_frame) * resampling_ratio)

        return_x = pd.Series(signal.resample(interm_frame[const_dict['x_col']], num_output_samples),
                             name=const_dict['x_col'])
        return_y = pd.Series(signal.resample(interm_frame[const_dict['y_col']], num_output_samples),
                             name=const_dict['y_col'])
        return_t = pd.Series(np.linspace(0, df[const_dict['time_col']].iloc[-1], num_output_samples),
                             name=const_dict['time_col'])

        return_annot = round(pd.Series(signal.resample(interm_frame[const_dict['Annotations']], num_output_samples),
                                       name=const_dict['Annotations'])).astype(int)
        const_dict['f'] = f_target
        resampled_df = pd.concat([return_t, return_x, return_y, return_annot], axis=1)
        return resampled_df, const_dict

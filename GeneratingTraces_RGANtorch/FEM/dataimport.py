import itertools
import os
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

class TimeSeriesFEM(Dataset):
    def __init__(self, folderpath, transform=None, no_mean=False, vital_signs=2, slice_length=None, input_freq=1920, resample_freq=None):

        self.folderpath = folderpath
        self.input_freq = input_freq
        self.transform = transform
        self.vital_signs = vital_signs
        self.no_mean = no_mean
        self.slice_length = slice_length
        self.resample_freq = resample_freq
        self.seq_length = int(self.slice_length * (self.resample_freq or self.input_freq))
        self.data, self.labels = self.load_data()
        self.label_dist = self.get_label_dist()

    def resample(self, data):
        resampling_ratio = self.resample_freq/self.input_freq
        num_output_samples = int(len(data) * resampling_ratio)
        # Cubic Spline Interpolation für 'x'
        cubic_spline_x = CubicSpline(np.arange(len(data)), data['x'])
        return_x = pd.Series(cubic_spline_x(np.linspace(0, len(data) - 1, num_output_samples)), name='x')

        # Cubic Spline Interpolation für 'y'
        cubic_spline_y = CubicSpline(np.arange(len(data)), data['y'])
        return_y = pd.Series(cubic_spline_y(np.linspace(0, len(data) - 1, num_output_samples)), name='y')
        '''
        return_x = pd.Series(signal.resample(data['x'], num_output_samples),
                             name='x')
        return_y = pd.Series(signal.resample(data['y'], num_output_samples),
                             name='y')
        '''
        return_annot = round(pd.Series(signal.resample(data['flags'], num_output_samples),
                                       name='flags')).astype(int)
        resampled_df = pd.concat([return_x, return_y, return_annot], axis=1)
        return resampled_df
    def load_data(self):
        samples_x = []
        samples_y = []
        labels = []
        csv_files = [f for f in os.listdir(self.folderpath) if f.endswith('.csv') and 'reversed' not in f]
        #csv_files = [f for f in os.listdir(self.folderpath) if f.endswith('.csv')]

        for filename in csv_files:
            filepath = os.path.join(self.folderpath, filename)
            df = pd.read_csv(filepath)

            if self.resample_freq:
                df = self.resample(df)

            segments_x = self.slice_df(df[['x']], seq_length=self.seq_length)
            segments_y = self.slice_df(df[['y']], seq_length=self.seq_length)
            segment_labels = self.slice_df(df[['flags']], seq_length=self.seq_length)

            samples_x.extend(segments_x)
            samples_y.extend(segments_y)
            labels.extend(segment_labels)

        samples_x = np.array(samples_x)
        samples_y = np.array(samples_y)
        labels = np.array(labels)
        assert labels.shape == samples_x.shape and labels.shape == samples_y.shape, "Die Dimensionen sind nicht identisch"

        data = np.concatenate((samples_x, samples_y), axis=2)
        transformation_func = getattr(self, self.transform, None)

        if callable(transformation_func) and transformation_func is not None:
            data = transformation_func(data)
        #labels = np.concatenate((labels, labels), axis=2)

        return data, labels

    def slice_df(self, dataframe, seq_length):
        if self.slice_length is None:
            return dataframe.values
        samples_per_segment = seq_length
        segments = []
        start_idx = 0

        while start_idx + samples_per_segment <= len(dataframe):
            end_idx = start_idx + samples_per_segment
            segment = dataframe.iloc[start_idx:end_idx].values
            segments.append(segment)
            start_idx = end_idx

        return segments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def normalize(self, sample, scale_range=(-1, 1)):
        signal_length = sample.shape[1]
        num_signals = sample.shape[2]
        old_sample = sample.copy()
        if self.no_mean:
            data_mean = np.mean(sample, axis=(1,), keepdims=True)
            sample -= data_mean
        # Reshape
        sample_r = sample.reshape(-1, signal_length * num_signals)

        # Fit scaler
        scaler = MaxAbsScaler().fit(sample_r)

        # Scale
        normalized_sample = scaler.transform(sample_r).reshape(-1, signal_length, num_signals)

        return normalized_sample
    def get_label_dist(self):
        _, labels = self.load_data()
        labels = labels.squeeze().reshape(-1).tolist()
        zero_lengths = [sum(1 for _ in group) for key, group in itertools.groupby(labels) if key == 0]
        one_lengths = [sum(1 for _ in group) for key, group in itertools.groupby(labels) if key == 1]
        return [zero_lengths, one_lengths]


"""
GenerateDataset Class

This class provides functionality to generate synthetic datasets using a trained RCGAN or RGAN model.

Author: Fabian Anzlinger
Date: 04.01.2024

Usage:
    1. Initialize the GenerateDataset class with the path to the checkpoint file.
    2. Call the generate_data method to create synthetic datasets.

Example:
    generator = GenerateDataset(checkpoint_file='path/to/checkpoint.pth')
    synthetic_datasets = generator.generate_data(model, num_samples=10, duration=60, fsamp=1000, fsamp_out=100,
                                                 folderpath_to_save_to='output_folder', labels='path/to/labels.csv',
                                                 noise_scale=0.3, scaling_x=30, scaling_y=12.5)
"""


import inspect
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import make_interp_spline

from GeneratingTraces_RGANtorch.FEM.models.rcgan import RCGANGenerator
from GeneratingTraces_RGANtorch.FEM.models.rgan import RGANGenerator


class GenerateDataset():
    def __init__(self,
                 checkpoint_file):
        self.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        self.checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        self.generator = RGANGenerator if self.checkpoint[
                                              'GANtype'] == 'RGAN' else RCGANGenerator  # self.checkpoint['GANtype']
        self.GANargs = self.checkpoint['GANargs']
        self.model = self.generator(**self.GANargs)  # set model to evaluation-mode (not train-mode)
        self.model.load_state_dict(self.checkpoint['generator_state_dict'])
        self.model = self.model.to(
            self.device).eval()

    @staticmethod
    def rand_batch_labels(labels, z, fsamp):
        def sample_rand_labels():
            sequence = []
            for _ in range(seq_len):
                if len(sequence) < seq_len:
                    zero_length = np.random.choice(drift)
                    one_length = np.random.choice(ms)
                    sequence.extend([0] * zero_length)
                    sequence.extend([1] * one_length)
                else:
                    break
            return sequence[:seq_len]

        seq_len = z.shape[1]
        if type(labels) == str or labels.isinstance(Path):
            data = pd.read_csv(labels)
            drift = [int(i * fsamp / 1000) for i in
                     data['drift']]  # Adjust duration of drift segments according to fsamp
            ms = [int(i * fsamp / 1000) for i in data['ms']]
        rand_labels_batch = [sample_rand_labels() for _ in range(z.shape[0])]
        return torch.tensor(rand_labels_batch, dtype=torch.float32).view(z.shape[0], z.shape[1], -1)

    @staticmethod
    def random_tensor(shape):
        random_samples = torch.rand(shape)
        random_tensor = 2 * random_samples - 1
        return random_tensor

    @staticmethod
    def generate_data(model, num_samples, duration, fsamp, fsamp_out, folderpath_to_save_to=None, labels=None,
                      noise_scale=0.3, scaling_x=30, scaling_y=12.5):
        device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        noise_size = model.noise_size
        z = GenerateDataset.random_tensor((num_samples, duration * fsamp, noise_size)).to(device) * noise_scale
        if 'y' in list(inspect.signature(model.forward).parameters):  # If labels are needed
            y = GenerateDataset.rand_batch_labels(labels, z, fsamp).to(device)
            synthetic_data = model(z, y, reshape=False, reshape_y=True).detach().cpu()
        else:
            synthetic_data = model(z, reshape=False).detach().cpu()
        dfs = []
        for i in range(synthetic_data.shape[0]):
            x_vals = synthetic_data[i, :, 0].numpy() * scaling_x
            y_vals = synthetic_data[i, :, 1].numpy() * scaling_y
            t_vals = np.arange(synthetic_data.shape[1]) / fsamp
            # Create Dataframe
            if 'y' in list(inspect.signature(model.forward).parameters):
                df = pd.DataFrame({'t': t_vals, 'x': x_vals, 'y': y_vals, 'flags': y[i].squeeze().cpu().numpy()})
            else:
                df = pd.DataFrame({'t': t_vals, 'x': x_vals, 'y': y_vals})
            # resample mit B-Spline Interpolation
            if fsamp != fsamp_out:
                spline = make_interp_spline(df['t'], df[['x', 'y']], k=3)
                t_new = np.arange(0, df['t'].max(), 1 / fsamp_out)
                resampled_df = pd.DataFrame({'t': t_new, 'x': spline(t_new)[:, 0], 'y': spline(t_new)[:, 1]})
            else:
                resampled_df = df
            if folderpath_to_save_to:
                df.to_csv(os.path.join(folderpath_to_save_to, f'trace{i}.csv'), index=False)
            dfs.append(resampled_df)
        return dfs

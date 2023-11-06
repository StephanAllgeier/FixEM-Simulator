import numpy as np
import pandas as pd
import torch
from scipy.interpolate import make_interp_spline

from GeneratingTraces_RGANtorch.FEM.models import RCGANGenerator, RCGANDiscriminator, RGANGenerator, RGANDiscriminator

class GenerateDataset():
    def __init__(self,
                 checkpoint_file):
        self.device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        self.checkpoint = torch.load(checkpoint_file)
        self.generator = RGANGenerator if self.checkpoint['GANtype'] == 'RGAN' else RCGANGenerator#self.checkpoint['GANtype']
        self.GANargs = self.checkpoint['GANargs']
        self.model = self.generator(**self.GANargs).to(self.device).eval() #set model to evaluation-mode (not train-mode) TODO: OUTPUT HAT HIER IMMER SEQUENCE LENGTH 500

    @staticmethod
    def generate_data(model, num_samples, duration, fsamp, fsamp_out, folderpath_to_save_to=None):
        device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
        noise_size = model.input_size  # Annahme: input_size ist ein Attribut des Modells
        z = torch.randn(num_samples, duration * fsamp, noise_size).to(device)
        synthetic_data = model(z, reshape=False).detach().cpu()
        dfs = []
        for i in range(synthetic_data.shape[0]):
            x_vals = synthetic_data[i, :, 0].numpy()
            y_vals = synthetic_data[i, :, 1].numpy()
            t_vals = np.arange(synthetic_data.shape[1]) / fsamp
            # Erstelle einen DataFrame für die aktuelle Nummer
            df = pd.DataFrame({'t': t_vals, 'x': x_vals, 'y': y_vals})
            # resample mit B-Spline Interpolation
            if fsamp != fsamp_out:
                spline = make_interp_spline(df['t'], df[['x', 'y']], k=3)
                t_new = np.arange(0, df['t'].max(), 1 / fsamp_out)
                resampled_df = pd.DataFrame({'t': t_new, 'x': spline(t_new)[:, 0], 'y': spline(t_new)[:, 1]})
            else:
                resampled_df = df
            if folderpath_to_save_to:
                df.to_csv(f'{folderpath_to_save_to}/trace{i}')
            dfs.append(resampled_df)
        return dfs
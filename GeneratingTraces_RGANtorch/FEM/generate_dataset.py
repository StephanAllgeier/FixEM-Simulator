import torch
from GeneratingTraces_RGANtorch.FEM.models import RCGANGenerator, RCGANDiscriminator, RGANGenerator, RGANDiscriminator

class GenerateDataset():
    def __init__(self,
                 checkpoint_file,
                 duration,
                 sampling_frequency):
        self.checkpoint = torch.load(checkpoint_file)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.GAN =1
        self.lr =1
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
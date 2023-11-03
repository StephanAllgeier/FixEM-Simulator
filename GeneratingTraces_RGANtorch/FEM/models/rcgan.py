import os

import torch
import torch.nn as nn
from GeneratingTraces_RGANtorch.FEM.models import RGANGenerator, RGANDiscriminator
from GeneratingTraces_RGANtorch.FEM.utils import tile


class RCGANGenerator(RGANGenerator):
    def __init__(self,
                 sequence_length,
                 output_size,
                 num_classes,
                 noise_size,
                 prob_classes=None,
                 label_embedding_size=None,
                 **kwargs):
        """Recursive Conditional GAN (Generator) implementation with RNN cells.

        Notes:
            This model adds upon ward2icu.models.rgan. See docs of parent classes
            for more information.

        Args (check parent class for extra arguments):
            sequence_length (int): Number of points in the sequence.
                Defined by the real dataset.
            output_size (int): Size of output (usually the last tensor dimension).
                Defined by the real dataset.
            num_classes (int): Number of classes in the dataset.
            noise_size (int): Size of noise used to generate fake data.
            label_embedding_size (int, optional): Size of embedding dimensions.
                Defaults to num_classes.
        """
        # Defaults
        label_embedding_size = label_embedding_size or num_classes
        if prob_classes is None:
            prob_classes  = torch.ones(num_classes)

        #TODO(dsevero): this could cause problems
        kwargs['input_size'] = label_embedding_size + noise_size
        kwargs['noise_size'] = noise_size
        kwargs['label_type'] = "generated"
        kwargs['sequence_length'] = sequence_length
        kwargs['output_size'] = output_size

        super(RCGANGenerator, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.label_embeddings = nn.Embedding(num_classes, label_embedding_size)
        self.label_embedding_size = label_embedding_size 
        self.prob_classes = torch.Tensor(prob_classes)

        # Initialize all weights.
        # Already initialized in parent class
    #TODO: DIE FUNKTION UNTEN HAT FUNKTIONIERT. Jetzt wird getestet...
    '''
    def forward(self, z, y):
        # y must be tiled so that labels repeat across the sequence dimensions
        # y_tiled[:, i] == y_tiled[:, j] for all i and j
        # shape: (batch_size, sequence_length)
        #TODO: LABEL TENSOR GENERIEREN UND MIT ZURÜCK GEBEN? Wie lernt der Generator dann, wenn die labels "Müll" sind?
        #y_test_gen = torch.randint(0, 2, (z.shape[0], z.shape[1]), dtype=torch.long).to(y.device)
        y_tiled = y.squeeze().type(torch.LongTensor).to(y.device)
        # shape: (batch-size, sequence_length, label_embedding_size)

        # y_emb = self.label_embeddings(y_tiled.type(torch.LongTensor).to(y.device))
        y_emb = self.label_embeddings(y_tiled)

        # shape: (batch-size, sequence_length, noise_size)
        z = z.view(-1, self.sequence_length, self.noise_size)

        # shape: (batch-size, encoding_dims)
        z_cond = torch.cat((z, y_emb), dim=2)
        # shape: (batch-size, sequence_length, output_size)
        return super(RCGANGenerator, self).forward(z_cond, reshape=False)
    '''

    def forward(self, z):
        # shape: (batch-size, sequence_length, noise_size)
        z = z.view(-1, self.sequence_length, self.noise_size)

        # shape: (batch-size, encoding_dims)
        output_generator = super(RCGANGenerator, self).forward(z, reshape=False)
        labels_generator = torch.randint(0, 2, (z.shape[0], z.shape[1]), dtype=torch.long).to(z.device)
        # shape: (batch-size, sequence_length, output_size)
        return output_generator, labels_generator

    def sampler(self, sample_size, device='cuda'):
        return [
        ]


class RCGANDiscriminator(RGANDiscriminator):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 input_size,
                 label_embedding_size=None,
                 **kwargs):
        """Recursive Conditional GAN (Discriminator) implementation with RNN cells.


        Args (check parent class for extra arguments):
            sequence_length (int): Number of points in the sequence.
            num_classes (int): Number of classes in the dataset.
            input_size (int): Size of input (usually the last tensor dimension).
            label_embedding_size (int, optional): Size of embedding dimensions.
                Defaults to num_classes.
        """

        # Defaults
        label_embedding_size = label_embedding_size or num_classes
        kwargs['input_size'] = label_embedding_size + input_size
        kwargs['label_type'] = "required"
        kwargs['sequence_length'] = sequence_length

        super(RCGANDiscriminator, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.label_embeddings = nn.Embedding(num_classes, label_embedding_size)
        self.label_embedding_size = label_embedding_size 

        # Initialize all weights.
        self._weight_initializer()

    def forward(self, x, y):
        # y must be tiled so that labels repeat across the sequence dimensions
        # y_tiled[:, i] == y_tiled[:, j] for all i and j
        # shape: (batch_size, sequence_length)
        #y_test = y.expand(-1,-1,2)
        y_tiled = y.squeeze().type(torch.LongTensor).to(y.device)
        # shape: (batch-size, sequence_length, label_embedding_size)

        #y_emb = self.label_embeddings(y_tiled.type(torch.LongTensor).to(y.device))
        y_emb = self.label_embeddings(y_tiled)

        #TODO: y_emb kann hier berechnet werden, ruft aber einen Error hervor, wenn ich versuche ihn anzuschauen, WIESO?!!!!!!!

        # shape: (batch-size, sequence_length, label_embedding_size + hidden_size)
        x_cond = torch.cat((x, y_emb), dim=2)

        return super(RCGANDiscriminator, self).forward(x_cond)

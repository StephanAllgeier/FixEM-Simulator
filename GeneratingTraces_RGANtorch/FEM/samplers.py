import numpy as np
import torch

from GeneratingTraces_RGANtorch.FEM.utils import tile as tile_func


class SimpleSampler:
    def __init__(self, X, y, label_dist=None, tile=False, batch_size=None, shuffle=True):
        self.X = X
        self.y = y
        self.tile = tile
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.reset_indices()
        self.sequence_length = X.shape[1]
        self.label_dist = label_dist

    def reset_indices(self):
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def sample(self):
        X_batch = self.X[:self.batch_size] if self.batch_size else self.X
        y_batch = self.y[:self.batch_size] if self.batch_size else self.y

        y_batch = tile_func(y_batch, self.sequence_length) if self.tile else y_batch
        return X_batch, y_batch

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        self.index = 0
        self.reset_indices()  # Reset der Indizes zu Beginn jeder Iteration
        return self

    def __next__(self):
        if self.index < len(self.X):
            start = self.index
            end = min(self.index + self.batch_size, len(self.X))
            batch_indices = self.indices[start:end]  # Verwende gemischte Indizes

            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]

            self.index += self.batch_size
            y_batch = tile_func(y_batch, self.sequence_length) if self.tile else y_batch
            return X_batch, y_batch
        else:
            # Wenn der Index das Ende erreicht hat, beenden Sie die Iteration
            self.index = 0
            if self.shuffle:
                self.reset_indices()  # Mische die Indizes am Ende jeder Epoche
            raise StopIteration

    def sample_rand_labels(self):
        assert self.label_dist is not None, print('Error')
        sequence = []
        for _ in range(self.sequence_length):
            if len(sequence) < self.sequence_length:
                zero_length = np.random.choice(self.label_dist[0])
                one_length = np.random.choice(self.label_dist[1])
                sequence.extend([0] * zero_length)
                sequence.extend([1] * one_length)
            else:
                break
        return sequence[:self.sequence_length]

    def rand_batch_labels(self, num_seq):
        rand_labels_batch = [self.sample_rand_labels() for _ in range(num_seq)]
        return torch.tensor(rand_labels_batch, dtype=torch.float32)

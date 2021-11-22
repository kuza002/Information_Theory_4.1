import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TitanicDB(Dataset):
    def __init__(self, path):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

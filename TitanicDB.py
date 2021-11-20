import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TitanicDB(Dataset):
    def __init__(self, path):
        self.labels = pd.read_csv(path)[["0"]].to_numpy()
        self.labels = np.array([i[0] for i in self.labels])

        self.parameters = pd.read_csv(path).drop(['0'], axis=1).to_numpy()
        self.parameters = torch.tensor(self.parameters)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.parameters[idx], self.labels[idx]

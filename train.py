import torch
from torch.utils.data import DataLoader

from TitanicDB import TitanicDB

training_data = TitanicDB("dataset/training_set.csv")

# If shuffle=True, after we iterate over all batches the data is shuffled
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))







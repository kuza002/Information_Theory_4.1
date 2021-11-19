import torch
from torch.utils.data import DataLoader

from TitanicDB import TitanicDB
from model import NeuralNetwork

# Include cuda if is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Include dataset
training_data = TitanicDB("dataset/training_set.csv")

# If shuffle=True, after we iterate over all batches the data is shuffled
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Create model

model = NeuralNetwork().to(device)
print(model)



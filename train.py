import torch
from torch import nn
from torch.utils.data import DataLoader

from TitanicDB import TitanicDB
from model import NeuralNetwork


def train_loop(dataloader, model, loss_fn, optimizer):

    for batch, (X, y) in enumerate(dataloader):
        size = len(dataloader.dataset)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(X, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if batch % 50 == 0:
        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(50*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# Define batch size, epoch number, loss function and learning rate
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
batch_size = 64

# Include cuda if is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Include dataset
training_data = TitanicDB("dataset/training_set.csv")
test_data = TitanicDB('dataset/test_set.csv')

# If shuffle=True, after we iterate over all batches the data is shuffled
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Create model
model = NeuralNetwork().to(device)

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# train
# epochs = 10
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")


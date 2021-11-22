import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TitanicDB import TitanicDB
from logisticregression import LogisticRegression

train_dataset = TitanicDB('dataset/training_set.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=10,
                          shuffle=True,)

model = LogisticRegression(6)

# 2) Loss and optimizer
num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(model(torch.tensor([3.0,0.0,22.0,1.0,0.0,7.25])))

# 3) Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Forward pass and loss
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        # if (epoch + 1) % 10 == 0:
        #     print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

print(model(torch.tensor([3.0,0.0,22.0,1.0,0.0,7.25])))
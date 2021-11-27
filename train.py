import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from TitanicDB import TitanicDB
from logisticregression import LogisticRegression


def training_loop(model, data_loader, optimizer):
    for inputs, labels in data_loader:
        # Forward pass and loss
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')


def test(model, train_dataset):
    correct_answers = 0
    with torch.no_grad():
        for inputs, labels in train_dataset:
            y_pred = model(inputs)
            y_pred = round(float(y_pred))
            if y_pred == labels:
                correct_answers += 1
    return correct_answers / len(train_dataset)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_dataset = TitanicDB('dataset/training_set.csv', device)
test_dataset = TitanicDB('dataset/test_set.csv', device)
validation_dataset = TitanicDB('dataset/validation_set.csv', device)

model = LogisticRegression(6).to(device)

# 2) Loss and optimizer
num_epochs = 100
criterion = nn.BCELoss()


print("Accuracy:", test(model, test_dataset))

# 3) Training loop

max_accuracy = 0.0
optimal_batch_size = 0
optimal_learning_rate = 0
optimal_model = None


for batch_size in range(1, len(validation_dataset)):

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    for learning_rate in range(5):
        optimizer = torch.optim.SGD(model.parameters(), lr=10 ** -learning_rate)

        for epoch in range(num_epochs):
            training_loop(model, train_loader, optimizer)

        accuracy = test(model, validation_dataset)
        print("With batch size =", batch_size, 'and lr =', learning_rate)
        print("Accuracy", accuracy, ' >', max_accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_learning_rate = 10**-learning_rate
            optimal_batch_size = batch_size
            optimal_model = model
            print("\nWith batch size =", optimal_batch_size, 'and lr =', optimal_learning_rate)
            print("Max accuracy:", max_accuracy)

        model = LogisticRegression(6).to(device)

# 4) Test loop

print("With batch size =", optimal_batch_size, 'and lr =', optimal_learning_rate)
print("Accuracy:", max_accuracy)
print(test(optimal_model, test_dataset))

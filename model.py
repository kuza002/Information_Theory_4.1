from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_regression = nn.Sequential(
            nn.Linear(6, 2),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 10),
        )

    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        return prediction

import torch
import torch.nn as nn


class Naive_NN_for_demand(nn.Module):

    def __init__(self, input_dim):
        super(Naive_NN_for_demand, self).__init__()

        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn


class MLP_for_demand(nn.Module):
    
    def __init__(self, input_dim):
        super(MLP_for_demand, self).__init__()

        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class cnn_dsprite(nn.Module):
    def __init__(self, input_dim):
        super(cnn_dsprite, self).__init__()

        self.conv1 = nn.Conv2d()
        self.maxpool1 = nn.MaxPool2d()
        self.conv2 = nn.Conv2d()
        self.maxpool2 = nn.MaxPool2d()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)

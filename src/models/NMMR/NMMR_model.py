import torch
import torch.nn as nn


class MLP_for_demand(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super(MLP_for_demand, self).__init__()

        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == (self.network_depth + 1):  # if last layer, don't apply relu activation
                x = layer(x)
            x = torch.relu(layer(x))

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

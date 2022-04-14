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


class cnn_for_dsprite(nn.Module):
    def __init__(self, train_params):
        super(cnn_for_dsprite, self).__init__()
        self.train_params = train_params
        self.batch_size = train_params["batch_size"]
        self.conv1a = nn.Conv2d(1, 6, 5)
        self.conv2a = nn.Conv2d(6, 16, 5)
        self.conv1w = nn.Conv2d(1, 6, 5)
        self.conv2w = nn.Conv2d(6, 16, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2704, 256)  # TODO: derive a formula for 2704
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, A, W):
        A = self.maxpool(torch.relu(self.conv1a(A)))
        W = self.maxpool(torch.relu(self.conv1w(W)))
        A = self.maxpool(torch.relu(self.conv2a(A)))
        W = self.maxpool(torch.relu(self.conv2w(W)))
        A = torch.flatten(A, 1)  # flatten all dimensions except batch
        W = torch.flatten(W, 1)
        x = torch.add(A, W)  # TODO: could try torch.cat() here instead
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

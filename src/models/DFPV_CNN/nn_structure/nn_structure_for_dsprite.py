from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F


def build_net_for_dsprite() -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    treatment_1st_net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Flatten(),
                                      nn.Linear(128*16*16, 32))

    treatment_2nd_net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Flatten(),
                                      nn.Linear(128*16*16, 32))

    outcome_proxy_net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Flatten(),
                                      nn.Linear(128*16*16, 32))

    treatment_proxy_net = nn.Sequential(nn.Linear(3, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 32),
                                        nn.ReLU())

    return treatment_1st_net, treatment_2nd_net, treatment_proxy_net, outcome_proxy_net, None, None

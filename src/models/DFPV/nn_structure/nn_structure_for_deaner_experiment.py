from typing import Optional, Tuple
import torch
from torch import nn


def build_net_for_deaner()-> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    treatment_1st_net = nn.Sequential(nn.Linear(1, 32),
                                  nn.ReLU(),
                                  nn.Linear(32, 5),
                                  nn.ReLU())

    treatment_2nd_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 6),
                                      nn.ReLU())

    treatment_proxy_net = nn.Sequential(nn.Linear(4, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 7),
                                        nn.ReLU())

    outcome_proxy_net = nn.Sequential(nn.Linear(3, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 4),
                                      nn.ReLU())

    return treatment_1st_net, treatment_2nd_net, treatment_proxy_net, outcome_proxy_net, None, None

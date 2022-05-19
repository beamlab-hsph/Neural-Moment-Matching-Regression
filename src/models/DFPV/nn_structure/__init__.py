from typing import Tuple, Optional

from torch import nn

from .nn_structure_for_demand import build_net_for_demand
from .nn_structure_for_dsprite import build_net_for_dsprite


def build_extractor(data_name: str) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    if data_name.startswith("demand"):
        return build_net_for_demand()
    elif data_name == "dsprite":
        return build_net_for_dsprite()
    else:
        raise ValueError(f"data name {data_name} is not valid")

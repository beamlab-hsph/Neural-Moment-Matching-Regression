from .nn_structure_for_demand import DemandDistribution
from .nn_structure_for_dsprite import DspriteDistribution


def build_extractor(data_name: str, hidden_dim, n_sample):
    if data_name.startswith("demand"):
        return DemandDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    elif data_name == "dsprite":
        return DspriteDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    else:
        raise ValueError(f"data name {data_name} is not valid")

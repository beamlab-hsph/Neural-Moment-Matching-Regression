import numpy as np
import torch


def squared_distance(x, y, is_torch=True):
    if y is None:
        y = x
    if is_torch:
        diffs = x-y
        sqdist = torch.sum(diffs**2)
    else:
        diffs = x-y
        sqdist = np.sum(diffs**2)
        del diffs
    return sqdist


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, length_scale=1):
    return torch.exp(-1. * torch.sum((x - y) ** 2, axis=0) / (2 * (length_scale ** 2)))


def calculate_kernel_matrix(dataset, kernel=rbf_kernel, **kwargs):

    tensor = dataset.permute(1, 0)
    tensor1 = tensor.unsqueeze(dim=2)
    tensor2 = tensor.unsqueeze(dim=1)

    return kernel(tensor1, tensor2, **kwargs)

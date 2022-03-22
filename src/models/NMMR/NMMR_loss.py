import torch


def NMMR_loss(model_output, target, kernel_matrix, batch_indices=None):
    residual = target - model_output
    n = residual.shape[0]
    if batch_indices is None:
        K = kernel_matrix
    else:
        K = kernel_matrix[batch_indices[:, None], batch_indices]
    # calculate V statistic (see Serfling 1980)
    loss = residual.T @ K @ residual / n**2
    return loss[0, 0]

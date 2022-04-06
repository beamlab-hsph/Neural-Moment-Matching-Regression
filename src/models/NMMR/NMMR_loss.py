import torch


def NMMR_loss(model_output, target, kernel_matrix, loss_name: str, batch_indices=None):
    residual = target - model_output
    n = residual.shape[0]
    if batch_indices is None:
        K = kernel_matrix
    else:
        K = kernel_matrix[batch_indices[:, None], batch_indices]

    if loss_name == "U_statistic":
        # calculate U statistic (see Serfling 1980)
        K.fill_diagonal_(0)
        loss = residual.T @ K @ residual / (n * (n-1))
    elif loss_name == "V_statistic":
        # calculate V statistic (see Serfling 1980)
        loss = residual.T @ K @ residual / n ** 2
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")

    return loss[0, 0]

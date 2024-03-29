import torch
from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch


def make_AW2_test(test_data_t: PVTestDataSetTorch,
                  val_data_t: PVTrainDataSetTorch) -> torch.Tensor:
    """
    Creates a 3-dim Tensor with shape (intervention_array_len, n_samples, 5)
    This will contain the test values for do(A) chosen by Xu et al. as well as len(val_data_t) random draws for W
    arranged as a concatenation of A, A^2, W, W^2, A*W.

    Parameters
    ----------
    test_data_t: expected to have a 'treatment' vector.
    val_data_t: expected to have 'outcome_proxy' vectors.

    Returns
    -------
    AW2_test: 3-dim torch.Tensor with shape (intervention_array_len, n_samples, 5)

    """
    # get array sizes
    intervention_array_len = test_data_t.treatment.shape[0]
    num_val_samples = val_data_t.treatment.shape[0]

    # tile A, and W to create a tensor with shape (intervention_array_len, n_samples, 2)
    temp_A = test_data_t.treatment.expand(-1, num_val_samples)
    temp_W = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
    AW2_test = torch.stack((temp_A, temp_A ** 2, temp_W.T, temp_W.T ** 2, temp_A * temp_W.T), dim=-1)

    return AW2_test

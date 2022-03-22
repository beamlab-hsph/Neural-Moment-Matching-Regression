from typing import Optional, Dict, Any
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch, split_train_data, PVTrainDataSet, \
    PVTestDataSet
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.models.NMMR.NMMR_loss import NMMR_loss
from src.models.NMMR.NMMR_model import MLP_for_demand
from sklearn.model_selection import train_test_split

from src.models.NMMR.utils import squared_distance, rbf_kernel, calculate_kernel_matrix


class NMMR_Trainer(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], dump_folder: Optional[Path] = None):

        self.data_config = data_configs
        self.n_sample = self.data_config['n_sample']
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = train_params['gpu_flg'] == "True"

        # TODO: send model layers to GPU/CUDA, change model init to accept them (see DFPV trainer)

    def train(self, train_t: PVTrainDataSetTorch, kernel_matrix, verbose: int = 0) -> MLP_for_demand:

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            kernel_matrix = kernel_matrix.cuda()

        # inputs consist of (A, W) tuples
        model = MLP_for_demand(input_dim=2).cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=3e-4, weight_decay=3e-6)

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A, batch_W, batch_y = train_t.treatment[indices], train_t.outcome_proxy[indices], \
                                            train_t.outcome[indices]

                batch_x = torch.cat((batch_A, batch_W), dim=1)

                # training loop
                optimizer.zero_grad()
                pred_y = model(batch_x)
                loss = NMMR_loss(pred_y, batch_y, kernel_matrix, indices)
                loss.backward()
                optimizer.step()

        return model


def NMMR_demand_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                           one_mdl_dump_dir: Path,
                           random_seed: int = 42, verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data_ate(data_config=data_config)

    # preprocess data
    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)

    # precompute the kernel matrix
    kernel_inputs_train = torch.cat((train_t.treatment, train_t.treatment_proxy), dim=1)
    kernel_matrix_train = calculate_kernel_matrix(kernel_inputs_train)

    # train model
    trainer = NMMR_Trainer(data_config, model_param)
    model = trainer.train(train_t, kernel_matrix_train, verbose)

    # prepare test data on the gpu
    if trainer.gpu_flg:
        # torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()

    # Create a distribution of W's for the model to predict on
    # TODO: switch these train W's to val/test W's (?)
    intervention_array_len = test_data_t.treatment.shape[0]
    train_data_len = train_t.outcome_proxy.shape[0]
    model_inputs_val_doA = torch.zeros([intervention_array_len, train_data_len, 2], dtype=torch.float32).cuda()

    # Set one axis to a constant W_train distribution, vary other axis across intervention values
    for i in range(intervention_array_len):
        model_inputs_val_doA[i, :, 0] = test_data_t.treatment[i]
        model_inputs_val_doA[i, :, 1] = train_t.outcome_proxy.squeeze()

    E_w_haw = torch.Tensor([torch.mean(model(model_inputs_val_doA[i])) for i in range(intervention_array_len)])
    pred = preprocessor.postprocess_for_prediction(E_w_haw).numpy()
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        # print(f"{oos_loss=}")
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return oos_loss

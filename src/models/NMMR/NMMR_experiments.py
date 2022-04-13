import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.models.NMMR.NMMR_loss import NMMR_loss
from src.models.NMMR.NMMR_model import MLP_for_demand
from src.models.NMMR.utils import rbf_kernel, calculate_kernel_matrix


class NMMR_Trainer(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int,
                 dump_folder: Optional[Path] = None):

        self.data_config = data_configs
        self.train_params = train_params
        self.n_sample = self.data_config['n_sample']
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available()
        self.log_metrics = train_params['log_metrics'] == "True"
        self.l2_penalty = train_params['l2_penalty']
        self.learning_rate = train_params['learning_rate']
        self.loss_name = train_params['loss_name']

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):

        return calculate_kernel_matrix(kernel_inputs)

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> MLP_for_demand:

        # inputs consist of (A, W) tuples
        model = MLP_for_demand(input_dim=2, train_params=self.train_params)

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A, batch_W, batch_y = train_t.treatment[indices], train_t.outcome_proxy[indices], \
                                            train_t.outcome[indices]

                batch_x = torch.cat((batch_A, batch_W), dim=1)

                kernel_inputs_train = torch.cat((train_t.treatment[indices], train_t.treatment_proxy[indices]), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                # training loop
                optimizer.zero_grad()
                pred_y = model(batch_x)
                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train, self.loss_name)  # indices)
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy), dim=1))
                    preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy), dim=1))

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # compute the full kernel matrix
                    kernel_inputs_train = torch.cat((train_t.treatment, train_t.treatment_proxy), dim=1)
                    kernel_inputs_val = torch.cat((val_t.treatment, val_t.treatment_proxy), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(preds_train, train_t.outcome, kernel_matrix_train, self.loss_name)
                    causal_loss_val = NMMR_loss(preds_val, val_t.outcome, kernel_matrix_val, self.loss_name)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model


def NMMR_demand_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                           one_mdl_dump_dir: Path,
                           random_seed: int = 42, verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data_org = generate_test_data_ate(data_config=data_config)

    # preprocess data
    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    val_data = preprocessor.preprocess_for_train(val_data_org)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    # train model
    trainer = NMMR_Trainer(data_config, model_param, random_seed, one_mdl_dump_dir)
    model = trainer.train(train_t, val_data_t, verbose)

    # prepare test data on the gpu
    if trainer.gpu_flg:
        # torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()
        val_data_t = val_data_t.to_gpu()

    # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
    # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
    intervention_array_len = test_data_t.treatment.shape[0]
    train_n_sample = data_config['n_sample']
    temp1 = test_data_t.treatment.expand(-1, train_n_sample)
    temp2 = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
    model_inputs_test = torch.stack((temp1, temp2.T), dim=-1)

    # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
    # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values
    E_w_haw = torch.mean(model(model_inputs_test), dim=1).cpu()
    pred = preprocessor.postprocess_for_prediction(E_w_haw).detach().numpy()
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        np.testing.assert_array_equal(pred.shape, test_data_org.structural.shape)
        oos_loss = np.mean((pred - test_data_org.structural) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))

    if trainer.log_metrics:
        return oos_loss, pd.DataFrame(
            data={'causal_loss_train': torch.Tensor(trainer.causal_train_losses[-50:], device="cpu").numpy(),
                  'causal_loss_val': torch.Tensor(trainer.causal_val_losses[-50:], device="cpu").numpy()})
    else:
        return oos_loss


if __name__ == "__main__":
    data_config = {"name": "demand", "n_sample": 5000}
    model_param = {"name": "nmmr",
                   "n_epochs": 50,
                   "batch_size": 1000,
                   "log_metrics": "True",
                   "l2_penalty": 0.003,
                   "learning_rate": 3e-6,
                   "loss_name": "V_statistic",
                   "network_width": 10,
                   "network_depth": 5}

    one_mdl_dump_dir = Path(
        op.join("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/dumps", "temp_new"))
    NMMR_demand_experiment(data_config, model_param, one_mdl_dump_dir, random_seed=41, verbose=0)

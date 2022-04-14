import os.path as op
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.models.NMMR.NMMR_loss import NMMR_loss
from src.models.NMMR.NMMR_model import MLP_for_demand, cnn_for_dsprite
from src.models.NMMR.kernel_utils import calculate_kernel_matrix


class NMMR_Trainer_DemandExperiment(object):
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

                optimizer.zero_grad()
                batch_x = torch.cat((batch_A, batch_W), dim=1)
                pred_y = model(batch_x)
                kernel_inputs_train = torch.cat((train_t.treatment[indices], train_t.treatment_proxy[indices]), dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train, self.loss_name)
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy), dim=1))
                    preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy), dim=1))

                    # compute the full kernel matrix
                    kernel_inputs_train = torch.cat((train_t.treatment, train_t.treatment_proxy), dim=1)
                    kernel_inputs_val = torch.cat((val_t.treatment, val_t.treatment_proxy), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(preds_train, train_t.outcome, kernel_matrix_train, self.loss_name)
                    causal_loss_val = NMMR_loss(preds_val, val_t.outcome, kernel_matrix_val, self.loss_name)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, n_sample: int):
        # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
        # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
        intervention_array_len = test_data_t.treatment.shape[0]
        temp1 = test_data_t.treatment.expand(-1, n_sample)
        temp2 = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
        model_inputs_test = torch.stack((temp1, temp2.T), dim=-1)

        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values
        E_w_haw = torch.mean(model(model_inputs_test), dim=1)

        return E_w_haw.cpu()


class NMMR_Trainer_dSpriteExperiment(object):
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
        self.A_scale = 0.05  # TODO: tune this value? Looked pretty good as is

        self.mse_loss = nn.MSELoss()

        if self.log_metrics:
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.causal_train_losses = []
            self.causal_val_losses = []

    def compute_kernel(self, kernel_inputs):

        return calculate_kernel_matrix(kernel_inputs)

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> cnn_for_dsprite:

        # inputs consist of (A, W) tuples
        model = cnn_for_dsprite(train_params=self.train_params)

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

                optimizer.zero_grad()
                pred_y = model(batch_A.reshape(-1, 1, 64, 64), batch_W.reshape(-1, 1, 64, 64))
                kernel_inputs_train = torch.cat(
                    (self.A_scale * train_t.treatment[indices], train_t.treatment_proxy[indices]),
                    dim=1)
                kernel_matrix_train = self.compute_kernel(kernel_inputs_train)

                causal_loss_train = NMMR_loss(pred_y, batch_y, kernel_matrix_train, self.loss_name)
                causal_loss_train.backward()
                optimizer.step()

            # at the end of each epoch, log metrics
            if self.log_metrics:
                with torch.no_grad():
                    preds_train = model(train_t.treatment.reshape(-1, 1, 64, 64), train_t.outcome_proxy.reshape(-1, 1, 64, 64))
                    preds_val = model(val_t.treatment.reshape(-1, 1, 64, 64), val_t.outcome_proxy.reshape(-1, 1, 64, 64))
                    kernel_inputs_train = torch.cat((self.A_scale * train_t.treatment, train_t.treatment_proxy), dim=1)
                    kernel_inputs_val = torch.cat((self.A_scale * val_t.treatment, val_t.treatment_proxy), dim=1)
                    kernel_matrix_train = self.compute_kernel(kernel_inputs_train)
                    kernel_matrix_val = self.compute_kernel(kernel_inputs_val)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = self.mse_loss(preds_train, train_t.outcome)
                    mse_val = self.mse_loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)

                    # calculate and log the causal loss (train & validation)
                    causal_loss_train = NMMR_loss(preds_train, train_t.outcome, kernel_matrix_train, self.loss_name)
                    causal_loss_val = NMMR_loss(preds_val, val_t.outcome, kernel_matrix_val, self.loss_name)
                    self.writer.add_scalar(f'{self.loss_name}/train', causal_loss_train, epoch)
                    self.writer.add_scalar(f'{self.loss_name}/val', causal_loss_val, epoch)
                    self.causal_train_losses.append(causal_loss_train)
                    self.causal_val_losses.append(causal_loss_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, num_W_test: int):

        # Create a 3-dim array with shape [intervention_array_len, n_samples, 2]
        # This will contain the test values for do(A) chosen by Xu et al. as well as {n_samples} random draws for W
        intervention_array_len = test_data_t.treatment.shape[0]

        # create n_sample copies of each test image (A), and 588 copies of each proxy image (W)
        test_A = test_data_t.treatment.repeat_interleave(num_W_test, dim=0)
        test_W = val_data_t.outcome_proxy.repeat_interleave(intervention_array_len, dim=0)

        # reshape test and proxy image to 1 x 64 x 64 (so that the model's conv2d layer is happy)
        test_A = test_A.reshape(-1, 1, 64, 64)
        test_W = test_W.reshape(-1, 1, 64, 64)

        # test_A = test_data_t.treatment.reshape(-1, 1, 1, 64, 64).expand(-1, n_sample, -1, -1, -1)
        # test_W = val_data_t.outcome_proxy.reshape(1, -1, 1, 64, 64).expand(intervention_array_len, -1, -1, -1, -1)
        mean = torch.nn.AvgPool1d(kernel_size=num_W_test, stride=num_W_test)
        E_w_haw = mean(model(test_A, test_W).unsqueeze(-1).T)

        # temp1 = test_data_t.treatment.expand(-1, n_sample)
        # temp2 = val_data_t.outcome_proxy.expand(-1, intervention_array_len)
        # model_inputs_test = torch.stack((temp1, temp2.T), dim=-1)

        # Compute model's predicted E[Y | do(A)] = E_w[h(a, w)]
        # Note: the mean is taken over the n_sample axis, so we obtain {intervention_array_len} number of expected values
        # E_w_haw = torch.mean(model(model_inputs_test), dim=1).cpu()

        return E_w_haw.T.squeeze(1).cpu()

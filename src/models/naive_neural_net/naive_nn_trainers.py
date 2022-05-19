from typing import Optional, Dict, Any
from pathlib import Path
import os.path as op
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.models.naive_neural_net.naive_nn_model import Naive_NN_for_demand, Naive_NN_for_dsprite_AY, Naive_NN_for_dsprite_AWZY
from src.utils.make_AWZ_test import make_AWZ_test


class Naive_NN_Trainer_DemandExperiment(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int,
                 dump_folder: Optional[Path] = None):

        self.data_config = data_configs
        self.train_params = train_params
        self.n_sample = self.data_config['n_sample']
        self.model_name = self.train_params['name']
        self.n_epochs = self.train_params['n_epochs']
        self.batch_size = self.train_params['batch_size']
        self.l2_penalty = self.train_params['l2_penalty']
        self.learning_rate = self.train_params['learning_rate']
        self.gpu_flg = torch.cuda.is_available()
        self.log_metrics = self.train_params.get('log_metrics', False)

        if self.log_metrics and (dump_folder is not None):
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.train_losses = []
            self.val_losses = []

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0) -> Naive_NN_for_demand:

        if self.model_name == "naive_neural_net_AY":
            # inputs consist of only A
            model = Naive_NN_for_demand(input_dim=1, train_params=self.train_params)
        elif self.model_name == "naive_neural_net_AWZY":
            # inputs consist of A, W, and Z (and Z is 2-dimensional)
            model = Naive_NN_for_demand(input_dim=4, train_params=self.train_params)
        else:
            raise ValueError(f"name {self.model_name} is not known")

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)
        loss = nn.MSELoss()

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_y = train_t.outcome[indices]

                if self.model_name == "naive_neural_net_AY":
                    batch_inputs = train_t.treatment[indices]

                if self.model_name == "naive_neural_net_AWZY":
                    batch_inputs = torch.cat((train_t.treatment[indices], train_t.outcome_proxy[indices],
                                              train_t.treatment_proxy[indices]), dim=1)

                # training loop
                optimizer.zero_grad()
                pred_y = model(batch_inputs)
                output = loss(pred_y, batch_y)
                output.backward()
                optimizer.step()

            if self.log_metrics:
                with torch.no_grad():
                    if self.model_name == "naive_neural_net_AY":
                        preds_train = model(train_t.treatment)
                        preds_val = model(val_t.treatment)
                    elif self.model_name == "naive_neural_net_AWZY":
                        preds_train = model(torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.treatment_proxy), dim=1))
                        preds_val = model(torch.cat((val_t.treatment, val_t.outcome_proxy, val_t.treatment_proxy), dim=1))

                        # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = loss(preds_train, train_t.outcome)
                    mse_val = loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)
                    self.train_losses.append(mse_train)
                    self.val_losses.append(mse_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch):
        if model.train_params['name'] == "naive_neural_net_AY":
            pred = model(test_data_t.treatment).cpu().detach().numpy()

        elif model.train_params['name'] == "naive_neural_net_AWZY":
            AWZ_test = make_AWZ_test(test_data_t, val_data_t)
            pred = torch.mean(model(AWZ_test), dim=1).cpu().detach().numpy()

        return pred

class Naive_NN_Trainer_dSpriteExperiment(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], random_seed: int,
                 dump_folder: Optional[Path] = None):

        self.data_config = data_configs
        self.train_params = train_params
        self.data_name = data_configs.get("name", None)
        self.n_sample = self.data_config['n_sample']
        self.model_name = self.train_params['name']
        self.n_epochs = self.train_params['n_epochs']
        self.batch_size = self.train_params['batch_size']
        self.l2_penalty = self.train_params['l2_penalty']
        self.learning_rate = self.train_params['learning_rate']
        self.gpu_flg = torch.cuda.is_available()
        self.log_metrics = self.train_params.get('log_metrics', False)

        if self.log_metrics and (dump_folder is not None):
            self.writer = SummaryWriter(log_dir=op.join(dump_folder, f"tensorboard_log_{random_seed}"))
            self.train_losses = []
            self.val_losses = []

    def train(self, train_t: PVTrainDataSetTorch, val_t: PVTrainDataSetTorch, verbose: int = 0):
        if self.model_name == "naive_neural_net_AY":
            # inputs consist of only A
            model = Naive_NN_for_dsprite_AY(train_params=self.train_params)
        elif self.model_name == "naive_neural_net_AWZY":
            # inputs consist of A, W, and Z (and Z is 2-dimensional)
            model = Naive_NN_for_dsprite_AWZY(train_params=self.train_params)
        else:
            raise ValueError(f"name {self.model_name} is not known")

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            val_t = val_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=self.learning_rate, weight_decay=self.l2_penalty)
        loss = nn.MSELoss()

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                optimizer.zero_grad()

                indices = permutation[i:i + self.batch_size]
                batch_y = train_t.outcome[indices]

                if self.model_name == "naive_neural_net_AY":
                    batch_inputs = train_t.treatment[indices]
                    pred_y = model(batch_inputs.reshape(-1, 1, 64, 64))
                if self.model_name == "naive_neural_net_AWZY":
                    batch_A, batch_W, batch_Z, batch_y = train_t.treatment[indices], train_t.outcome_proxy[indices], \
                                                         train_t.treatment_proxy[indices], train_t.outcome[indices]
                    batch_A = batch_A.reshape(-1, 1, 64, 64)
                    batch_W = batch_A.reshape(-1, 1, 64, 64)
                    pred_y = model(batch_A, batch_W, batch_Z)
                output = loss(pred_y, batch_y)
                output.backward()
                optimizer.step()

            if self.log_metrics:
                with torch.no_grad():
                    if self.model_name == "naive_neural_net_AY":
                        preds_train = model(train_t.treatment.reshape(-1, 1, 64, 64))
                        preds_val = model(val_t.treatment.reshape(-1, 1, 64, 64))
                    elif self.model_name == "naive_neural_net_AWZY":
                        preds_train = model(train_t.treatment.reshape(-1, 1, 64, 64), train_t.outcome_proxy.reshape(-1, 1, 64, 64), train_t.treatment_proxy)
                        preds_val = model(val_t.treatment.reshape(-1, 1, 64, 64), val_t.outcome_proxy.reshape(-1, 1, 64, 64), val_t.treatment_proxy)

                    # "Observed" MSE (not causal MSE) loss calculation
                    mse_train = loss(preds_train, train_t.outcome)
                    mse_val = loss(preds_val, val_t.outcome)
                    self.writer.add_scalar('obs_MSE/train', mse_train, epoch)
                    self.writer.add_scalar('obs_MSE/val', mse_val, epoch)
                    self.train_losses.append(mse_train)
                    self.val_losses.append(mse_val)

        return model

    @staticmethod
    def predict(model, test_data_t: PVTestDataSetTorch, val_data_t: PVTrainDataSetTorch, batch_size=None):

        model_name = model.train_params['name']
        intervention_array_len = test_data_t.treatment.shape[0]
        num_W_test = val_data_t.outcome_proxy.shape[0]

        mean = torch.nn.AvgPool1d(kernel_size=num_W_test, stride=num_W_test)
        with torch.no_grad():
            if batch_size is None:
                # create n_sample copies of each test image (A), and 588 copies of each proxy image (W)
                # reshape test and proxy image to 1 x 64 x 64 (so that the model's conv2d layer is happy)
                test_A = test_data_t.treatment.repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                test_W = val_data_t.outcome_proxy.repeat(intervention_array_len, 1).reshape(-1, 1, 64, 64)
                test_Z = val_data_t.treatment_proxy.repeat(intervention_array_len, 1)
                if model_name == "naive_neural_net_AY":
                    E_w_haw = mean(model(test_A).unsqueeze(-1).T)
                elif model_name == "naive_neural_net_AWZY":
                    E_w_haw = mean(model(test_A, test_W, test_Z).unsqueeze(-1).T)
            else:
                # the number of A's to evaluate each batch
                a_step = max(1, batch_size//num_W_test)
                E_w_haw = torch.zeros([1, 1, intervention_array_len])
                for a_idx in range(0, intervention_array_len, a_step):
                    temp_A = test_data_t.treatment[a_idx:(a_idx+a_step)].repeat_interleave(num_W_test, dim=0).reshape(-1, 1, 64, 64)
                    temp_W = val_data_t.outcome_proxy.repeat(a_step, 1).reshape(-1, 1, 64, 64)
                    temp_Z = val_data_t.treatment_proxy.repeat(a_step, 1)
                    # in this case, we're only predicting for a single A, so we have a ton of W's
                    # therefore, we'll batch this step as well
                    if a_step == 1:
                        model_preds = torch.zeros((temp_A.shape[0]))
                        for temp_idx in range(0, temp_A.shape[0], batch_size):
                            if model_name == "naive_neural_net_AY":
                                model_preds[temp_idx:(temp_idx+batch_size)] = model(temp_A[temp_idx:temp_idx+batch_size]).squeeze()
                            elif model_name == "naive_neural_net_AWZY":
                                model_preds[temp_idx:(temp_idx+batch_size)] = model(temp_A[temp_idx:temp_idx+batch_size], temp_W[temp_idx:temp_idx+batch_size], temp_Z[temp_idx:temp_idx+batch_size]).squeeze()
                        E_w_haw[0, 0, a_idx] = torch.mean(model_preds)
                    else:
                        if model_name == "naive_neural_net_AY":
                            temp_E_w_haw = mean(model(temp_A).unsqueeze(-1).T)
                        elif model_name == "naive_neural_net_AWZY":
                            temp_E_w_haw = mean(model(temp_A, temp_W, temp_Z).unsqueeze(-1).T)
                        E_w_haw[0, 0, a_idx:(a_idx+a_step)] = temp_E_w_haw[0, 0]

        return E_w_haw.T.squeeze(1).cpu().detach().numpy()
from typing import Optional, Dict, Any
from pathlib import Path
import os.path as op
import torch
import numpy as np
import pandas as pd
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.models.naive_neural_net.naive_nn_model import Naive_NN_for_demand
from src.utils.make_AWZ_test import make_AWZ_test


class Naive_NN_Trainer(object):
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


def naive_nn_demand_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
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
    trainer = Naive_NN_Trainer(data_config, model_param, random_seed, one_mdl_dump_dir)
    model = trainer.train(train_t, val_data_t, verbose)

    # prepare test and val data on the gpu
    if trainer.gpu_flg:
        # torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()
        val_data_t = val_data_t.to_gpu()

    # get model predictions on do(A) intervention values
    if model_param['name'] == "naive_neural_net_AY":
        pred = model(test_data_t.treatment).cpu().detach().numpy()

    elif model_param['name'] == "naive_neural_net_AWZY":
        AWZ_test = make_AWZ_test(test_data_t, val_data_t)
        pred = torch.mean(model(AWZ_test), dim=1).cpu().detach().numpy()

    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if trainer.log_metrics:
        return oos_loss, pd.DataFrame(
            data={'obs_MSE_train': torch.Tensor(trainer.train_losses[-50:], device="cpu").numpy(),
                  'obs_MSE_val': torch.Tensor(trainer.val_losses[-50:], device="cpu").numpy()})
    else:
        return oos_loss


if __name__ == "__main__":
    data_config = {"name": "demand", "n_sample": 5000}
    model_param = {
        "name": "naive_neural_net_AWZY",
        "n_epochs": 50,
        "batch_size": 1000,
        "learning_rate": 3e-4,
        "l2_penalty": 3e-6
    }

    one_mdl_dump_dir = Path(op.join("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/dumps", "temp_new"))
    naive_nn_demand_experiment(data_config, model_param, one_mdl_dump_dir, random_seed=41, verbose=0)

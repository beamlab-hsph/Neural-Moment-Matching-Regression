from typing import Optional, Dict, Any
from pathlib import Path
import torch
import numpy as np
from torch import optim, nn
from tqdm import tqdm

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.models.naive_neural_net.naive_nn_model import Naive_NN_for_demand


class Naive_NN_Trainer(object):
    def __init__(self, data_configs: Dict[str, Any], train_params: Dict[str, Any], dump_folder: Optional[Path] = None):

        self.data_config = data_configs
        self.n_sample = self.data_config['n_sample']
        self.n_epochs = train_params['n_epochs']
        self.batch_size = train_params['batch_size']
        self.gpu_flg = torch.cuda.is_available()

    def train(self, train_t: PVTrainDataSetTorch, verbose: int = 0) -> Naive_NN_for_demand:

        # inputs consist of only A
        model = Naive_NN_for_demand(input_dim=1)

        if self.gpu_flg:
            train_t = train_t.to_gpu()
            model.cuda()

        # weight_decay implements L2 penalty
        optimizer = optim.Adam(list(model.parameters()), lr=3e-4, weight_decay=3e-6)
        loss = nn.MSELoss()

        # train model
        for epoch in tqdm(range(self.n_epochs)):
            permutation = torch.randperm(self.n_sample)

            for i in range(0, self.n_sample, self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_A, batch_y = train_t.treatment[indices], train_t.outcome[indices]

                # training loop
                optimizer.zero_grad()
                pred_y = model(batch_A)
                output = loss(pred_y, batch_y)
                output.backward()
                optimizer.step()

        return model


def naive_nn_demand_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
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

    # train model
    trainer = Naive_NN_Trainer(data_config, model_param)
    model = trainer.train(train_t, verbose)

    # get model predictions on do(A) intervention values
    pred = model(test_data_t.treatment).detach().numpy()
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return oos_loss

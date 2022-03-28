from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from sklearn import linear_model


def linear_reg_demand_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
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

    # train model
    model = linear_model.LinearRegression()
    model.fit(train_t.treatment.reshape(-1, 1), train_t.outcome.reshape(-1, 1))

    # get model predictions on do(A) intervention values
    pred = model.predict(test_data.treatment.reshape(-1, 1))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return oos_loss

from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
import os.path as op

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from sklearn import linear_model
from src.utils.make_AW_test import make_AW_test


def twoSLS_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                           one_mdl_dump_dir: Path,
                           random_seed: int = 42, verbose: int = 0):

    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    first_stage_train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    second_stage_train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed + 2)
    val_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data_org = generate_test_data_ate(data_config=data_config)

    # preprocess data
    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    first_stage_train_data = preprocessor.preprocess_for_train(first_stage_train_data_org)
    second_stage_train_data = preprocessor.preprocess_for_train(second_stage_train_data_org)
    first_stage_train_t = PVTrainDataSetTorch.from_numpy(first_stage_train_data)
    second_stage_train_t = PVTrainDataSetTorch.from_numpy(second_stage_train_data)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    val_data = preprocessor.preprocess_for_train(val_data_org)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    # train 2SLS model (from Miao et al.)
    first_stage_model = linear_model.LinearRegression()  # W ~ A + Z
    second_stage_model = linear_model.LinearRegression()  # Y ~ A + \hat{W}

    first_stage_W = first_stage_train_t.outcome_proxy.reshape(-1, 1)
    first_stage_features = torch.cat((first_stage_train_t.treatment, first_stage_train_t.treatment_proxy), dim=1)
    first_stage_model.fit(first_stage_features, first_stage_W)
    W_hat = torch.Tensor(first_stage_model.predict(torch.cat((second_stage_train_t.treatment, second_stage_train_t.treatment_proxy), dim=1)))
    second_stage_model.fit(torch.cat((second_stage_train_t.treatment, W_hat), dim=1), second_stage_train_t.outcome.reshape(-1, 1))

    AW_test = make_AW_test(test_data_t, val_data_t)

    # get model predictions on do(A) intervention values
    pred = [np.mean(second_stage_model.predict(AW_test[i, :, :])) for i in range(AW_test.shape[0])]
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    return oos_loss


if __name__ == "__main__":
    data_config = {"name": "demand", "n_sample": 5000}
    model_param = {"name": "twoSLS"}

    one_mdl_dump_dir = Path(op.join("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/dumps", "temp_new"))
    twoSLS_experiment(data_config, model_param, one_mdl_dump_dir, random_seed=41, verbose=0)

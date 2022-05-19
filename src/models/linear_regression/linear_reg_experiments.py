from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
import os.path as op

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.data.ate import generate_train_data_ate, generate_test_data_ate, get_preprocessor_ate
from sklearn import linear_model
from src.utils.make_AWZ_test import make_AWZ_test
from src.utils.make_AWZ2_test import make_AWZ2_test


def linear_reg_demand_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                           one_mdl_dump_dir: Path,
                           random_seed: int = 42, verbose: int = 0):
    model_name = model_param['name']

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
    model = linear_model.LinearRegression()

    Y = train_t.outcome.reshape(-1, 1)
    if model_name == "linear_regression_AY":
        features = train_t.treatment.reshape(-1, 1)
        model.fit(features, Y)

        # get model predictions on do(A) intervention values
        pred = model.predict(test_data.treatment.reshape(-1, 1))

    elif model_name == "linear_regression_AWZY":
        features = torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.treatment_proxy), dim=1)
        model.fit(features, Y)

        # get model predictions on do(A) intervention values
        AWZ_test = make_AWZ_test(test_data_t, val_data_t)
        pred = [np.mean(model.predict(AWZ_test[i, :, :])) for i in range(AWZ_test.shape[0])]

    elif model_name == "linear_regression_AY2":
        features = torch.cat((train_t.treatment, train_t.treatment ** 2), dim=1)
        model.fit(features, Y)

        # get model predictions on do(A) intervention values
        test_features = np.concatenate((test_data.treatment, test_data.treatment ** 2), axis=-1)
        pred = model.predict(test_features)

    elif model_name == "linear_regression_AWZY2":
        features = torch.cat((train_t.treatment, train_t.outcome_proxy, train_t.treatment_proxy,
                              train_t.treatment**2, train_t.outcome_proxy**2, train_t.treatment_proxy**2,
                              train_t.treatment * train_t.outcome_proxy,
                              train_t.treatment * train_t.treatment_proxy,
                              train_t.outcome_proxy * train_t.treatment_proxy,
                              train_t.treatment_proxy[:, 0:1] * train_t.treatment_proxy[:, 1:2]), dim=-1)
        model.fit(features, Y)

        # get model predictions on do(A) intervention values
        AWZ2_test = make_AWZ2_test(test_data_t, val_data_t)
        pred = [np.mean(model.predict(AWZ2_test[i, :, :])) for i in range(AWZ2_test.shape[0])]

    else:
        raise ValueError(f"name {model_name} is not known")

    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    return oos_loss

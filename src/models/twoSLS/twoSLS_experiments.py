from typing import Dict, Any
from pathlib import Path
import torch
import numpy as np
import os.path as op

from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch, RHCTestDataSetTorch, PVTrainDataSet
from src.data.ate import generate_train_data_ate, generate_val_data_ate, generate_test_data_ate, get_preprocessor_ate
from sklearn import linear_model
from src.utils.make_AW_test import make_AW_test


def twoSLS_RHCexperiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                         one_mdl_dump_dir: Path,
                         random_seed: int = 42):

    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # load train/val/test data
    train_data = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data = generate_test_data_ate(data_config=data_config)

    # Combine the train & val splits, then evenly divide in half for first stage & second stage regression
    A = np.concatenate((train_data.treatment, val_data.treatment))
    Z = np.concatenate((train_data.treatment_proxy, val_data.treatment_proxy))
    W = np.concatenate((train_data.outcome_proxy, val_data.outcome_proxy))
    Y = np.concatenate((train_data.outcome, val_data.outcome))
    X = np.concatenate((train_data.backdoor, val_data.backdoor))

    n_sample = len(A)
    permutation = torch.randperm(n_sample)
    first_stage_ix = permutation[: len(permutation) // 2]
    second_stage_ix = permutation[len(permutation) // 2:]

    first_stage_train = PVTrainDataSet(treatment=A[first_stage_ix],
                                       treatment_proxy=Z[first_stage_ix],
                                       outcome_proxy=W[first_stage_ix],
                                       outcome=Y[first_stage_ix],
                                       backdoor=X[first_stage_ix])

    second_stage_train = PVTrainDataSet(treatment=A[second_stage_ix],
                                        treatment_proxy=Z[second_stage_ix],
                                        outcome_proxy=W[second_stage_ix],
                                        outcome=Y[second_stage_ix],
                                        backdoor=X[second_stage_ix])

    # convert datasets to Torch (for GPU runtime)
    first_stage_train_t = PVTrainDataSetTorch.from_numpy(first_stage_train)
    second_stage_train_t = PVTrainDataSetTorch.from_numpy(second_stage_train)
    test_data_t = RHCTestDataSetTorch.from_numpy(test_data)

    # prepare test data on the gpu
    if torch.cuda.is_available():
        # torch.cuda.empty_cache()
        first_stage_train_t = first_stage_train_t.to_gpu()
        second_stage_train_t = second_stage_train_t.to_gpu()
        test_data_t = test_data_t.to_gpu()

    # train 2SLS model (from Miao et al.)
    first_stage_model1 = linear_model.LinearRegression()  # W1 ~ A + X + Z
    first_stage_model2 = linear_model.LinearRegression()  # W2 ~ A + X + Z
    second_stage_model = linear_model.LinearRegression()  # Y ~ A' + X' + \hat{W}

    first_stage_W1 = first_stage_train_t.outcome_proxy[:, 0].reshape(-1, 1)
    first_stage_W2 = first_stage_train_t.outcome_proxy[:, 1].reshape(-1, 1)
    first_stage_features = torch.cat((first_stage_train_t.treatment, first_stage_train_t.backdoor, first_stage_train_t.treatment_proxy), dim=1)
    first_stage_model1.fit(first_stage_features, first_stage_W1)
    first_stage_model2.fit(first_stage_features, first_stage_W2)

    W_hat1 = torch.Tensor(first_stage_model1.predict(
        torch.cat((second_stage_train_t.treatment, second_stage_train_t.backdoor, second_stage_train_t.treatment_proxy), dim=1)))

    W_hat2 = torch.Tensor(first_stage_model2.predict(torch.cat((second_stage_train_t.treatment, second_stage_train_t.backdoor,
                                                                second_stage_train_t.treatment_proxy), dim=1)))

    W_hat = torch.cat((W_hat1, W_hat2), dim=1)
    second_stage_model.fit(torch.cat((second_stage_train_t.treatment, second_stage_train_t.backdoor, W_hat), dim=1),
                           second_stage_train_t.outcome)

    # Create a 3-dim array with shape [2, n_samples, (3 + len(X))]
    # The first axis contains the two values of do(A): 0 and 1
    # The last axis contains A, W, X, needed for the model's forward pass
    intervention_array_len = 2
    n_samples = test_data_t.outcome_proxy.shape[0]
    tempA = test_data_t.treatment.unsqueeze(-1).expand(-1, n_samples, -1)
    tempW = test_data_t.outcome_proxy.unsqueeze(0).expand(intervention_array_len, -1, -1)
    tempX = test_data_t.backdoor.unsqueeze(0).expand(intervention_array_len, -1, -1)
    model_inputs_test = torch.dstack((tempA, tempW, tempX))

    # get model predictions for A=0 and A=1 on test data
    EY_noRHC = np.mean(second_stage_model.predict(model_inputs_test[0, :, :]))
    EY_RHC = np.mean(second_stage_model.predict(model_inputs_test[1, :, :]))
    pred = [EY_noRHC, EY_RHC]
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if hasattr(test_data, 'structural'):
        # test_data.structural is equivalent to EY_doA
        np.testing.assert_array_equal(pred.shape, test_data.structural.shape)
        oos_loss = np.mean((pred - test_data.structural) ** 2)
    else:
        oos_loss = None


def twoSLS_Demandexperiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                            one_mdl_dump_dir: Path,
                            random_seed: int = 42):
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
    W_hat = torch.Tensor(first_stage_model.predict(
        torch.cat((second_stage_train_t.treatment, second_stage_train_t.treatment_proxy), dim=1)))
    second_stage_model.fit(torch.cat((second_stage_train_t.treatment, W_hat), dim=1),
                           second_stage_train_t.outcome.reshape(-1, 1))

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


def twoSLS_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
                      one_mdl_dump_dir: Path,
                      random_seed: int = 42, verbose: int = 0):
    data_name = data_config.get("name", None)

    if data_name.lower() == 'demand':
        return twoSLS_Demandexperiment(data_config, model_param, one_mdl_dump_dir, random_seed)
    elif data_name.lower() == 'rhc':
        return twoSLS_RHCexperiment(data_config, model_param, one_mdl_dump_dir, random_seed)
    else:
        raise KeyError(f"The `name` key in config.json was {data_name} but must be one of [demand, rhc]")

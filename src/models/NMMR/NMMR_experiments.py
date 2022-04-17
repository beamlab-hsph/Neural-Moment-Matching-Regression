import os.path as op
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from src.data.ate import generate_train_data_ate, generate_val_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.data.ate.data_class import PVTrainDataSetTorch, PVTestDataSetTorch
from src.models.NMMR.NMMR_trainers import NMMR_Trainer_DemandExperiment, NMMR_Trainer_dSpriteExperiment


def NMMR_experiment(data_config: Dict[str, Any], model_config: Dict[str, Any],
                    one_mdl_dump_dir: Path,
                    random_seed: int = 42, verbose: int = 0):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # generate train data
    train_data_org = generate_train_data_ate(data_config=data_config, rand_seed=random_seed)
    val_data_org = generate_val_data_ate(data_config=data_config, rand_seed=random_seed + 1)
    test_data_org = generate_test_data_ate(data_config=data_config)

    # preprocess data
    preprocessor = get_preprocessor_ate(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    train_t = PVTrainDataSetTorch.from_numpy(train_data)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)
    test_data_t = PVTestDataSetTorch.from_numpy(test_data)
    val_data = preprocessor.preprocess_for_train(val_data_org)
    val_data_t = PVTrainDataSetTorch.from_numpy(val_data)

    # retrieve the trainer for this experiment
    data_name = data_config.get("name", None)
    if data_name == "dsprite":
        trainer = NMMR_Trainer_dSpriteExperiment(data_config, model_config, random_seed, one_mdl_dump_dir)
    elif data_name == "demand":
        trainer = NMMR_Trainer_DemandExperiment(data_config, model_config, random_seed, one_mdl_dump_dir)
    else:
        raise KeyError("No key 'name' found in data_config.")

    # train model
    model = trainer.train(train_t, val_data_t, verbose)

    # prepare test data on the gpu
    if trainer.gpu_flg:
        # torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()
        val_data_t = val_data_t.to_gpu()

    E_w_haw = trainer.predict(model, test_data_t, val_data_t)
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
    data_configuration = {"name": "demand", "n_sample": 5000}
    model_param = {"name": "nmmr",
                   "n_epochs": 50,
                   "batch_size": 1000,
                   "log_metrics": "True",
                   "l2_penalty": 0.003,
                   "learning_rate": 3e-6,
                   "loss_name": "V_statistic",
                   "network_width": 10,
                   "network_depth": 5}

    dump_dir = Path(
        op.join("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/dumps", "temp_new"))
    NMMR_experiment(data_configuration, model_param, dump_dir, random_seed=41, verbose=0)

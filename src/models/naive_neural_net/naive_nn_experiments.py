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
from src.data.ate import generate_train_data_ate, generate_val_data_ate, generate_test_data_ate, get_preprocessor_ate
from src.models.naive_neural_net.naive_nn_trainers import Naive_NN_Trainer_DemandExperiment, Naive_NN_Trainer_dSpriteExperiment
from src.utils.make_AWZ_test import make_AWZ_test

def naive_nn_experiment(data_config: Dict[str, Any], model_param: Dict[str, Any],
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

    data_name = data_config.get("name", None)
    # train model
    if data_name == "demand":
        trainer = Naive_NN_Trainer_DemandExperiment(data_config, model_param, random_seed, one_mdl_dump_dir)
    elif data_name == "dsprite":
        trainer = Naive_NN_Trainer_dSpriteExperiment(data_config, model_param, random_seed, one_mdl_dump_dir)
    model = trainer.train(train_t, val_data_t, verbose)

    # prepare test and val data on the gpu
    if trainer.gpu_flg:
        # torch.cuda.empty_cache()
        test_data_t = test_data_t.to_gpu()
        val_data_t = val_data_t.to_gpu()

    if data_name == "demand":
        E_w_haw = trainer.predict(model, test_data_t, val_data_t)
    elif data_name == "dsprite":
        E_w_haw = trainer.predict(model, test_data_t, val_data_t, batch_size=model_param.get('val_batch_size', None))

    #pred = preprocessor.postprocess_for_prediction(E_w_haw).detach().numpy()
    pred = E_w_haw
    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)

    if test_data.structural is not None:
        # test_data_org.structural is equivalent to EY_doA
        oos_loss: float = np.mean((pred - test_data_org.structural.squeeze()) ** 2)
        if data_config["name"] in ["kpv", "deaner"]:
            oos_loss = np.mean(np.abs(pred.numpy() - test_data_org.structural.squeeze()))

    if trainer.log_metrics:
        return oos_loss, pd.DataFrame(
            data={'obs_MSE_train': torch.Tensor(trainer.train_losses[-50:], device="cpu").numpy(),
                  'obs_MSE_val': torch.Tensor(trainer.val_losses[-50:], device="cpu").numpy()})
    else:
        return oos_loss

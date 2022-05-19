from typing import Dict, Any, Optional
from pathlib import Path
import os
import numpy as np
import pandas as pd
import logging

from src.utils import grid_search_dict
from src.models.kernelPV.model import kpv_experiments
from src.models.DFPV.trainer import dfpv_experiments
from src.models.PMMR.model import pmmr_experiments
from src.models.CEVAE.trainer import cevae_experiments
from src.models.NMMR.NMMR_experiments import NMMR_experiment
from src.models.linear_regression.linear_reg_experiments import linear_reg_demand_experiment
from src.models.naive_neural_net.naive_nn_experiments import naive_nn_experiment
from src.models.twoSLS.twoSLS_experiments import twoSLS_experiment
logger = logging.getLogger()


def get_run_func(mdl_name: str):
    if mdl_name == "kpv":
        return kpv_experiments
    elif mdl_name == "dfpv":
        return dfpv_experiments
    elif mdl_name == "pmmr":
        return pmmr_experiments
    elif mdl_name == "cevae":
        return cevae_experiments
    elif mdl_name == "nmmr":
        return NMMR_experiment
    elif mdl_name in ["linear_regression_AY", "linear_regression_AWZY", "linear_regression_AY2", "linear_regression_AWZY2"]:
        return linear_reg_demand_experiment
    elif mdl_name == "naive_neural_net_AY" or mdl_name == "naive_neural_net_AWZY":
        return naive_nn_experiment
    elif mdl_name == "twoSLS":
        return twoSLS_experiment
    else:
        raise ValueError(f"name {mdl_name} is not known")


def experiments(configs: Dict[str, Any],
                dump_dir: Path,
                num_cpus: int):

    data_config = configs["data"]
    model_config = configs["model"]
    n_repeat: int = configs["n_repeat"]

    if num_cpus <= 1 and n_repeat <= 1:
        verbose: int = 2
    else:
        verbose: int = 0

    run_func = get_run_func(model_config["name"])
    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = dump_dir.joinpath(dump_name)
        os.mkdir(one_dump_dir)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = one_dump_dir.joinpath(mdl_dump_name)
                os.mkdir(one_mdl_dump_dir)
            else:
                one_mdl_dump_dir = one_dump_dir

            if model_config.get("log_metrics", False) == "True":
                test_losses = []
                train_metrics_ls = []
                for idx in range(n_repeat):
                    test_loss, train_metrics = run_func(env_param, mdl_param, one_mdl_dump_dir, idx, verbose)
                    train_metrics['rep_ID'] = idx
                    train_metrics_ls.append(train_metrics)
                    if test_loss is not None:
                        test_losses.append(test_loss)

                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))
                metrics_df = pd.concat(train_metrics_ls).reset_index()
                metrics_df.rename(columns={'index': 'epoch_num'}, inplace=True)
                metrics_df.to_csv(one_mdl_dump_dir.joinpath("train_metrics.csv"), index=False)
            else:
                test_losses = []
                for idx in range(n_repeat):
                    test_loss = run_func(env_param, mdl_param, one_mdl_dump_dir, idx, verbose)
                    if test_loss is not None:
                        test_losses.append(test_loss)
                if test_losses:
                    np.savetxt(one_mdl_dump_dir.joinpath("result.csv"), np.array(test_losses))

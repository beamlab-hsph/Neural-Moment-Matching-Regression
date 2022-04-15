import argparse
import json
import os
import os.path as op

import pandas as pd
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator

from src.utils import grid_search_dict


def get_hyperparameter_results_dataframe(dump_dir):
    config_path = os.path.join(dump_dir, 'configs.json')
    with open(config_path) as config_file:
        config = json.load(config_file)
    data_config = config["data"]
    model_config = config["model"]

    results_df = pd.DataFrame()

    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = os.path.join(dump_dir, dump_name)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = os.path.join(one_dump_dir,mdl_dump_name)
            else:
                one_mdl_dump_dir = one_dump_dir
            combined_param_dict = env_param | mdl_param
            if os.path.exists(one_mdl_dump_dir):
                temp_df = pd.DataFrame([combined_param_dict])
                metrics_df = pd.read_csv(os.path.join(one_mdl_dump_dir, 'train_metrics.csv'))
                temp_df['causal_loss_val'] = np.mean(metrics_df[metrics_df.epoch_num==49].causal_loss_val)

                results_df = pd.concat([results_df, temp_df])

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir')
    parser.add_argument('--out_dir')
    parser.add_argument('--experiment')
    #args = parser.parse_args()
    args = parser.parse_args(['--dump_dir', '/Users/kompa/Downloads/NMMR_tune2/',
                              '--out_dir', '/Users/kompa/Downloads/',
                              '--experiment', 'demand'])

    results_df = get_hyperparameter_results_dataframe(args.dump_dir)
    results_df.to_csv(op.join(args.out_dir, "hp_results_nmmr2.csv"), index=False)
    results_df.to_pickle(op.join(args.out_dir, "hp_results_nmmr2.pkl"))

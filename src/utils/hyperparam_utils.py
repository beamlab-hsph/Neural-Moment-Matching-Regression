import os, json, argparse
import pandas as pd
import numpy as np
from src.utils import grid_search_dict

from src.data.ate.demand_pv import cal_structural, psi, cal_outcome

def get_hyperparameter_results_dataframe(dump_dir):
    config_path = os.path.join(dump_dir, 'configs.json')
    with open(config_path) as config_file:
        config = json.load(config_file)
    data_config = config["data"]
    model_config = config["model"]

    results_df = pd.DataFrame()

    true_value_arr = None
    if args.experiment is not None:
        if args.experiment == 'demand':
            #ticket_prices_fine = np.linspace(10, 30, 1000)
            ticket_prices_coarse = np.linspace(10, 30, 10)
            true_value_arr = np.array([cal_structural(a) for a in ticket_prices_coarse])


    for dump_name, env_param in grid_search_dict(data_config):
        one_dump_dir = os.path.join(dump_dir, dump_name)
        for mdl_dump_name, mdl_param in grid_search_dict(model_config):
            if mdl_dump_name != "one":
                one_mdl_dump_dir = os.path.join(one_dump_dir,mdl_dump_name)
            else:
                one_mdl_dump_dir = one_dump_dir
            combined_param_dict = env_param | mdl_param
            #combined_param_dict['dump_path'] = one_mdl_dump_dir
            if os.path.exists(one_mdl_dump_dir):
                results = np.loadtxt(os.path.join(one_mdl_dump_dir, 'result.csv'))
                temp_df = pd.DataFrame([combined_param_dict])
                temp_df['results'] = [results]

                if true_value_arr is not None:
                    temp_df['mae'] = np.mean(np.abs(results-true_value_arr))
                    temp_df['mse'] = np.mean((results-true_value_arr)**2)

                results_df = pd.concat([results_df, temp_df])

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir')
    parser.add_argument('--out_file')
    parser.add_argument('--experiment')
    args = parser.parse_args()
    # args = parser.parse_args(['--dump_dir', '/Users/kompa/Downloads/04-06-15-34-49',
    #                           '--out_file', '/Users/kompa/Downloads/hp_results_metrics.csv',
    #                           '--experiment', 'demand'])

    results_df = get_hyperparameter_results_dataframe(args.dump_dir)
    results_df.to_csv(args.out_file, index=False)
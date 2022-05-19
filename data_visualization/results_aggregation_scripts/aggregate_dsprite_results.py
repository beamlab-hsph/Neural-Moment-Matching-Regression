import os
import os.path as op
import pandas as pd
import numpy as np

"""
Aggregates the results from the dSprite experiment.

* This script expects a folder named `data_for_dsprite_figures` within this directory (`data_visualization`)
* The folder should contain a sub-directory for each method. Each method folder should contain a unique sub-directory
    for each sample in the experiment: 1000, 5000, and 7500. 
* The name of these directories should follow the convention: "n_sample:7500"

Saves the aggregated results to a csv file in the results/ folder 
"""

cwd = os.getcwd()
data_for_figs = op.join(cwd, "data_for_dsprite_figures")
method_dirs = next(os.walk(data_for_figs))[1]

boxplot_df = pd.DataFrame(columns=['sample_size', 'method', 'oos_mse'])
for method_dir in method_dirs:
    method_dirpath = op.join(data_for_figs, method_dir)

    result_dirs = ['n_sample:1000', 'n_sample:5000', 'n_sample:7500']
    for result_dir in result_dirs:

        method_name = method_dir.split()[0].lower()
        sample_size = int(result_dir.split(':')[1])
        result_dir_path = op.join(data_for_figs, method_dir, result_dir)

        boxplot_result_path = op.join(result_dir_path, "result.csv")
        if op.isdir(op.join(data_for_figs, method_dir, result_dir)):
            oos_mse = np.loadtxt(boxplot_result_path)
            boxplot_df = boxplot_df.append(
                pd.DataFrame({'sample_size': sample_size, 'method': method_name, 'oos_mse': oos_mse}))

        else:
            boxplot_df = boxplot_df.append(
                pd.DataFrame({'sample_size': [sample_size], 'method': [method_name], 'oos_mse': [999999]}))

boxplot_df.to_csv(op.join(data_for_figs, "../../results/aggregated_results_for_figures/dsprite_boxplot_data.csv"), index=False)

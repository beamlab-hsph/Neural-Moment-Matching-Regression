import os
import os.path as op
import pandas as pd
import numpy as np

"""
Computes bias, variance and standard error from the dSprite experiment.

* This script expects a folder named `results` at the same level as this directory (`data_visualization`)
* The results folder should contain an already-aggregated dsprite_predcurve_data.csv file.

Saves the results to csv files in the results/ folder 
"""

cwd = os.getcwd()

dsprite_df = pd.read_csv('../../results/aggregated_results_for_figures/dsprite_predcurve_data.csv')


dsprite_df['bias_diff'] = dsprite_df['pred_EY_doA'] - dsprite_df['true_EY_doA']

# group by sample size, method, treatment level
vars = dsprite_df.groupby(['sample_size', 'method', 'A']).pred_EY_doA.agg(['var'])  # calculate variance of pred_EY_doA
biases = dsprite_df.groupby(['sample_size', 'method', 'A']).bias_diff.agg(['mean'])  # calculate bias: avg diff of pred_EY_doA and true_EY_doA
df = vars.join(biases)
df.rename(columns={"mean": "bias", "var": "variance"}, inplace=True)
df.reset_index(inplace=True)
df['standard_error'] = np.sqrt(df['variance'])
df['bias_over_stderr'] = df['bias'] / df['standard_error']

df.to_csv("../../results/aggregated_results_for_figures/dsprite_biasvariance.csv", index=False)

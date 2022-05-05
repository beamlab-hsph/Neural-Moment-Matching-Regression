import os
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
cwd = os.getcwd()
data_for_figs = op.join(cwd, "data_for_demand_figures")
# method_dirs = next(os.walk(data_for_figs))[1]

# order the method dirs from data_for_demand_figures/ as you wish for the plot
method_dirs = ['KPV', 'pmmr', 'CEVAE', 'DFPV', 'naivenet_ay', 'naivenet_awzy',
               'linear_reg_AY', 'linear_reg_AWZY', 'NMMR_U', 'NMMR_V', 'linear_reg_AY2', 'linear_reg_AWZY2', 'twoSLS']

df_dict = {}
for method_dir in method_dirs:
    method_dirpath = op.join(data_for_figs, method_dir)
    result_dirs = next(os.walk(method_dirpath))[1]

    # overwrite result_dirs with the full range of sample sizes if you want to have blank spots for methods that didn't scale
    result_dirs = ['n_sample:1000', 'n_sample:5000', 'n_sample:10000', 'n_sample:50000']

    for result_dir in result_dirs:

        method_name = method_dir.split()[0].lower()
        sample_size = int(result_dir.split(':')[1])

        result_path = op.join(data_for_figs, method_dir, result_dir, "result.csv")
        if sample_size not in df_dict:
            if op.isdir(op.join(data_for_figs, method_dir, result_dir)):
                df_dict[sample_size] = [pd.read_csv(result_path, header=None, names=[f'{method_name}'])]
            else:
                df_dict[sample_size] = [pd.DataFrame({f"{method_name}": np.zeros(20) - 5})]
        else:
            if op.isdir(op.join(data_for_figs, method_dir, result_dir)):
                df_dict[sample_size].append(pd.read_csv(result_path, header=None, names=[f'{method_name}']))
            else:
                df_dict[sample_size].append(pd.DataFrame({f"{method_name}": np.zeros(20) - 5}))

# Vertically concatenate experimental results that share the same sample size (facilitates plotting)
df_dict2 = {}
for sample_size in df_dict:
    df_dict2[sample_size] = pd.concat(df_dict[sample_size], axis=1, ignore_index=False)

# sort df_dict2 items by increasing sample size (keys)
df_dict2 = dict(sorted(df_dict2.items()))

# Create MSE boxplots for each sample size
for sample_size, df in df_dict2.items():
    g = sns.boxplot(x="variable", y="value", data=pd.melt(df))
    g.set_ylabel("Out-of-sample MSE", fontsize=16)
    g.set_ylim(0, 800)
    g.set_xlabel("Method")
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    g.set_title(f"Causal MSE for Xu's Demand experiment (n={sample_size})")
    plt.tight_layout()
    plt.show()

    # zoom-in plot
    g2 = sns.boxplot(x="variable", y="value", data=pd.melt(df))
    g2.set_ylabel("Out-of-sample MSE", fontsize=16)
    g2.set_ylim(0, 200)
    g2.set_xlabel("Method")
    g2.set_xticklabels(g2.get_xticklabels(), rotation=60)
    g2.set_title(f"Causal MSE for Xu's Demand experiment (n={sample_size})")
    plt.tight_layout()
    plt.show()

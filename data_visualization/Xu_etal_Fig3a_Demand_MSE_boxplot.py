import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
# cwd = os.getcwd()
cwd = "/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/data_visualization"
data_for_figs = op.join(cwd, "data_for_demand_figures")
method_dirs = next(os.walk(data_for_figs))[1]

df_dict = {}
for method_dir in method_dirs:
    method_dirpath = op.join(data_for_figs, method_dir)
    result_dirs = next(os.walk(method_dirpath))[1]
    for result_dir in result_dirs:
        method_name = method_dir.split()[0].lower()
        sample_size = int(result_dir.split(':')[1])

        result_path = op.join(data_for_figs, method_dir, result_dir, "result.csv")
        if sample_size not in df_dict:
            df_dict[sample_size] = [pd.read_csv(result_path, header=None, names=[f'{method_name}'])]
        else:
            df_dict[sample_size].append(pd.read_csv(result_path, header=None, names=[f'{method_name}']))

# Vertically concatenate experimental results that share the same sample size (facilitates plotting)
df_dict2 = {}
for sample_size in df_dict:
    df_dict2[sample_size] = pd.concat(df_dict[sample_size], axis=1, ignore_index=False)

# Create MSE boxplots for each sample size
for sample_size, df in df_dict2.items():
    g = sns.boxplot(x="variable", y="value", data=pd.melt(df))
    g.set_ylabel("Method", fontsize=16)
    g.set_xlabel("Out-of-sample MSE")
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

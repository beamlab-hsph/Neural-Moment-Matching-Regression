import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
cwd = os.getcwd()
data_for_figs = op.join(cwd, "data_for_dsprite_figures")
# method_dirs = next(os.walk(data_for_figs))[1]
# method_dirs = ['KPV', 'PMMR', 'CEVAE', 'DFPV', 'NMMR_u_VGGstyle', 'NMMR_v_VGGstyle', 'small_nmmr_10Ws', 'small_nmmr_100Ws', 'small_nmmr_1000Ws']
method_dirs = ['KPV', 'PMMR', 'CEVAE', 'DFPV', 'naivenet_AY', 'naivenet_AWZY', 'NMMR_u_VGGstyle', 'NMMR_v_VGGstyle']

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

# sort df_dict2 items by increasing sample size (keys)
df_dict2 = dict(sorted(df_dict2.items()))

# Create MSE boxplots for each sample size
for sample_size, df in df_dict2.items():
    g = sns.boxplot(x="variable", y="value", data=pd.melt(df))
    g.set_ylabel("Out-of-sample MSE", fontsize=16)
    g.set_ylim(0, 90)
    g.set_xlabel("Method")
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    g.set_title(f"Causal MSE for Xu's dSprite experiment (n={sample_size})")
    plt.tight_layout()
    plt.show()

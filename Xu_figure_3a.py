import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load results
kpv_1000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-06-52 KPV Fig 3 Demand/n_sample:1000/result.csv", header=None, names=['kpv_1000'])
kpv_5000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-06-52 KPV Fig 3 Demand/n_sample:5000/result.csv", header=None, names=['kpv_5000'])
cevae_1000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-10-43 CEVAE Fig 3 Demand/n_sample:1000/result.csv", header=None, names=['cevae_1000'])
cevae_5000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-10-43 CEVAE Fig 3 Demand/n_sample:5000/result.csv", header=None, names=['cevae_5000'])
pmmr_1000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-16-20 PMMR Fig 3 Demand/n_sample:1000/result.csv", header=None, names=['pmmr_1000'])
pmmr_5000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-16-20 PMMR Fig 3 Demand/n_sample:5000/result.csv", header=None, names=['pmmr_5000'])
dfpv_1000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-21-15 DFPV Fig 3 Demand/n_sample:1000/result.csv", header=None, names=['dfpv_1000'])
dfpv_5000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-21-21-21-15 DFPV Fig 3 Demand/n_sample:5000/result.csv", header=None, names=['dfpv_5000'])
nmmr_1000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-22-02-23-52 NMMR for Fig 3 Demand/n_sample:1000/result.csv", header=None, names=['nmmr_1000'])
nmmr_5000 = pd.read_csv("/Users/dab1963/PycharmProjects/DeepFeatureProxyVariable/dumps/03-22-02-23-52 NMMR for Fig 3 Demand/n_sample:5000/result.csv", header=None, names=['nmmr_5000'])

# Vertically concatenate 1000 & 5000 dataset size results
df_1000 = pd.concat([kpv_1000, cevae_1000, pmmr_1000, dfpv_1000, nmmr_1000], axis=1, ignore_index=False)
df_5000 = pd.concat([kpv_5000, cevae_5000, pmmr_5000, dfpv_5000, nmmr_5000], axis=1, ignore_index=False)

g = sns.boxplot(x="variable", y="value", data=pd.melt(df_1000))
g.set(ylim=(0, 1000))
g.set_ylabel("Out-of-sample MSE", fontsize=16)
g.set_xlabel("Method")
g.set_title("Causal MSE for Xu's Demand experiment (n=1000)")
plt.show()

g2 = sns.boxplot(x="variable", y="value", data=pd.melt(df_5000))
g2.set(ylim=(0, 1000))
g2.set_ylabel("Out-of-sample MSE", fontsize=16)
g2.set_xlabel("Method")
g2.set_title("Causal MSE for Xu's Demand experiment (n=5000)")
plt.show()

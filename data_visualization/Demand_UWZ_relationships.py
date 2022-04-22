import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from src.data.ate.demand_pv import cal_structural, psi, cal_outcome, generatate_demand_core


# Create a mock training dataset (to show the training distribution of A and Y)
n_samples = 1000
demand, cost1, cost2, price, views, outcom = generatate_demand_core(n_sample=n_samples, rng=default_rng(seed=42))


plt.scatter(cost1, demand, alpha=0.5, label="$Z_1$")
plt.scatter(cost2, demand, alpha=0.5, label="$Z_2$")
plt.ylabel("Demand (U)")
plt.xlabel("Fuel cost (Z)")
plt.title(f"Relationship between \n U (demand) and Z (fuel cost) (n={n_samples})")
plt.legend()
plt.show()

plt.scatter(views, demand, alpha=0.5, label="W")
plt.ylabel("Demand (U)")
plt.xlabel("Webpage views (W)")
plt.title(f"Relationship between \n U (demand) and W (webpage views) (n={n_samples})")
plt.show()

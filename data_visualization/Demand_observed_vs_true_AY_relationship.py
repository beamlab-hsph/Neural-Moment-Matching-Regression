import numpy as np
import matplotlib.pyplot as plt
from src.data.ate.demand_pv import cal_structural, psi, cal_outcome
from src.data.ate import generate_train_data_ate


# Create a mock training dataset (to show the training distribution of A and Y)
n_samples = 1000
W_noise = 1
ticket_prices_fine = np.linspace(10, 30, n_samples)
EY_doA = np.array([cal_structural(a, W_noise) for a in ticket_prices_fine])

train_data = generate_train_data_ate(data_config={"name": "demand", "n_sample": n_samples}, rand_seed=42)
plt.plot(ticket_prices_fine, EY_doA, color="black", label="true E[Y | do(A)] (test range)")
plt.scatter(train_data.treatment, train_data.outcome, alpha=0.5, label="observed (Y, A) (train range)")
plt.xlabel("Plane ticket price (A)")
plt.ylabel("Ticket sales")
plt.title(f"Relationship between \n A (ticket price) and Y (ticket sales) (n={n_samples})")
plt.legend()
plt.show()

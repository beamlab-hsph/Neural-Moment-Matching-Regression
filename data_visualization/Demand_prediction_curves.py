import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from src.data.ate.demand_pv import cal_structural, psi, cal_outcome

# Create the test values for A (ticket price) and Y (ticket sales)
ticket_prices_fine = np.linspace(10, 30, 1000)
ticket_prices_coarse = np.linspace(10, 30, 10)
EY_doA = np.array([cal_structural(a) for a in ticket_prices_fine])

cwd = os.getcwd()
data_for_figs = op.join(cwd, "data_for_demand_figures")
method_dirs = next(os.walk(data_for_figs))[1]

sample_sizes_to_plot = [1000, 5000]

for n in sample_sizes_to_plot:

    num_subplots = len(method_dirs)
    num_figure_rows = int(np.ceil(num_subplots / 3))
    fig, axs = plt.subplots(num_figure_rows, 3, sharey=True)

    # Delete any empty plots on the last row
    if num_subplots % 3 == 1:
        fig.delaxes(axs[num_figure_rows - 1, 1])
        fig.delaxes(axs[num_figure_rows - 1, 2])
    elif num_subplots % 3 == 2:
        fig.delaxes(axs[num_figure_rows - 1, 2])

    dot_plot = False
    for i, method_dir in enumerate(method_dirs):
        x_coord = 0 + (i // 3)
        y_coord = 0 + (i % 3)

        method_name = method_dir.split()[0].lower()
        method_dirpath = op.join(data_for_figs, method_dir)
        result_path = op.join(method_dirpath, f"n_sample:{n}")

        # Plot the true EY_doA curve and prepare the subplot
        axs[x_coord, y_coord].plot(ticket_prices_fine, EY_doA, color="black")
        axs[x_coord, y_coord].set_title(f"{method_name}", fontsize=10)

        for filename in os.listdir(result_path):
            if filename.endswith('.txt'):
                strings = open(os.path.join(result_path, filename)).read().split()
                pred_EY_doA = np.array([float(x) for x in strings])
                if dot_plot:
                    axs[x_coord, y_coord].scatter(ticket_prices_coarse, pred_EY_doA)
                else:
                    axs[x_coord, y_coord].plot(ticket_prices_coarse, pred_EY_doA, linewidth=1, linestyle='dashed')

    for ax in axs.flat:
        ax.set(xlabel='Plane ticket price (A)')

    # fig.text(0, 0.5, 'Expected plane ticket sales: E[Y | do(A)]', va='center', rotation='vertical')
    fig.supylabel('Expected plane ticket sales: E[Y | do(A)]')
    fig.suptitle(f"Predictions of E[Y | do(A)] by each method (n={n})")
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()

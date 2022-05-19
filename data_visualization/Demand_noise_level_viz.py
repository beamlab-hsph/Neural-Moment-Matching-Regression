import os
import re
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.ate.demand_pv import cal_structural
from src.utils.misc_utils import sort_by_noise_level

# Create the test values for A (ticket price) and Y (ticket sales)
ticket_prices_fine = np.linspace(10, 30, 1000)
ticket_prices_coarse = np.linspace(10, 30, 10)
W_noise = 1  # just use 1 for plotting the smooth black test curve (for data viz)
EY_doA_fine = np.array([cal_structural(a, W_noise) for a in ticket_prices_fine])
EY_doA_coarse = np.array([cal_structural(a, W_noise) for a in ticket_prices_coarse])

cwd = os.getcwd()
data_for_figs = op.join(cwd, "data_for_demand_noiselevel_figures")
# method_dirs = next(os.walk(data_for_figs))[1]
method_dirs = ['KPV', 'pmmr', 'CEVAE', 'linear_reg_AY', 'linear_reg_AY2',
               'linear_reg_AWZY', 'linear_reg_AWZY2', 'naivenet_AWZY',
               'DFPV', 'nmmr_u', 'nmmr_v']

plt.rc('font', size=18)
plt.rc('axes', titlesize=10)

for method in method_dirs:

    method_name = method.split()[0].lower()
    method_path = op.join(data_for_figs, method)
    noise_dirs = next(os.walk(method_path))[1]
    noise_dirs = sort_by_noise_level(noise_dirs)

    num_subplots = len(noise_dirs)
    num_figure_rows = 9
    num_figure_cols = 8
    fig, axs = plt.subplots(num_figure_rows, num_figure_cols, sharey=True, figsize=(15, 15))

    dot_plot = False
    for i, noise_dir in enumerate(noise_dirs):
        x_coord = 0 + (i // num_figure_cols)
        y_coord = 0 + (i % num_figure_cols)

        noise_level_title = noise_dir
        method_dirpath = op.join(method_path, noise_dir)

        # Plot the true EY_doA curve and prepare the subplot
        axs[x_coord, y_coord].plot(ticket_prices_fine, EY_doA_fine, color="black")
        axs[x_coord, y_coord].set_title(f"{noise_level_title}")

        for filename in os.listdir(method_dirpath):
            if filename.endswith('.txt'):
                strings = open(os.path.join(method_dirpath, filename)).read().split()
                pred_EY_doA = np.array([float(x) for x in strings])
                if dot_plot:
                    axs[x_coord, y_coord].scatter(ticket_prices_coarse, pred_EY_doA)
                else:
                    axs[x_coord, y_coord].plot(ticket_prices_coarse, pred_EY_doA, linewidth=1, linestyle='dashed')

    # for ax in axs.flat:
    #     ax.set(xlabel='Plane ticket price (A)')

    # fig.text(0, 0.5, 'Expected plane ticket sales: E[Y | do(A)]', va='center', rotation='vertical')
    fig.supylabel('Expected plane ticket sales: $E[Y^a]$')
    fig.supxlabel('Plane ticket price (A)')
    fig.suptitle(f"Predictions of $E[Y^a]$ by {method_name}")
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()


num_subplots = len(method_dirs)
num_figure_rows = int(np.ceil(num_subplots / 3))
fig, axs = plt.subplots(num_figure_rows, 3, sharey=True, sharex=True, figsize=(15, 15), visible=True)

plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.rc('ytick', labelsize=20)

# Delete any empty plots on the last row
if num_subplots % 3 == 1:
    fig.delaxes(axs[num_figure_rows - 1, 1])
    fig.delaxes(axs[num_figure_rows - 1, 2])
elif num_subplots % 3 == 2:
    fig.delaxes(axs[num_figure_rows - 1, 2])

for i, method in enumerate(method_dirs):

    x_coord = 0 + (i // 3)
    y_coord = 0 + (i % 3)

    method_name = method.split()[0].lower()
    method_path = op.join(data_for_figs, method)
    noise_dirs = next(os.walk(method_path))[1]
    noise_dirs = sort_by_noise_level(noise_dirs)

    df = pd.DataFrame(columns=['noise_level', 'error'])
    Z_noise = 0
    W_noise = 0
    for j, noise_dir in enumerate(noise_dirs):
        noise_dirpath = op.join(method_path, noise_dir)

        current_noise_levels = re.findall(r'(\d+(?:\.\d+)?)', noise_dir)
        current_Z_noise = float(current_noise_levels[0])
        current_W_noise = float(current_noise_levels[1])

        # if noise_dir not on diagonal, continue
        if current_Z_noise > Z_noise and current_W_noise > W_noise:
            Z_noise = current_Z_noise
            W_noise = current_W_noise
        else:
            # continue
            Z_noise = current_Z_noise
            W_noise = current_W_noise

        for filename in os.listdir(noise_dirpath):
            if filename.endswith('.txt'):
                strings = open(op.join(noise_dirpath, filename)).read().split()
                pred_EY_doA = np.array([float(x) for x in strings])

                # compute the errors
                error = EY_doA_coarse - pred_EY_doA

                # append the errors to df
                df = df.append(pd.DataFrame({'error': error, 'noise_level': noise_dir}))

    g = sns.stripplot(ax=axs[x_coord, y_coord], x="noise_level", y="error", data=df, jitter=0.2, size=3, alpha=0.5)
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    axs[x_coord, y_coord].tick_params(axis='both', labelsize=20)
    axs[x_coord, y_coord].set_title(f"{method_name}")
    axs[x_coord, y_coord].set_ylim(-80, 80)
    axs[x_coord, y_coord].set(xlabel=None, ylabel=None)


fig.supylabel('Error: $E[Y^a] - E_w[h(a, w)]$')
fig.supxlabel('Noise level')
plt.tight_layout()
plt.show()

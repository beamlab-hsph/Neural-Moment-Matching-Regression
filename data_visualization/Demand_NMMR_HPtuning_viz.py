import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("/Users/dab1963/PycharmProjects/Neural-Moment-Matching-Regression/data_visualization/data_for_hyperparameter_tuning/hp_results.pkl")
df.sort_values(by=['median_MSE'], inplace=True)

plt.scatter(list(range(len(df))), df.median_MSE)
plt.xlabel("Hyperparameter config number (arbitrary)")
plt.ylabel("Median out-of-sample MSE")
plt.title("Median out-of-sample MSE across hyperparameter configurations")
plt.show()

plt.scatter(list(range(len(df))), df.median_MSE)
plt.xlabel("Hyperparameter config number (arbitrary)")
plt.ylabel("Median out-of-sample MSE")
plt.title("Median out-of-sample MSE across hyperparameter configurations")
plt.ylim(0, 100)
plt.show()

# Drop any that have a max MSE above 100
df_best = df[df.results.apply(max) < 50]
df_best.index.name = "ix"
df_best_long = df_best.explode("results")

fig, axs = plt.subplots(4, 2, sharey=True)

for i in range(8):
    row_coord = 0 + (i // 2)
    col_coord = 0 + (i % 2)

    sns.boxplot(ax=axs[row_coord, col_coord], x=df_best_long.index[i*50:50*(i+1)], y=df_best_long.results[i*50:50*(i+1)])
    axs[row_coord, col_coord].set

for ax in axs.flat:
    ax.set(xlabel='index #')

fig.supylabel('Out-of-sample MSE')
fig.suptitle(f"MSE boxplots of the best NMMR hyperparameter configurations")
# for ax in axs.flat:
#     ax.label_outer()

plt.setp(ax, ylim=(0, 60))
plt.tight_layout()
plt.show()

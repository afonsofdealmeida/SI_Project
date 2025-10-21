import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the dataset
df = pd.read_excel("../data/processed/cleaned_dataset_tree.xlsx", header=0)

# Select the 4 columns to compare
cols = ['RecovTime_log_e', 'RecovTime_log_2', 'RecovTime_log_10', 'RecovTime_log_20']

colors = ["#1082d3", "#ef760b", "#1dcb1d", "#da1919"]

titles = [
    "Log base e",
    "Log base 2",
    "Log base 10",
    "Log base 20"
]
# Create figure and axes (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

# "axes" is a 2x2 matrix → iterate through with zip
for ax, col, color, title in zip(axes.flat, cols, colors, titles):
    ax.plot(df.index, df[col], '.', alpha=0.7, color=color)  # scatter points
    ax.set_title(title, fontsize=13)
    ax.grid(True)
    ax.set_xlabel("Sample number", fontsize=10)
    ax.set_ylabel("Value of feature", fontsize=10)
    ax.tick_params(labelbottom=True, labelleft=True)

# Main title and clean layout
fig.suptitle("Distribution of RecovTime in a logarithmic scale with different bases", fontsize=16)
plt.tight_layout()
plt.show()
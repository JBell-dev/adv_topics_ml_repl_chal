import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import pickle

# Please provide here the visit count dictionary of each algorithm that is saved at the end of the training.
# The z prefixed files are from runs with zero negative reward on illegal actions.
with open("z_seed9876_ppo_rle_visit_counts.pkl", "rb") as file:
    state_visit_counts_ppo_rle = pickle.load(file)

with open("z_seed9876_ppo_visit_counts.pkl", "rb") as file:
    state_visit_counts_ppo = pickle.load(file)

with open("z_seed9876_ppo_rnd_visit_counts.pkl", "rb") as file:
    state_visit_counts_ppo_rnd = pickle.load(file)

with open("z_seed9876_ppo_noisy_net_visit_counts.pkl", "rb") as file:
    state_visit_counts_ppo_noisy_net = pickle.load(file)


def plot_combined_heatmaps(env_size, visit_counts_dicts, titles, save_dir):
    """
    Plots combined heatmaps for multiple visit count dictionaries.

    Args:
        env_size (int): Size of the environment (width and height assumed equal).
        visit_counts_dicts (list of dict): List of visit count dictionaries.
        titles (list of str): Titles for each subplot.
        save_dir (str): Directory to save the plot.
    """
    assert len(visit_counts_dicts) == len(titles), "Number of visit count dictionaries must be equal to the number of titles"

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    # Determine global min and max
    min_val = float("inf")
    max_val = -1
    for visit_counts_dict in visit_counts_dicts:
        for value in visit_counts_dict.values():
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value

    # Plot each heatmap
    for i, (visit_counts_dict, title) in enumerate(zip(visit_counts_dicts, titles)):
        ax = axes[i // 2, i % 2]  # Access the correct subplot
        states = np.zeros((env_size + 1, env_size + 1))
        for (x, y), count in visit_counts_dict.items():
            states[y, x] = count  # imshow expects row, column

        im = ax.imshow(states, cmap='viridis', norm=LogNorm(vmin=min_val, vmax=max_val))
        ax.set_title(title)
        ax.axis('off')

    # Add a shared colorbar
    fig.colorbar(im, ax=axes, location='right', aspect=40, pad=0.02)
    plt.suptitle("State Visit Heatmaps", fontsize=16)
    plt.savefig(f"{save_dir}/combined_state_visit_heatmaps.png")
    print(f"Saved combined heatmap plot: {save_dir}/combined_state_visit_heatmaps.png")

env_size = 50
save_dir = "."

titles = ["RLE", "PPO", "RND", "Noisy Net"]
visit_counts_dicts = [
    state_visit_counts_ppo_rle,
    state_visit_counts_ppo,
    state_visit_counts_ppo_rnd,
    state_visit_counts_ppo_noisy_net
]

plot_combined_heatmaps(env_size, visit_counts_dicts, titles, save_dir)

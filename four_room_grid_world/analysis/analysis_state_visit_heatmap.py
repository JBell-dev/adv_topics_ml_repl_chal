import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

# Need to select the runs with the highest entropy in the no-goal setting and the runs with the highest game_score
# in the setting with a goal.
RUN_REWARD_FREE_RLE = "random-latent-exploration/RLE/bzo2znfs"
RUN_REWARD_FREE_PPO = "random-latent-exploration/RLE/4v3gqh91"
RUN_REWARD_FREE_NOISY_NET = "random-latent-exploration/RLE/6aw0fha8"
RUN_REWARD_FREE_RND = "random-latent-exploration/RLE/c8l3xsv5"
REWARD_FREE_ALG_RUN_LIST = [RUN_REWARD_FREE_RLE, RUN_REWARD_FREE_PPO, RUN_REWARD_FREE_NOISY_NET, RUN_REWARD_FREE_RND]
REWARD_FREE_ALG_LABELS = ["RLE", "PPO", "NoisyNet", "RND"]
REWARD_FREE_ALG_TITLE = "heatmaps_no_goal_algorithms"
# ---
RUN_REWARD_FREE_NORMAL = "random-latent-exploration/RLE/bzo2znfs"
RUN_REWARD_FREE_UNIFORM = "random-latent-exploration/RLE/jybcsrfv"
RUN_REWARD_FREE_VON_MISES = "random-latent-exploration/RLE/9nff7264"
RUN_REWARD_FREE_RND_EXPONENTIAL = "random-latent-exploration/RLE/xo1nqbdl"
REWARD_FREE_DIST_RUN_LIST = [RUN_REWARD_FREE_NORMAL, RUN_REWARD_FREE_UNIFORM, RUN_REWARD_FREE_VON_MISES, RUN_REWARD_FREE_RND_EXPONENTIAL]
REWARD_FREE_DIST_LABELS = ["Standard Normal", "Standard Uniform", "Von Mises", "Exponential"]
REWARD_FREE_DIST_TITLE = "heatmaps_no_goal_distributions"
# ---
RUN_NO_REWARD_FREE_RLE = "random-latent-exploration/RLE/uo31t8jk"
RUN_NO_REWARD_FREE_PPO = "random-latent-exploration/RLE/1rogn8iu"
RUN_NO_REWARD_FREE_NOISY_NET = "random-latent-exploration/RLE/bf3oz896"
RUN_NO_REWARD_FREE_RND = "random-latent-exploration/RLE/wpxm7and"
NO_REWARD_FREE_ALG_RUN_LIST = [RUN_NO_REWARD_FREE_RLE, RUN_NO_REWARD_FREE_PPO, RUN_NO_REWARD_FREE_NOISY_NET, RUN_NO_REWARD_FREE_RND]
NO_REWARD_FREE_ALG_LABELS = ["RLE", "PPO", "NoisyNet", "RND"]
NO_REWARD_FREE_ALG_TITLE = "heatmaps_goal_algorithms"
# ---
RUN_NO_REWARD_FREE_NORMAL = "random-latent-exploration/RLE/nubtwlun"
RUN_NO_REWARD_FREE_UNIFORM = "random-latent-exploration/RLE/xnits1sr"
RUN_NO_REWARD_FREE_VON_MISES = "random-latent-exploration/RLE/ytz9pi8j"
RUN_NO_REWARD_FREE_RND_EXPONENTIAL = "random-latent-exploration/RLE/luz5fgdh"
NO_REWARD_FREE_DIST_RUN_LIST = [RUN_NO_REWARD_FREE_NORMAL, RUN_NO_REWARD_FREE_UNIFORM, RUN_NO_REWARD_FREE_VON_MISES, RUN_NO_REWARD_FREE_RND_EXPONENTIAL]
NO_REWARD_FREE_DIST_LABELS = ["Standard Normal", "Standard Uniform", "Von Mises", "Exponential"]
NO_REWARD_FREE_DIST_TITLE = "heatmaps_goal_distributions"

api = wandb.Api()


def get_state_visit_lists(run_names, api):
    state_visit_lists = []
    for run_name in run_names:
        run = api.run(run_name)
        summary = run.summary
        visit_counts = summary["visit_counts"]
        state_visit_lists.append(visit_counts)
    return state_visit_lists


def plot_state_visit_heatmap(state_visit_lists, labels, file_name):
    assert len(state_visit_lists) == len(labels) == 4

    state_visit_array = np.array(state_visit_lists)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    # Determine global min and max
    min_val = np.max(1, np.min(state_visit_array))  # 0 is not good because of logarithm
    max_val = np.max(state_visit_array)

    # Plot each heatmap
    for i, label in enumerate(labels):
        ax = axes[i // 2, i % 2]
        # Transpose because imshow has vertical x-axis
        im = ax.imshow(state_visit_array[i].T, cmap="viridis", norm=LogNorm(vmin=min_val, vmax=max_val))
        ax.set_title(label)
        ax.axis("off")

    # Add a shared colorbar
    fig.colorbar(im, ax=axes, location="right", aspect=40, pad=0.02)
    plt.savefig(f"./plots/gridworld_{file_name}.pdf", dpi=600, bbox_inches="tight")



state_visit_lists = get_state_visit_lists(REWARD_FREE_ALG_RUN_LIST, api)
plot_state_visit_heatmap(state_visit_lists, REWARD_FREE_ALG_LABELS, REWARD_FREE_ALG_TITLE)

state_visit_lists = get_state_visit_lists(REWARD_FREE_DIST_RUN_LIST, api)
plot_state_visit_heatmap(state_visit_lists, REWARD_FREE_DIST_LABELS, REWARD_FREE_DIST_TITLE)

state_visit_lists = get_state_visit_lists(NO_REWARD_FREE_ALG_RUN_LIST, api)
plot_state_visit_heatmap(state_visit_lists, NO_REWARD_FREE_ALG_LABELS, NO_REWARD_FREE_ALG_TITLE)

state_visit_lists = get_state_visit_lists(NO_REWARD_FREE_DIST_RUN_LIST, api)
plot_state_visit_heatmap(state_visit_lists, NO_REWARD_FREE_DIST_LABELS, NO_REWARD_FREE_DIST_TITLE)
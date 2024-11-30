import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

# Need to select the runs with the highest entropy in the no-goal setting and the runs with the highest game_score
# in the setting with a goal.
RUN_REWARD_FREE_PPO = ""
RUN_REWARD_FREE_NOISY_NET = ""
RUN_REWARD_FREE_RLE = ""
RUN_REWARD_FREE_RND = ""
REWARD_FREE_ALG_RUN_LIST = [RUN_REWARD_FREE_PPO, RUN_REWARD_FREE_NOISY_NET, RUN_REWARD_FREE_RLE, RUN_REWARD_FREE_RND]
REWARD_FREE_ALG_LABELS = ["PPO", "PPO NOISY NET", "RLE", "RND"]
REWARD_FREE_ALG_TITLE = "State visit heatmaps of different exploration methods in the setting without a goal"
# ---
RUN_REWARD_FREE_NORMAL = ""
RUN_REWARD_FREE_UNIFORM = ""
RUN_REWARD_FREE_VON_MISES = ""
RUN_REWARD_FREE_RND_EXPONENTIAL = ""
REWARD_FREE_DIST_RUN_LIST = [RUN_REWARD_FREE_NORMAL, RUN_REWARD_FREE_UNIFORM, RUN_REWARD_FREE_VON_MISES, RUN_REWARD_FREE_RND_EXPONENTIAL]
REWARD_FREE_DIST_LABELS = ["standard_normal", "standard_uniform", "von_mises", "exponential"]
REWARD_FREE_DIST_TITLE = "State visit heatmaps of RLE with different z distributions the setting without a goal"
# ---
RUN_NO_REWARD_FREE_PPO = "random-latent-exploration/RLE/pe2mf5kz"
RUN_NO_REWARD_FREE_NOISY_NET = "random-latent-exploration/RLE/f28bzzj0"
RUN_NO_REWARD_FREE_RLE = "random-latent-exploration/RLE/146oa5h2"
RUN_NO_REWARD_FREE_RND = "random-latent-exploration/RLE/ofsbnmt2"
NO_REWARD_FREE_ALG_RUN_LIST = [RUN_NO_REWARD_FREE_PPO, RUN_NO_REWARD_FREE_NOISY_NET, RUN_NO_REWARD_FREE_RLE, RUN_NO_REWARD_FREE_RND]
NO_REWARD_FREE_ALG_LABELS = ["PPO", "PPO NOISY NET", "RLE", "RND"]
NO_REWARD_FREE_ALG_TITLE = "State visit heatmaps of different exploration methods in the setting with a goal"
# ---
RUN_NO_REWARD_FREE_NORMAL = "random-latent-exploration/RLE/146oa5h2"
RUN_NO_REWARD_FREE_UNIFORM = "random-latent-exploration/RLE/cm4eidwn"
RUN_NO_REWARD_FREE_VON_MISES = "random-latent-exploration/RLE/0ajny05e"
RUN_NO_REWARD_FREE_RND_EXPONENTIAL = "random-latent-exploration/RLE/b212ia7m"
NO_REWARD_FREE_DIST_RUN_LIST = [RUN_NO_REWARD_FREE_NORMAL, RUN_NO_REWARD_FREE_UNIFORM, RUN_NO_REWARD_FREE_VON_MISES, RUN_NO_REWARD_FREE_RND_EXPONENTIAL]
NO_REWARD_FREE_DIST_LABELS = ["standard_normal", "standard_uniform", "von_mises", "exponential"]
NO_REWARD_FREE_DIST_TITLE = "State visit heatmaps of RLE with different z distributions the setting with a goal"

api = wandb.Api()


def get_state_visit_lists(run_names, api):
    state_visit_lists = []
    for run_name in run_names:
        run = api.run(run_name)
        summary = run.summary
        visit_counts = summary["visit_counts"]
        state_visit_lists.append(visit_counts)
    return state_visit_lists


def plot_state_visit_heatmap(state_visit_lists, labels, title):
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
    plt.suptitle(title, fontsize=16)
    plt.show()


state_visit_lists = get_state_visit_lists(NO_REWARD_FREE_ALG_RUN_LIST, api)
plot_state_visit_heatmap(state_visit_lists, NO_REWARD_FREE_ALG_LABELS, NO_REWARD_FREE_ALG_TITLE)

state_visit_lists = get_state_visit_lists(NO_REWARD_FREE_DIST_RUN_LIST, api)
plot_state_visit_heatmap(state_visit_lists, NO_REWARD_FREE_DIST_LABELS, NO_REWARD_FREE_DIST_TITLE)

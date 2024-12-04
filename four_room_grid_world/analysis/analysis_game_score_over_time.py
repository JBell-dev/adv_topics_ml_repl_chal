import numpy as np
from matplotlib import pyplot as plt
import wandb

from four_room_grid_world.analysis.mapper import NAME_MAPPER, COLOR_MAPPER, sort

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

def bootstrap_confidence_interval(data, num_bootstrap=10_000, confidence=0.95):
    if len(data) == 0:
        return 0.0, 0.0
    bootstrap_samples = np.random.choice(data, (num_bootstrap, len(data)), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    lower = np.percentile(bootstrap_means, ((1.0 - confidence) / 2.0) * 100)
    upper = np.percentile(bootstrap_means, (1.0 - (1.0 - confidence) / 2.0) * 100)
    return lower, upper


def get_plot_data(row):
    new_row = []
    for entry in row:
        if entry == "NaN":
            new_row.append(0)
        else:
            new_row.append(entry)

    lower, upper = bootstrap_confidence_interval(new_row)

    return np.array([np.mean(new_row), lower, upper])


def get_game_score_per_step_per_run_per_algorithm(runs, distributions=False):
    """
    Returns a dictionary that contains an entry for every algorithm. The key is the tag of the algorithm. The value
    is a numpy array where dimension zero are the steps and dimension one are the runs for this algorithm. Also
    returns a list of the algorithm tags and a list that contains the global steps at which point the game score was
    calculated.
    """
    data_per_algorithm = {}
    tags = []

    for run in runs:
        if distributions:
            tag = run.tags[2]  # Distribution is 2nd tag
        else:
            tag = run.tags[1]

        history_df = run.history(keys=["charts/game_score", "_step"])
        steps = history_df["_step"].to_numpy()
        game_score = history_df["charts/game_score"].to_numpy()

        if tag not in tags:
            tags.append(tag)

        if tag not in data_per_algorithm.keys():
            data_per_algorithm[tag] = [game_score]
        else:
            data_per_algorithm[tag].append(game_score)

    for key, value in data_per_algorithm.items():
        data_per_algorithm[key] = np.array(value).T

    return data_per_algorithm, tags, steps


def plot_game_score_algorithms(api):
    query_tags = ["standard_normal", "PPO", "PPO_NOISY_NET", "PPO_RND"]

    runs = api.runs("RLE", filters={
        "$and": [
            {"tags": {"$in": query_tags}},
            {"tags": {"$in": ["NOT_REWARD_FREE"]}}
        ]
    })

    data_per_algorithm, tags, steps = get_game_score_per_step_per_run_per_algorithm(runs)
    data_per_algorithm, tags = sort(data_per_algorithm, tags)

    plt.figure(figsize=(8, 6))

    for i in range(len(tags)):
        data = np.array([get_plot_data(row) for row in data_per_algorithm[tags[i]]])
        plt.plot(steps, data[:, 0], label=NAME_MAPPER[tags[i]], color=COLOR_MAPPER[tags[i]], linewidth=2)
        #plt.fill_between(steps, data[:, 1], data[:, 2], color=colors[i], alpha=0.2)

    plt.ylim(bottom=0)
    plt.ylim(top=1)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Game Score", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig("./plots/gridworld_mean_game_score_algorithms.pdf", dpi=600, bbox_inches="tight")


def plot_game_score_distributions(api):
    query_tags = ["standard_normal", "standard_uniform", "von_mises", "exponential"]
    runs = api.runs("RLE", filters={
        "$and": [
            {"tags": {"$in": query_tags}},
            {"tags": {"$in": ["NOT_REWARD_FREE"]}}
        ]
    })

    data_per_algorithm, tags, steps = get_game_score_per_step_per_run_per_algorithm(runs, distributions=True)

    plt.figure(figsize=(8, 6))

    for i in range(len(tags)):
        data = np.array([get_plot_data(row) for row in data_per_algorithm[tags[i]]])
        plt.plot(steps, data[:, 0], label=NAME_MAPPER[tags[i]], color=COLOR_MAPPER[tags[i]], linewidth=2)
        #plt.fill_between(steps, data[:, 1], data[:, 2], color=colors[i], alpha=0.2)

    plt.ylim(bottom=0)
    plt.ylim(top=1)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Game Score", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig("./plots/gridworld_mean_game_score_distributions.pdf", dpi=600, bbox_inches="tight")


api = wandb.Api()
plot_game_score_algorithms(api)
plot_game_score_distributions(api)

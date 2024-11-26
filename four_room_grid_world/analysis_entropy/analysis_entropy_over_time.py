import numpy as np
from matplotlib import pyplot as plt
import wandb


def bootstrap_confidence_interval(data, num_bootstrap=10_000, confidence=0.95):
    if len(data) == 0:
        return 0.0, 0.0
    bootstrap_samples = np.random.choice(data, (num_bootstrap, len(data)), replace=True)
    bootstrap_means = bootstrap_samples.mean(axis=1)
    lower = np.percentile(bootstrap_means, ((1.0 - confidence) / 2.0) * 100)
    upper = np.percentile(bootstrap_means, (1.0 - (1.0 - confidence) / 2.0) * 100)
    return lower, upper


def get_plot_data(row):
    lower, upper = bootstrap_confidence_interval(row)
    return np.array([np.mean(row), lower, upper])


def get_entropy_per_step_per_run_per_algorithm(runs):
    """
    Returns a dictionary that contains an entry for every algorithm. The key is the tag of the algorithm. The value
    is a numpy array where dimension zero are the steps and dimension one are the runs for this algorithm. Also
    returns a list of the algorithm tags and a list that contains the global steps at which point the entropy was
    calculated.
    """
    data_per_algorithm = {}
    tags = []

    for run in runs:
        tag = run.tags[0]

        history_df = run.history(keys=["charts/state_visit_entropy", "_step"])
        steps = history_df["_step"].to_numpy()
        entropy_per_step = history_df["charts/state_visit_entropy"].to_numpy()

        if tag not in tags:
            tags.append(tag)

        if tag not in data_per_algorithm.keys():
            data_per_algorithm[tag] = [entropy_per_step]
        else:
            data_per_algorithm[tag].append(entropy_per_step)

    for key, value in data_per_algorithm.items():
        data_per_algorithm[key] = np.array(value).T

    return data_per_algorithm, tags, steps


def plot_entropy_algorithms(api):
    tags = ["PPO", "PPO_NOISY_NET", "PPO_RLE", "PPO_RND"]
    runs = api.runs("RLE", filters={"tags": {"$in": tags}})

    data_per_algorithm, tags, steps = get_entropy_per_step_per_run_per_algorithm(runs)

    colors = ["gray", "orange", "green", "blue"]

    assert len(tags) == len(colors), "Number of colors does not match number of tags (algorithms)"

    plt.figure(figsize=(8, 6))

    for i in range(len(tags)):
        data = np.array([get_plot_data(row) for row in data_per_algorithm[tags[i]]])
        plt.plot(steps, data[:, 0], label=tags[i], color=colors[i], linewidth=2)
        plt.fill_between(steps, data[:, 1], data[:, 2], color=colors[i], alpha=0.2)

    plt.ylim(bottom=0)
    plt.ylim(top=10)
    plt.title("Mean Entropy of the State Visit Count over the Training", fontsize=14)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def plot_entropy_distributions(api):
    tags = ["PPO_RLE", "PPO_RLE_STANDARD_UNIFORM", "PPO_RLE_VON_MISES", "PPO_RLE_EXPONENTIAL"]
    runs = api.runs("RLE", filters={"tags": {"$in": tags}})

    data_per_algorithm, tags, steps = get_entropy_per_step_per_run_per_algorithm(runs)

    colors = ["gray", "orange", "green", "blue"]

    assert len(tags) == len(colors), "Number of colors does not match number of tags (algorithms)"

    plt.figure(figsize=(8, 6))

    for i in range(len(tags)):
        data = np.array([get_plot_data(row) for row in data_per_algorithm[tags[i]]])
        plt.plot(steps, data[:, 0], label=tags[i], color=colors[i], linewidth=2)
        plt.fill_between(steps, data[:, 1], data[:, 2], color=colors[i], alpha=0.2)

    plt.ylim(bottom=0)
    plt.ylim(top=10)
    plt.title("Mean Entropy of the State Visit Count over the Training", fontsize=14)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


api = wandb.Api()
plot_entropy_distributions(api)
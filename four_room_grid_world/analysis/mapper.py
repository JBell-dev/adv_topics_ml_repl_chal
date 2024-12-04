# Map the tag used in weights & biases to displayable names
NAME_MAPPER = {
    "PPO_RLE": "RLE",
    "PPO": "PPO",
    "PPO_NOISY_NET": "NoisyNet",
    "PPO_RND": "RND",
    "standard_normal": "Standard Normal",
    "standard_uniform": "Standard Uniform",
    "von_mises": "Von Mises",
    "exponential": "Exponential",
}

COLOR_MAPPER = {
    "PPO_RLE": "blue",
    "PPO": "gray",
    "PPO_NOISY_NET": "orange",
    "PPO_RND": "green",
    "standard_normal": "blue",
    "standard_uniform": "gray",
    "von_mises": "orange",
    "exponential": "green",
}

def sort(data_per_algorithm, tags):
    data_per_algorithm_sorted = {}
    tags_sorted = [-1, -1, -1, -1]

    for i in range(len(tags)):
        if tags[i] == "standard_normal" or tags[i] == "PPO_RLE":
            tags_sorted[0] = tags[i]
            data_per_algorithm_sorted[tags_sorted[0]] = data_per_algorithm[tags[i]]
        elif tags[i] == "PPO":
            tags_sorted[1] = tags[i]
            data_per_algorithm_sorted[tags_sorted[1]] = data_per_algorithm[tags[i]]
        elif tags[i] == "PPO_NOISY_NET":
            tags_sorted[2] = tags[i]
            data_per_algorithm_sorted[tags_sorted[2]] = data_per_algorithm[tags[i]]
        elif tags[i] == "PPO_RND":
            tags_sorted[3] = tags[i]
            data_per_algorithm_sorted[tags_sorted[3]] = data_per_algorithm[tags[i]]
        else:
            raise RuntimeError(f"Can not sort because unknown tag provided: {tags[i]}")

    return data_per_algorithm_sorted, tags_sorted
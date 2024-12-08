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
    "PPO_RLE": (84 / 255, 113 / 255, 171 / 255),
    "PPO": (127 / 255, 127 / 255, 127 / 255),
    "PPO_NOISY_NET": (238 / 255, 134 / 255, 54 / 255),
    "PPO_RND": (81 / 255, 158 / 255, 63 / 255),
    "standard_normal": (84 / 255, 113 / 255, 171 / 255),
    "standard_uniform": (127 / 255, 127 / 255, 127 / 255),
    "von_mises": (238 / 255, 134 / 255, 54 / 255),
    "exponential": (81 / 255, 158 / 255, 63 / 255),
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

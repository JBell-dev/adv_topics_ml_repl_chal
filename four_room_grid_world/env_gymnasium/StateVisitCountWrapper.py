import gymnasium as gym
from collections import defaultdict


class StateVisitCountWrapper(gym.Wrapper):
    # Counts the number of times each state is visited.
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.visit_counts = defaultdict(int)  # Single dictionary for all envs

    def reset(self, **kwargs):
        # Reset visit counts
        self.visit_counts = defaultdict(int)
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, dones, truncations, infos = self.env.step(actions)

        # Track visit counts across all environments in the vector
        for i in range(self.num_envs):
            state = tuple(obs[i])  # Convert observation to a hashable type
            self.visit_counts[state] += 1

        # Optionally add visit counts to the `infos` dictionary for each environment
        infos["visit_counts"] = dict(self.visit_counts)

        return obs, rewards, dones, truncations, infos

import gymnasium as gym
from collections import defaultdict
import numpy as np


class StateVisitCountWrapper(gym.Wrapper):
    # Counts the number of times each state is visited.
    def __init__(self, env):
        super().__init__(env)
        self.visit_counts = defaultdict(int)  # Single dictionary for all envs

    def reset(self, **kwargs):
        # Reset visit counts
        self.visit_counts = defaultdict(int)
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, dones, truncations, infos = self.env.step(actions)

        # Handle both vectorized and single environment cases
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            # Vectorized environment case
            for i in range(obs.shape[0]):
                if dones[i]:
                    state = (0, 50)  # Goal state
                else:
                    state = tuple(obs[i])
                self.visit_counts[state] += 1
        else:
            # Single environment case
            if dones:
                state = (0, 50)  # Goal state
            else:
                state = tuple(obs)
            self.visit_counts[state] += 1

        infos["visit_counts"] = dict(self.visit_counts)
        return obs, rewards, dones, truncations, infos

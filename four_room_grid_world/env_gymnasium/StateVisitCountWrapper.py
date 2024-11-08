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
            # One must be very careful because the SyncVectorEnv environment automatically
            # outputs the start state when the goal is reached
            # (i.e., the goal state is never actually returned by .step()).
            # Note that we do not count being in the start at the beginning of an episode as a state visit.
            if dones[i]:
                state = (0, 50)  # Goal state
            else:
                state = tuple(obs[i])  # Convert observation to a hashable type

            self.visit_counts[state] += 1

        infos["visit_counts"] = dict(self.visit_counts)

        return obs, rewards, dones, truncations, infos

import gym
from gym import spaces


class EnvTransformator(gym.ObservationWrapper):
    def __init__(self, env):
        super(EnvTransformator, self).__init__(env)
        # Set the observation space to just the agent's location
        self.observation_space = env.observation_space["agent"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._get_agent_location(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._get_agent_location(obs), reward, done, truncated, info

    def _get_agent_location(self, obs):
        return obs["agent"]

import os.path

import imageio
import numpy as np
import gym
from tqdm import tqdm

import four_room_grid_world.env.registration  # Do not remove this import
from four_room_grid_world.env.EnvTransformator import EnvTransformator


class QLearningAgent:
    """
    This agent is a baseline to test the dynamics of the environment.
    Implementation by ChatGPT (mostly).
    Learns to avoid hitting walls and the grid border.
    """

    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_prob=1.0, exploration_decay=0.99,
                 min_exploration_prob=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        self.min_exploration_prob = min_exploration_prob

        # Create a Q-table
        self.q_table = np.zeros((env.size, env.size, env.action_space.n))  # Size of state space x action space

    def choose_action(self, state):
        # Ensure exploration probability is above the minimum threshold
        effective_exploration_prob = max(self.min_exploration_prob, self.exploration_prob)

        # Epsilon-greedy action selection
        if np.random.rand() < effective_exploration_prob:
            return self.env.action_space.sample()  # Explore: select a random action
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit: select the best action based on Q-table

    def update_q_value(self, state, action, reward, next_state):
        # Update Q-value using the Q-learning formula
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        td_delta = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.learning_rate * td_delta

    def train(self, episodes):
        for episode in tqdm(range(episodes)):
            state, info = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                done = done or truncated

            # Decay exploration probability
            self.exploration_prob = max(self.min_exploration_prob, self.exploration_prob * self.exploration_decay)

            if episode % 100 == 0:
                self.test_agent(episode)

    def evaluate(self, episodes):
        total_reward = 0
        for episode in range(episodes):
            state, info = self.env.reset()
            done = False

            while not done:
                action = np.argmax(self.q_table[state[0], state[1]])
                next_state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                state = next_state
                done = done or truncated

        return total_reward / episodes

    def test_agent(self, episode):
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        frames = []  # To store rendered frames

        while not done:
            frame = self.env.render()
            frames.append(frame)

            action = np.argmax(self.q_table[state[0], state[1]])
            next_state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            state = next_state
            done = done or truncated

        print(f"Test Reward: {total_reward}")

        # Create a GIF from the collected frames
        if not os.path.exists("gifs_Q_learning"):
            os.mkdir("gifs_Q_learning")
        gif_path = f'gifs_Q_learning/episode_{episode}_reward_{total_reward}.gif'
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF to {gif_path}")


# Create the environment
env = gym.make("advtop/FourRoomGridWorld-v0", render_mode="rgb_array", max_episode_steps=1_000)
env = EnvTransformator(env)  # Wrap the environment to get only the agent's location

# Initialize the Q-learning agent
agent = QLearningAgent(env)

# Train the agent
agent.train(episodes=10_000)

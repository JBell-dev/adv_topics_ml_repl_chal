import gym
import imageio
import torch
import torch.nn as nn
import torch.optim as optim

import four_room_grid_world.env.registration  # Do not remove this import
from four_room_grid_world.env.EnvTransformator import EnvTransformator

# Create the environment
env = gym.make("advtop/FourRoomGridWorld-v0", render_mode="rgb_array", max_episode_steps=1_000)
env = EnvTransformator(env)  # Wrap the environment to get only the agent's location


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)  # Input layer
        self.fc2 = nn.Linear(128, env.action_space.n)  # Output layer for action probabilities

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)  # Output probabilities for each action
        return x


# Initialize the network and optimizer
policy_net = PolicyNetwork()
optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
gamma = 0.99  # Discount factor


# Collect trajectory function
def collect_trajectory(env, policy_net):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Ensure state tensor has batch dimension

        action_probs = policy_net(state_tensor).squeeze(0)  # Remove batch dimension for action sampling

        action = torch.multinomial(action_probs, 1).item()  # Sample action based on probabilities

        # TODO:
        # Not sure if truncated is required. Although if it is removed then the program gets stuck.
        # This suggests that "done" only indicates that the task was successfully archived while
        # "truncated" would mean the task was stopped after the defined number of maximum steps (max_episode_steps).
        next_state, reward, done, truncated, _ = env.step(action)  # Unpack updated return format
        done = done or truncated

        # Store data
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
    return states, actions, rewards


# Compute returns function
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)  # Insert at the start to get future rewards
    return returns


# Training loop using policy gradient
def train_policy_gradient(env, policy_net, optimizer, gamma, num_episodes=10_000):
    for episode in range(num_episodes):
        states, actions, rewards = collect_trajectory(env, policy_net)
        returns = compute_returns(rewards, gamma)

        # Normalize returns for stability
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute the policy loss
        policy_loss = []
        for i in range(len(states)):
            state_tensor = torch.FloatTensor(states[i])
            action_prob = policy_net(state_tensor)[actions[i]]
            # We want to maximize the probability but adam does minimize costs
            # Therefore, a minus is added
            policy_loss.append(-torch.log(action_prob) * returns[i])

        # Backpropagate and update weights
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        if episode % 100 == 0:  # Test every 100 episodes
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")
            print("Start test")
            test_agent(env, policy_net, episode)
            print("End test")


def test_agent(env, policy_net, episode):
    state, _ = env.reset()  # Updated to match new Gym API
    done = False
    total_reward = 0

    frames = []  # To store rendered frames

    while not done:
        # Render as RGB array and store the frame
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        action_probs = policy_net(state_tensor).squeeze(0)  # Remove batch dimension
        action = torch.argmax(action_probs).item()

        state, reward, done, truncated, _ = env.step(action)  # Unpack updated return format
        total_reward += reward
        done = done or truncated

    print(f"Test Reward: {total_reward}")

    # Create a GIF from the collected frames
    gif_path = f'gifs/episode_{episode}_reward_{total_reward}.gif'
    imageio.mimsave(gif_path, frames, fps=30)  # Adjust fps for desired speed
    print(f"Saved GIF to {gif_path}")


# Train the policy
train_policy_gradient(env, policy_net, optimizer, gamma)

import gym
import imageio
import torch
import torch.nn as nn
import torch.optim as optim

LATENT_SPACE_DIMENSIONALITY = 16

# Set up environment
env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=1000)


# Define the Actor Network
class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0] + LATENT_SPACE_DIMENSIONALITY, 128)
        self.fc2 = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


# Define the Critic Network
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0] + LATENT_SPACE_DIMENSIONALITY, 128)
        self.fc2 = nn.Linear(128, 1)  # Single output for value estimation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Define the Feature Extraction Network
class FeatureExtractionNetwork(nn.Module):
    def __init__(self):
        super(FeatureExtractionNetwork, self).__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, LATENT_SPACE_DIMENSIONALITY)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def sample_z():
    mu = torch.zeros(LATENT_SPACE_DIMENSIONALITY)
    Sigma = torch.eye(LATENT_SPACE_DIMENSIONALITY)
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma)
    sample = mvn.sample()
    return sample


# Update Feature Extraction Network
def update_feature_network(critic_net, feature_net, alpha=0.1):
    # Iterate through each parameter in the feature network and update
    for feature_param, critic_param in zip(feature_net.parameters(), critic_net.parameters()):
        # Combine weights: alpha * critic_weights + (1 - alpha) * feature_weights
        feature_param.data = alpha * critic_param.data + (1 - alpha) * feature_param.data


# Initialize the networks and optimizers
actor_net = ActorNetwork()
critic_net = CriticNetwork()
feature_net = FeatureExtractionNetwork()
actor_optimizer = optim.Adam(actor_net.parameters(), lr=0.01)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=0.01)
feature_optimizer = optim.Adam(feature_net.parameters(), lr=0.01)
gamma = 0.99  # Discount factor


# Collect trajectory function
def collect_trajectory(env, actor_net, feature_net):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False

    # Sample z
    z = sample_z()

    while not done:
        state_tensor = torch.FloatTensor(state).squeeze(0)

        # Get feature vector
        features = feature_net(state_tensor).squeeze(0)
        state_with_features = torch.concat((state_tensor, features))

        # TODO: this is wrong. Must feed state with z instead of state with latent vector
        action_probs = actor_net(state_with_features).squeeze(0)
        action = torch.multinomial(action_probs, 1).item()

        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated

        # Add noise to reward
        features /= torch.norm(features)
        F = torch.dot(features, z)
        reward += F

        states.append(state_with_features.tolist())
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
        returns.insert(0, G)
    return returns


# Training loop using Actor-Critic
def train_actor_critic(env, actor_net, critic_net, feature_net, actor_optimizer, critic_optimizer, feature_optimizer, gamma, num_episodes=5000):
    for episode in range(num_episodes):
        states, actions, rewards = collect_trajectory(env, actor_net, feature_net)
        returns = compute_returns(rewards, gamma)

        # Convert to tensors
        states_tensor = torch.Tensor(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize returns for stability
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # Update Critic
        critic_optimizer.zero_grad()
        value_estimates = critic_net(states_tensor).squeeze()  # Get value estimates for the states
        critic_loss = nn.functional.mse_loss(value_estimates, returns_tensor)  # Mean squared error loss
        critic_loss.backward()
        critic_optimizer.step()

        # Update Actor
        actor_optimizer.zero_grad()
        action_probs = actor_net(states_tensor)
        log_probs = torch.log(action_probs[range(len(actions_tensor)), actions_tensor])
        # Simply using aggregated rewards can lead to high variance in the training signal
        # because the rewards received can be noisy and depend heavily on the specific episode.
        # (returns_tensor-value_estimates) captures how much better (or worse) the action taken was
        # compared to the expected value of that state.
        actor_loss = -(log_probs * (returns_tensor - value_estimates.detach())).mean()  # Policy gradient loss
        actor_loss.backward()
        actor_optimizer.step()

        # Update Feature Network using linear combination of weights
        # update_feature_network(critic_net, feature_net, alpha=0.1)

        if episode % 100 == 0:  # Test every 100 episodes
            print(f"Episode {episode}, Total Reward: {sum(rewards)}")
            test_agent(env, actor_net, feature_net)


def test_agent(env, actor_net, feature_net):
    state, _ = env.reset()
    done = False
    total_reward = 0
    frames = []  # To store rendered frames

    while not done:
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.FloatTensor(state).squeeze(0)

        # Get feature vector
        features = feature_net(state_tensor).squeeze(0)
        state_with_features = torch.concat((state_tensor, features))

        action_probs = actor_net(state_with_features).squeeze(0)
        action = torch.argmax(action_probs).item()

        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        done = done or truncated

    print(f"Test Reward: {total_reward}")

    # Create a GIF from the collected frames
    gif_path = f'gifs_actor_critic_RLE/cartpole_test_reward_{total_reward}.gif'
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved GIF to {gif_path}")


# Train the actor-critic agent
train_actor_critic(env, actor_net, critic_net, feature_net, actor_optimizer, critic_optimizer, feature_optimizer, gamma)

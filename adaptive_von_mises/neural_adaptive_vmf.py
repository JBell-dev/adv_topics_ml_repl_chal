# This file implements the PPO algorithm with RLE as the exploration method
# The RLE intrinsic reward is based on features from the value function
# The policy network is a CNN



import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
import matplotlib.pyplot as plt
import seaborn as sns
import functools

import envpool
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from copy import deepcopy
import logging
import sys
import os
from datetime import datetime
import torch.nn.functional as F
import traceback
# At the start of your script
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Create log directory if it doesn't exist
log_dir = "/content/drive/MyDrive/rle/logs"
os.makedirs(log_dir, exist_ok=True)

# Setup logging with created directory
log_filename = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# Test logging
logging.info("Logging initialized successfully")


if os.environ.get("WANDB_MODE", "online") == "offline":
    from wandb_osh.hooks import TriggerWandbSyncHook
    trigger_sync = TriggerWandbSyncHook()
else:
    def dummy():
        pass
    trigger_sync = dummy


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="random-latent-exploration",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (log it on wandb)")
    parser.add_argument("--capture-video-interval", type=int, default=10,
        help="How many training updates to wait before capturing video")
    parser.add_argument("--gpu-id", type=int, default=0,
        help="ID of GPU to use")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Alien-v5", # NAME OF THE GAME
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=40000000, #I've tried to 10M, number of epochs. 
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=128,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networs")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--int-vf-coef", type=float, default=0.5,
        help="coefficient of the intrinsic value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--sticky-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, sticky action will be used")
    parser.add_argument("--normalize-ext-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, extrinsic rewards will be normalized")

    # Evaluation specific arguments
    parser.add_argument("--eval-interval", type=int, default=0,
        help="number of epochs between evaluations (0 to skip)")
    parser.add_argument("--num-eval-envs", type=int, default=32,
        help="the number of evaluation environments")
    parser.add_argument("--num-eval-episodes", type=int, default=32,
        help="the number of episodes to evaluate with")

    # RLE arguments
    parser.add_argument("--switch-steps", type=int, default=500,
        help="number of timesteps to switch the RLE network")
    parser.add_argument("--norm-rle-features", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True, help="if toggled, rle features will be normalized")
    parser.add_argument("--int-coef", type=float, default=0.01,
        help="coefficient of extrinsic reward")
    parser.add_argument("--ext-coef", type=float, default=1.0,
        help="coefficient of intrinsic reward")
    parser.add_argument("--int-gamma", type=float, default=0.99,
        help="Intrinsic reward discount rate")
    parser.add_argument("--feature-size", type=int, default=16,
        help="Size of the feature vector output by the rle network") #tau, 0 if no update
    parser.add_argument("--tau", type=float, default=0.005,
        help="The parameter for soft updating the rle network")
    parser.add_argument("--save-rle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, save the rle network at the end of training")
    parser.add_argument("--num-iterations-feat-norm-init", type=int, default=1,
        help="number of iterations to initialize the feature normalization parameters")

    parser.add_argument("--z-layer-init", type=str, default="ortho_1.41:0.0",
        help="z layer init")  # Options: "sparse_{sparsity}:{std}:{bias}", "ortho_{std}:{bias}"

    parser.add_argument("--local-dir", type=str, default="./results")
    parser.add_argument("--use-local-dir", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the local directory will be used")

    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Enable debug printing")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews, not_done=None):
        if not_done is None:
            if self.rewems is None:
                self.rewems = rews
            else:
                self.rewems = self.rewems * self.gamma + rews
            return self.rewems
        else:
            if self.rewems is None:
                self.rewems = rews
            else:
                mask = np.where(not_done == 1.0)
                self.rewems[mask] = self.rewems[mask] * self.gamma + rews[mask]
            return deepcopy(self.rewems)


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        
        # Update the episode returns and lengths
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        
        # Store the current values before resetting
        self.returned_episode_returns = self.episode_returns.copy()
        self.returned_episode_lengths = self.episode_lengths.copy()
        
        # Reset returns and lengths for environments that are done
        episode_done_mask = infos["terminated"] | infos["TimeLimit.truncated"]
        self.episode_returns = self.episode_returns * (1 - episode_done_mask)
        self.episode_lengths = self.episode_lengths * (1 - episode_done_mask)
        
        # Update info with per-environment returns and lengths
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        
        return observations, rewards, dones, infos


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def sparse_layer_init(layer, sparsity=0.1, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.sparse_(layer.weight, sparsity, std=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def create_layer_init_from_spec(spec: str):
    if spec.startswith("ortho"):
        params = spec.split("_")[1].split(":")
        print(f"Create ortho init with {params}")
        return functools.partial(layer_init,
                                 std=float(params[0]),
                                 bias_const=float(params[1]))
    elif spec.startswith("sparse"):
        params = spec.split("_")[1].split(":")
        print(f"Create sparse init with {params}")
        return functools.partial(sparse_layer_init,
                                 sparsity=float(params[0]),
                                 std=float(params[1]),
                                 bias_const=float(params[2]))


class Agent(nn.Module):
    def __init__(self, envs, rle_net):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(rle_net.feature_size, 448)),
            nn.ReLU(),
            layer_init(nn.Linear(448, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, reward, goal, action=None, deterministic=False):
        obs_hidden = self.network(x / 255.0)

        # goal is the goal vector
        # it is a tensor of shape (num_envs, feature_size)
        goal_hidden = self.goal_encoder(goal)
        hidden = obs_hidden + goal_hidden

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None and not deterministic:
            action = probs.sample()
        elif action is None and deterministic:
            action = probs.probs.argmax(dim=1)
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )
    def get_features(self, x):
        return self.network(x / 255.0)  # Returns shape (batch, 448)

    def get_value(self, x, reward, goal):
        obs_hidden = self.network(x / 255.0)

        # goal is the goal vector
        # it is a tensor of shape (num_envs, feature_size)
        goal_hidden = self.goal_encoder(goal)
        hidden = obs_hidden + goal_hidden

        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

class RLEModel(nn.Module):
    def __init__(self, feature_size, num_envs, device, total_timesteps):
        super().__init__()
        self.feature_size = feature_size  #  z vector
        self.device = device
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.hidden_size = 896
        
        # the idea is to use the lstm as an "encoding" of the states keeping the sequential influences of decisions.
        self.lstm = nn.LSTM(
            input_size=448,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # hidden states per environment
        #self.hidden_states = {i: None for i in range(num_envs)}
        
        # VMF parameter networks
        self.mu_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), 
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  
            nn.Linear(self.hidden_size // 2, feature_size)
        )

        self.kappa_net = nn.Sequential(
            nn.Linear(self.hidden_size + feature_size, self.hidden_size),  # Takes both lstm hidden and mu
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Softplus()
        )

        
        # Intrinsic reward prediction network
        self.reward_net = nn.Sequential(
            nn.Linear(448 + feature_size, self.hidden_size),  # Agent state + z
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), 
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  
            nn.Linear(self.hidden_size, 1)
        )
        
        # Trajectory storage for each environment
        self.env_trajectories = {i: {
            'states': [],
            'goals': [],
            'rewards': [],
            'returns': [],
            'hidden': None
        } for i in range(num_envs)}

        # just starting goal
        self.goals = F.normalize(torch.randn(num_envs, feature_size, device=device), dim=1)
        
        
        # general params
        self.min_kappa = 0.1
        self.max_kappa = 50.0
        self.switch_steps = 500
        self.num_steps_left = torch.full((num_envs,), self.switch_steps, dtype=torch.float32, device=device)
        self.switch_goals_mask = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
        # optimizer 
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.global_step = 0
        self.eps = 1e-8
        
    
        self.current_features = None
        self.current_obs = None


        self.alignment_coef = 0.25
        self.entropy_coef = 0.4
        self.reward_coef = 6.0
        self.kappa_coef = 1.0

        self.initial_entropy_coef = 0.6  
        self.final_entropy_coef = 0.01   

        self.return_ema = RunningMeanStd()  # Moving statistics for returns

        self.returns_history = deque(maxlen=100)  
        self.stagnation_window = 50
        self.stagnation_boost = 0.5
        self.kl_coef = 0.01

    def update_entropy_coef(self):
        """The idea is to explore more at the beginning and gradually exploit more such that the exploration is more centered
        in the beginning"""
        progress = self.global_step / self.total_timesteps
        quadratic_progress = progress ** 2
        self.entropy_coef = self.initial_entropy_coef * (1 - quadratic_progress) + self.final_entropy_coef * quadratic_progress
        self.entropy_coef = max(self.entropy_coef, self.final_entropy_coef)

    def init_hidden(self, batch_size=1):
        """Initialize LSTM hidden states - we need to reset the hidden after each episode finish to not biase the next one!
        However, can be extended to keep the cell state per environment.
        """
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=self.device),
            torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        )
        
    def process_state_sequence(self, states, hidden_state=None):
        """Process sequence of agent states through LSTM"""
        if hidden_state is None:
            hidden_state = self.init_hidden(states.size(0))
            
        # we encode the trajectory
        lstm_out, new_hidden = self.lstm(states, hidden_state)
        
        # we predict mu first since we need it for kappa
        mu = self.mu_net(lstm_out)
        mu = F.normalize(mu, dim=-1)
        
        # Combine lstm output and mu for kappa prediction
        combined_features = torch.cat([lstm_out, mu], dim=-1)
        kappa = self.kappa_net(combined_features)
        kappa = torch.clamp(kappa, self.min_kappa, self.max_kappa)
        
        return mu, kappa, new_hidden


    def random_VMF(self, mu, kappa, size=None):
        """
        Von Mises-Fisher distribution - base on paper but accomodated for pytorch such that we can backpropagate 
        """
        device = mu.device
        
        # if needed we need to put in tensors
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu, device=device)
        if not torch.is_tensor(kappa):
            kappa = torch.tensor(kappa, device=device)
                
        # dim checker
        n = 1 if size is None else np.prod(size)
        shape = () if size is None else tuple(np.ravel(size))
        

        mu = F.normalize(mu, dim=-1)  # mu as unit vector
        d = mu.shape[-1]  #we need the last dimension size for feature dimension
        
        # if k=0 then we sample of the unit sphere
        if torch.all(kappa == 0):
            z = torch.randn(n, d, device=device)
            return F.normalize(z, dim=1)
        
        # Step 1: Get samples perpendicular to mu
        z = torch.randn(n, d, device=device)
        z = F.normalize(z, dim=1)
        mu_expanded = mu.unsqueeze(0) if mu.dim() == 1 else mu
        z = z - (z * mu_expanded).sum(dim=1, keepdim=True) * mu_expanded
        z = F.normalize(z, dim=1)
        
        # Step 2: Sample angles using Wood's method
        cos = self._random_VMF_cos_torch(d, kappa, n)
        sin = torch.sqrt(1 - cos**2)
        
        # Step 3: Combine to get points on sphere
        x = z * sin.unsqueeze(1) + cos.unsqueeze(1) * mu_expanded
        return F.normalize(x, dim=1)

    def _random_VMF_cos_torch(self, d, kappa, n):
        """Modified Wood's rejection sampling method"""
        device = kappa.device
        
        #  d is at least 3 to avoid zero parameters -> almost all features are higher anyway.
        d = max(3, d) 
        
        # in case we pass a batch of kappas
        if kappa.dim() > 0:
            kappa = kappa.view(-1)  
            b = (d - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (d - 1)**2))
            x0 = (1 - b) / (1 + b)
            c = kappa * x0 + (d - 1) * torch.log(1 - x0**2)
        else:
            b = (d - 1) / (2 * kappa + torch.sqrt(4 * kappa**2 + (d - 1)**2))
            x0 = (1 - b) / (1 + b)
            c = kappa * x0 + (d - 1) * torch.log(1 - x0**2)
        
        
        out = torch.empty(n, device=device) # output vector
        
        # batches processing:
        found = 0
        while found < n:
            m = min(n - found, int((n - found) * 1.5))
            
            alpha = (d-1)/2
            if alpha <= 0:
                alpha = 1.0  
                
            # Sample from Beta distribution
            beta_dist = torch.distributions.Beta(alpha, alpha)
            z = beta_dist.sample((m,)).to(device)
            
            t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            # Acceptance test
            test = kappa * t + (d - 1) * torch.log(1 - x0 * t) - c
            accept = test >= -torch.distributions.Exponential(1.0).sample((m,)).to(device)
            
            num_accept = torch.sum(accept)
            if num_accept > 0:
                out[found:found + num_accept] = t[accept][:min(num_accept, n - found)]
                found += num_accept
        
        return out

    def compute_reward(self, agent_states, z):
        """Predict intrinsic reward for state-goal pairs"""
        combined = torch.cat([agent_states, z], dim=-1)
        return self.reward_net(combined).squeeze(-1)


    def add_to_trajectory(self, env_idx, state, goal, reward, ret):
        """Add step maintaining gradient flow"""
        with torch.set_grad_enabled(True):
            # important! we store without detaching to maintain graph
            self.env_trajectories[env_idx]['states'].append(state)
            self.env_trajectories[env_idx]['goals'].append(goal)
            self.env_trajectories[env_idx]['rewards'].append(torch.tensor(reward, device=self.device))
            self.env_trajectories[env_idx]['returns'].append(torch.tensor(ret, device=self.device))


    def process_trajectory(self, env_idx):
        """Process whole trajectory at once"""
        trajectory = self.env_trajectories[env_idx] #take the specific env data
        if len(trajectory['states']) < 2:
            return None

        self.optimizer.zero_grad()

        # shaping:
        states_seq = torch.stack(trajectory['states']).unsqueeze(0)  # Shape: [1, seq_length, feature_size]
        rewards_seq = torch.stack(trajectory['rewards'])  # Shape: [seq_length]
        returns_seq = torch.stack(trajectory['returns'])  # Shape: [seq_length]
        goals_seq = torch.stack(trajectory['goals'])  # Shape: [seq_length, feature_size]

        # lstm processing
        hidden_state = trajectory['hidden']
        mu, kappa, new_hidden_state = self.process_state_sequence(states_seq, hidden_state)

        # updated global step for the entropy 
        self.global_step += len(trajectory['states'])
        self.update_entropy_coef() #updated entropy coef

        loss, metrics = self.update_from_trajectory(
            trajectory,
            mu.squeeze(0), 
            kappa.squeeze(0)
        )

        with torch.no_grad():
            kappa_values = kappa.squeeze(0)
            metrics['kappa_mean'] = kappa_values.mean().item()
            metrics['kappa_std'] = kappa_values.std().item()
            metrics['kappa_min'] = kappa_values.min().item()
            metrics['kappa_max'] = kappa_values.max().item()
            metrics['entropy_coef'] = self.entropy_coef

        # backprop! 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # new hidden state
        self.env_trajectories[env_idx]['hidden'] = (new_hidden_state[0].detach(), new_hidden_state[1].detach())

        return metrics


    def step(self, next_done, next_ep_done, env_rewards, returns, agent):
        """Environment interaction step"""
        agent_states = agent.get_features(self.current_obs) #we get the agent preprocess features -> 448 dim , he should learn the representation of the space since he is running through it! 
        
        self.current_features = agent_states
        self.current_env_rewards = torch.as_tensor(env_rewards, device=self.device)
        self.current_returns = torch.as_tensor(returns, device=self.device)
        
        # Update switch mask: criteria of paper, end episode, termination or resampling
        self.switch_goals_mask = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.switch_goals_mask[next_done.bool()] = 1.0
        self.num_steps_left -= 1
        self.switch_goals_mask[self.num_steps_left <= 0] = 1.0
        
        # Update goals if needed
        if self.switch_goals_mask.any():
            with torch.no_grad():  # No gradients needed for goal switching, we do this later with the whole trajectory since for the loss we want the returns and rewards t + 1
                mu, kappa, _ = self.process_state_sequence(agent_states.unsqueeze(1))
                new_goals = self.random_VMF(mu[:, 0], kappa[:, 0])
                self.goals = self.goals * (1 - self.switch_goals_mask.unsqueeze(1)) + \
                            new_goals * self.switch_goals_mask.unsqueeze(1)
                self.num_steps_left[self.switch_goals_mask.bool()] = self.switch_steps
        
        return self.switch_goals_mask

    def update_from_trajectory(self, trajectory, mu, kappa):
        """
        Update network using trajectory with proper temporal alignment.
        mu[t], kappa[t] -> z[t] (with states) -> reward[t+1], return[t+1]
        """
        # we shift rewards and returns one step back to align with their causing states/actions
        aligned_rewards = torch.stack(trajectory['rewards'][1:])  # and we convert them to tensor
        aligned_returns = torch.stack(trajectory['returns'][1:])  

        mu_causes = mu[:-1]
        kappa_causes = kappa[:-1]
        states_causes = torch.stack(trajectory['states'][:-1])  # Convert to tensor

        # Update return statistics with new returns (use detached values)
        with torch.no_grad():
            self.return_ema.update(aligned_returns.cpu().numpy())

        # normalized returns 
        returns_mean = torch.tensor(self.return_ema.mean, device=self.device)
        returns_std = torch.tensor(np.sqrt(self.return_ema.var), device=self.device) 
        normalized_returns = (aligned_returns - returns_mean) / (returns_std + 1e-8)

        with torch.no_grad():
            return_uncertainty = np.sqrt(self.return_ema.var) / (self.return_ema.mean + 1e-8)
            kappa_weight = 1.0 / (1.0 + return_uncertainty + 1e-8)

        # Convert to range [0,1] using sigmoid, maintaining gradients
        scaled_returns = torch.sigmoid(normalized_returns)
        
        # Target kappa 
        target_kappa = self.min_kappa + (self.max_kappa - self.min_kappa) * scaled_returns

        # we get z for each timestep using corresponding parameters
        z = self.random_VMF(mu_causes, kappa_causes)

        # we get the intrinsic rewards
        predicted_rewards = self.compute_reward(states_causes, z)

        # After computing aligned_returns, store the latest return
        self.returns_history.append(aligned_returns[-1].item())
        # Get stagnation boost for entropy
        stagnation_factor = self.compute_stagnation_boost()

        # and now we get the losses
        alignment_loss = -(torch.sum(mu_causes * z, dim=1) * aligned_returns).mean() #notice that if high return we want to align our mu with that z that give the high return -> guide mus to high return z! 
        entropy_loss = -self.compute_vmf_log_entropy(mu_causes, kappa_causes).mean() * stagnation_factor # we want to maximize the entropy of the distribution of the mus -> for exploration
        reward_loss = F.smooth_l1_loss(predicted_rewards, aligned_rewards) # we want that in the end, the intrinsic rewards reflect the reality so we can predict with accuracy what the agent is going to find
        #F.mse_loss(predicted_rewards, aligned_rewards)

        # Compute kappa loss with adaptive weighting
        kappa_loss = F.smooth_l1_loss(kappa_causes, target_kappa.unsqueeze(-1))

        

        ###for plotting mu:
        # angle between mu and e1
        e1 = torch.zeros_like(mu_causes)
        e1[:, 0] = 1.0  # base e1

        # dot product between mu and e1
        dot_product = (mu_causes * e1).sum(dim=1)
        dot_product = dot_product.clamp(-1.0, 1.0)  

        # angles in radians and convert to degrees
        angle_rad = torch.acos(dot_product)
        angle_deg = torch.rad2deg(angle_rad)

        # stats
        angle_mean = angle_deg.mean().item()
        angle_min = angle_deg.min().item()
        angle_max = angle_deg.max().item()
        ####


        total_loss = alignment_loss * self.alignment_coef + entropy_loss * self.entropy_coef + reward_loss * self.reward_coef + kappa_loss * kappa_weight * self.kappa_coef



        return total_loss, {
            'alignment_loss': alignment_loss.item() * self.alignment_coef,
            'entropy_loss': entropy_loss.item() * self.entropy_coef,
            'reward_loss': reward_loss.item() * self.reward_coef,
            'kappa_loss': kappa_loss.item() * self.kappa_coef * kappa_weight,
            'angle_mean': angle_mean,
            'scaled_returns': scaled_returns.mean().item(),
            'target_kappa': target_kappa.mean().item(),
            'normalized_returns': normalized_returns.mean().item(),
            'angle_min': angle_min,
            'angle_max': angle_max,
            'kappa_weight': kappa_weight,
            'stagnation_factor': stagnation_factor
        }


    def compute_intrinsic_reward(self, agent_states, current_goals):
        """Compute intrinsic reward for RLE, no gradient!"""
        with torch.no_grad():  
            reward = self.compute_reward(agent_states, current_goals)
            return reward



    def compute_log_modified_bessel(self, v, x):
        """Compute log of modified Bessel function with tensor operations.
        this is the exact formula of the bessel that we could reconstruct 
        thanks to a property of the function that allow us to write 
        higher orders as function of previous two! 
        """
        eps = 1e-6  # for numerical issues 
        
        x = torch.as_tensor(x, device=self.device)
        v = torch.as_tensor(v, device=self.device)
        
        if torch.all(x < eps):
            return torch.log(torch.tensor(eps, device=self.device))
        
        #given the int
        v_is_int = torch.all((v - v.round()).abs() < eps)
        
        if v_is_int:  
            v_int = int(v.round().item())
            if v_int == 0:
                return torch.log(torch.special.i0(x) + eps)
            elif v_int == 1:
                return torch.log(torch.special.i1(x) + eps)
            
            # For higher orders, use recurrence relation
            log_i_nm1 = torch.log(torch.special.i0(x) + eps)
            log_i_n = torch.log(torch.special.i1(x) + eps)
            
            for n in range(1, v_int):
                log_term = torch.log(2 * n / x + eps)
                log_i_np1 = torch.logaddexp(log_i_nm1, log_term + log_i_n)
                log_i_nm1 = log_i_n
                log_i_n = log_i_np1
            
            return log_i_n
        
        # For non-integer orders, use series expansion however we have integer orders here! 
        k = torch.arange(20, dtype=x.dtype, device=self.device)
        log_terms = (2 * k + v) * torch.log(x / 2) - (
            torch.lgamma(k + 1) + torch.lgamma(k + v + 1)
        )
        return torch.logsumexp(log_terms, dim=0)

    def compute_vmf_log_entropy(self, mu, kappa):
        """Compute the entropy of the von Mises-Fisher (vMF) distribution."""
        dim = mu.shape[-1]
        
        if not isinstance(kappa, torch.Tensor):
            kappa = torch.tensor(kappa, device=self.device, requires_grad=True)
        elif not kappa.requires_grad:
            kappa = kappa.clone().detach().to(self.device).requires_grad_(True)
        else:
            kappa = kappa.to(self.device)
        
        kappa = torch.clamp(kappa, min=self.min_kappa, max=self.max_kappa)
        
        order = dim / 2.0 - 1  # v = p/2 - 1
        
        
        log_cp = (
            order * torch.log(kappa)
            - (dim / 2.0) * torch.log(torch.tensor(2.0 * np.pi, device=self.device))
            - self.compute_log_modified_bessel(order, kappa)
        )
        
        log_bessel_v = self.compute_log_modified_bessel(order + 1, kappa)
        log_bessel_v_minus_1 = self.compute_log_modified_bessel(order, kappa)
        log_ap_kappa = log_bessel_v - log_bessel_v_minus_1
        ap_kappa = torch.exp(log_ap_kappa)
        
        entropy = -log_cp - kappa * ap_kappa
        
        return entropy

    def compute_stagnation_boost(self):
        if len(self.returns_history) < self.stagnation_window:
            return 1.0
            
        returns = np.array(list(self.returns_history))
        
        rolling_mean = np.convolve(returns, np.ones(10)/10, mode='valid')
        
        # if recent means are too similar, we're stagnating
        recent_variation = np.std(rolling_mean[-20:])
        baseline_variation = np.std(returns)
        
        stagnation_measure = 1.0 - (recent_variation / (baseline_variation + 1e-8))
        stagnation_measure = np.clip(stagnation_measure, 0, 1)
        
        return 1.0 + self.stagnation_boost * stagnation_measure

    def compute_vmf_log_kl(self, mu0, kappa0, mu1, kappa1):
        """Compute KL divergence between two VMF distributions using log space"""
        try:
            kappa0 = torch.as_tensor(kappa0, device=self.device)
            kappa1 = torch.as_tensor(kappa1, device=self.device)
            
            kappa0 = torch.clamp(kappa0, min=0, max=self.max_kappa)
            kappa1 = torch.clamp(kappa1, min=0, max=self.min_kappa)
            
            # Get dimension from feature size
            dim = mu0.shape[-1]
            order = dim/2.0
            
            # log normalization constants
            log_cp0 = (
                (order - 1) * torch.log(kappa0) 
                - (dim/2.0) * torch.log(torch.tensor(2.0 * np.pi, device=self.device))
                - self.compute_log_modified_bessel(order - 1, kappa0)
            )
            log_cp1 = (
                (order - 1) * torch.log(kappa1)
                - (dim/2.0) * torch.log(torch.tensor(2.0 * np.pi, device=self.device))
                - self.compute_log_modified_bessel(order - 1, kappa1)
            )
            
            # log of mean resultant length
            log_bessel_v = self.compute_log_modified_bessel(order, kappa0)
            log_bessel_v_minus_1 = self.compute_log_modified_bessel(order-1, kappa0)
            log_ap_kappa0 = log_bessel_v - log_bessel_v_minus_1
            ap_kappa0 = torch.exp(log_ap_kappa0)
            
            # cosine similarity with numerical stability
            cos_theta = torch.sum(mu0 * mu1, dim=-1)  
            cos_theta = torch.clamp(cos_theta, -1 + 0.000001, 1 - 0.000001)
            
            # KL divergence
            kl = (log_cp1 - log_cp0) + kappa1 * ap_kappa0 * cos_theta - kappa0 * ap_kappa0
            
            return torch.abs(kl)
            
        except Exception as e:
            logging.warning(f"Log KL computation failed: {str(e)}")
            return torch.tensor(0.000001, device=self.device).expand(len(kappa0))





   


        
class VideoRecorder(object):

    def __init__(self,
                 max_buffer_size: int = 27000 * 5,  # Avoid out-of-memory (OOM) issues. Flush video when the buffer exceeds this size
                 local_dir: str = "./results",
                 use_wandb: bool = False) -> None:
        self.use_wandb = use_wandb
        self.local_dir = local_dir
        self.max_buffer_size = max_buffer_size
        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.rewards = deque(maxlen=max_buffer_size)
        self.int_rewards = deque(maxlen=max_buffer_size)
        self.episode_count = 0
        self.fig = plt.figure()  # create a figure object for plotting rle statistics

    def record(self, frames: np.ndarray, rewards: float, int_reward_info: dict, global_step: int):
        self.frame_buffer.append(np.expand_dims(frames, axis=0).astype(np.uint8))  # Expand dim for concatenation later
        self.rewards.append(rewards)
        self.int_rewards.append(int_reward_info["int_rewards"])

    def reset(self):
        self.frame_buffer.clear()  # Reset the buffer
        self.rewards.clear()
        self.int_rewards.clear()
        self.episode_count += 1

    def flush(self, global_step: int, caption: str = ""):
        if len(self.frame_buffer) <= 0:  # If frame buffer is empty, do nothing
            return
        if len(caption) <= 0:
            caption = f"episode-{self.episode_count}-score-{np.stack(self.rewards).sum()}"

        video_array = np.concatenate(self.frame_buffer, axis=0)
        video_array = video_array[:, None, ...]  # Add channel axis

        save_path = os.path.join(self.local_dir, str(self.episode_count), str(caption))
        print(f"Log frames and rewards at {save_path}")
        if args.use_local_dir:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "frames.npy"), video_array)
            np.save(os.path.join(save_path, "rewards.npy"), np.stack(self.rewards))
            np.save(os.path.join(save_path, "int_rewards.npy"), np.stack(self.int_rewards))

        if self.use_wandb:
            wandb.log({"media/video": wandb.Video(video_array, fps=30, caption=str(caption))}, step=global_step)
            # Log task rewards
            task_lineplot = sns.lineplot(np.stack(self.rewards))
            log_data = wandb.Image(self.fig)
            wandb.log({"media/task_rewards": log_data}, step=global_step)
            plt.clf()

            # Log intrinsic rewards
            int_reward_lineplot = sns.lineplot(np.stack(self.int_rewards))
            log_data = wandb.Image(self.fig)
            wandb.log({"media/int_reward": log_data}, step=global_step)
            plt.clf()

        self.reset()


class VideoRecordScoreCondition:

    score_thresholds = [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1800,
        2500,
        3000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
        90000,
        100000,
        np.inf,
    ]

    def __init__(self) -> None:
        self.has_recorded = pd.DataFrame({"value": [False] * (len(self.score_thresholds) - 1)},
            index=pd.IntervalIndex.from_breaks(self.score_thresholds, closed='left'))

        print("Record score intervals: ", self.score_thresholds)

    def __call__(self, score: float, global_step: int):
        if not self.has_recorded.iloc[self.has_recorded.index.get_loc(score)]["value"]:
            print(f"Record the first video with score {score}")
            self.has_recorded.iloc[self.has_recorded.index.get_loc(score)] = True
            return True
        return False


class VideoStepConditioner:

    def __init__(self, global_step_interval: int) -> None:
        self.global_step_interval = global_step_interval
        self.last_global_step = 0

    def __call__(self, score: float, global_step: int):
        if global_step - self.last_global_step >= self.global_step_interval:
            self.last_global_step = global_step
            return True
        return False





if __name__ == "__main__":
    fig = plt.figure()  # create a figure object for plotting rle statistics
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device: ", device)

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        max_episode_steps=int(108000 / 4),
        seed=args.seed,
        repeat_action_probability=0.25,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    rle_output_size = args.feature_size  # NOTE: this is not used
    num_actions = envs.single_action_space.n
    rle_network = RLEModel(
        feature_size=args.feature_size,
        num_envs=args.num_envs,
        device=device,
        total_timesteps=args.total_timesteps
    ).to(device)
    rle_feature_size = rle_network.feature_size

    agent = Agent(envs, rle_network).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    int_reward_rms = RunningMeanStd()
    int_discounted_reward = RewardForwardFilter(args.int_gamma)
    ext_reward_rms = RunningMeanStd()
    ext_discounted_reward = RewardForwardFilter(args.gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    #next_obss = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    goals = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rle_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    prev_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rle_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    raw_rle_feats = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)
    rle_feats = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)

    # At the start of training, initialize separate return trackers per env:
    env_episode_returns = torch.zeros(args.num_envs, device=device)

    # Logging setup
    avg_returns = deque(maxlen=128)
    avg_ep_lens = deque(maxlen=128)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    video_filenames = set()

    video_recorder = VideoRecorder(local_dir=args.local_dir, use_wandb=args.track & args.capture_video)
    video_record_conditioner = VideoStepConditioner(global_step_interval=int(args.capture_video_interval))
    is_early_stop = False

    ############################################################
    completed_trajectories = []

    # Initialize episode returns tracker
    env_episode_returns = torch.zeros(args.num_envs, device=device)

    #torch.autograd.set_detect_anomaly(True)
    for update in range(1, num_updates + 1):
        prev_rewards[0] = rle_rewards[-1] * args.int_coef  # last step of prev rollout
        it_start_time = time.time()

        # current obs for RLE
        rle_network.current_obs = next_obs

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow



        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done  # done is True if the episode ended in the previous step
            rle_dones[step] = rle_network.switch_goals_mask  # rle_done is True if the goal is switched in the previous step
            goals[step] = rle_network.goals

            rle_obs = next_obs.clone().float()
            rle_obs_norm = rle_obs.pow(2).sum(dim=(1, 2, 3)).sqrt()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(
                    next_obs, prev_rewards[step], goals[step], deterministic=False
                )
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1).clone()
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            #  experience to trajectories
            agent_features = agent.get_features(next_obs)
            for env_idx in range(args.num_envs):
                rle_network.add_to_trajectory(
                    env_idx=env_idx,
                    state=agent_features[env_idx].detach(),
                    goal=rle_network.goals[env_idx].clone(),
                    reward=info["reward"][env_idx],
                    ret=info["r"][env_idx]
                )
            
            for idx, d in enumerate(done):
                if info["terminated"][idx] or info["TimeLimit.truncated"][idx]:
                    episode_return = info["r"][idx].item()
                    avg_returns.append(episode_return)
                    avg_ep_lens.append(info["l"][idx])

                    # Process trajectory for this environment
                    with torch.set_grad_enabled(True):  # keeping gradients for bptt ! 
                        metrics = rle_network.process_trajectory(idx)
                        if metrics is not None and args.track:
                            wandb.log({
                                "rle/alignment_loss": metrics['alignment_loss'],
                                "rle/entropy_loss": metrics['entropy_loss'],
                                "rle/reward_loss": metrics['reward_loss'],
                                "rle/total_loss": sum(metrics.values()),
                                "rle/kappa_mean": metrics['kappa_mean'],
                                "rle/kappa_std": metrics['kappa_std'],
                                "rle/kappa_min": metrics['kappa_min'],
                                "rle/kappa_max": metrics['kappa_max'],
                                "rle/entropy_coef": metrics['entropy_coef'],
                                "rle/angle_mean": metrics['angle_mean'],
                                "rle/kappa_weight": metrics['kappa_weight'],
                                "rle/kappa_loss": metrics['kappa_loss'],
                                "rle/scaled_returns": metrics['scaled_returns'],
                                "rle/target_kappa": metrics['target_kappa'],
                                "rle/normalized_returns": metrics['normalized_returns'],
                                "rle/angle_min": metrics['angle_min'],
                                "rle/angle_max": metrics['angle_max'],
                                "rle/stagnation_factor": metrics['stagnation_factor']
                            }, step=global_step)

                    # Reset environment trajectory after processing
                    rle_network.env_trajectories[idx] = {
                        'states': [],
                        'goals': [],
                        'rewards': [],
                        'returns': [],
                        'hidden': rle_network.init_hidden()
                    }

                    env_episode_returns[idx] = 0
                    
                    #  episode info
                    if args.track:
                        wandb.log({
                            "training/episode_return": episode_return,
                            "training/episode_length": info["l"][env_idx],
                            "training/running_return_mean": np.mean(list(avg_returns)),
                            "training/running_length_mean": np.mean(list(avg_ep_lens))
                        }, step=global_step)

            # Compute intrinsic rewards for current states
            rle_reward = rle_network.compute_intrinsic_reward(
                agent.get_features(next_obs),
                rle_network.goals
            )
            rle_rewards[step] = rle_reward.clone()

            # Update goals (if required)
            switch_mask = rle_network.step(
                next_done=next_done,
                next_ep_done=info["terminated"] | info["TimeLimit.truncated"],
                returns=info["r"],
                env_rewards=info["reward"],
                agent=agent
            )

            # Store info for video recording if needed
            int_reward_info = {
                "int_rewards": rle_rewards[step, 0].cpu().numpy(),
            }

            if args.capture_video:
                video_recorder.record(
                    obs[step][0, 3, ...].cpu().numpy().copy(),
                    info["reward"][0],
                    int_reward_info,
                    global_step=global_step
                )

            # Update prev rewards for next step
            if step < args.num_steps - 1:
                prev_rewards[step + 1] = rle_rewards[step].clone() * args.int_coef

            # In main loop, after step
            if step % 32 == 0:
                torch.cuda.empty_cache()

        # After collecting steps, normalize rewards if needed
        not_dones = (1.0 - dones).cpu().data.numpy()
        rewards_cpu = rewards.cpu().data.numpy()
        rle_rewards_cpu = rle_rewards.cpu().data.numpy()

        raw_rewards = rewards.clone() #added for debugging 
        if args.normalize_ext_rewards:
            ext_reward_per_env = np.array([
                ext_discounted_reward.update(rewards_cpu[i], not_dones[i]) 
                for i in range(args.num_steps)
            ])
            ext_reward_rms.update(ext_reward_per_env.flatten())
            rewards /= np.sqrt(ext_reward_rms.var)

        # Normalize random rewards
        rle_not_dones = (1.0 - rle_dones).cpu().data.numpy()
        if True:
            rle_reward_per_env = np.array([
                int_discounted_reward.update(rle_rewards_cpu[i], rle_not_dones[i]) 
                for i in range(args.num_steps)
            ])
            int_reward_rms.update(rle_reward_per_env.flatten())
            rle_rewards /= np.sqrt(int_reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs, prev_rewards[step], goals[step])
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(rle_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0 - rle_network.switch_goals_mask
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0 - rle_dones[t + 1]
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = rle_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        #b_next_obs = next_obss.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_goals = goals.reshape((-1, rle_feature_size))
        b_dones = dones.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_int_values = int_values.reshape(-1)
        #b_raw_rle_feats = raw_rle_feats.reshape((-1, rle_feature_size))
        b_prev_rewards = prev_rewards.reshape(-1)

        # Update rms of the rle network
        #if args.norm_rle_features:
            #rle_network.update_rms(b_raw_rle_feats.cpu().numpy())

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_prev_rewards[mb_inds],
                    b_goals[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss * args.vf_coef + int_v_loss * args.int_vf_coef
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()

                # for param, target_param in zip(agent.network.parameters(), rle_network.rle_net.parameters()):
                #     target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        it_end_time = time.time()


        if args.eval_interval != 0 and update % args.eval_interval == 0:
            print(f"Evaluating at step {update}...")
            eval_start_time = time.time()
            eval_scores = []
            eval_ep_lens = []
            # Create eval envs
            eval_envs = envpool.make(
                args.env_id,
                env_type="gym",
                num_envs=args.num_eval_envs,  # keep this small to save memory
                episodic_life=True,
                reward_clip=True,
                max_episode_steps=int(108000 / 4),
                seed=args.seed,
                repeat_action_probability=0.25,
            )
            eval_envs.num_envs = args.num_eval_envs
            eval_envs.single_action_space = eval_envs.action_space
            eval_envs.single_observation_space = eval_envs.observation_space
            eval_envs = RecordEpisodeStatistics(eval_envs)

            # Evaluate the agent
            eval_obs = torch.Tensor(eval_envs.reset()).to(device)
            eval_done = torch.zeros(args.num_eval_envs).to(device)

            # Create the goal vector
            eval_goal = rle_network.sample_goals(args.num_eval_envs,agent).to(device)

            # Goal switching mechanism
            num_steps_left = args.switch_steps * torch.ones(args.num_eval_envs).to(device)
            switch_goals_mask = torch.zeros(args.num_eval_envs).to(device)

            # Initialize the prev reward vector to 0
            eval_prev_reward = torch.zeros((args.num_eval_envs,)).to(device)

            # Rollout the environments until the number of completed episodes is equal to the number of evaluation environments
            while len(eval_scores) < args.num_eval_episodes:
                # Compute the obs before stepping
                eval_rle_obs = eval_obs.clone().float()

                # Sample actions from the policy
                with torch.no_grad():
                    eval_action, _, _, _, _ = agent.get_action_and_value(
                        eval_obs, eval_prev_reward, eval_goal, deterministic=True
                    )
                eval_obs, eval_reward, eval_done, eval_info = eval_envs.step(eval_action.cpu().numpy())
                eval_reward = torch.tensor(eval_reward).to(device).view(-1)
                eval_obs, eval_done = torch.Tensor(eval_obs).to(device), torch.Tensor(eval_done).to(device)  # Removed extra )

                eval_rle_next_obs = eval_obs.clone().float()

                for idx, d in enumerate(eval_done):
                    if eval_info["terminated"][idx] or eval_info["TimeLimit.truncated"][idx]:
                        eval_scores.append(eval_info["r"][idx])
                        eval_ep_lens.append(eval_info["elapsed_step"][idx])

                # Goal switching mechanism --> switch_goals_mask is 1 when num_steps_left is 0 or when a life ends
                switch_goals_mask = torch.zeros(args.num_eval_envs).to(device)
                num_steps_left -= 1
                switch_goals_mask[num_steps_left == 0] = 1.0
                switch_goals_mask[eval_done.bool()] = 1.0

                new_goals = rle_network.sample_goals(args.num_eval_envs,agent).to(device)
                eval_goal = eval_goal * (1 - switch_goals_mask.unsqueeze(1)) + new_goals * switch_goals_mask.unsqueeze(1)

                # update the num_steps_left
                num_steps_left[switch_goals_mask.bool()] = args.switch_steps

            eval_envs.close()
            eval_end_time = time.time()

            print(f"Evaluation finished in {eval_end_time - eval_start_time} seconds")
            print(f"Step {update}: game score: {np.mean(eval_scores)}")

            eval_data = {}
            eval_data["eval/score"] = np.mean(eval_scores)
            eval_data["eval/min_score"] = np.min(eval_scores)
            eval_data["eval/max_score"] = np.max(eval_scores)
            eval_data["eval/ep_len"] = np.mean(eval_ep_lens)
            eval_data["eval/min_ep_len"] = np.min(eval_ep_lens)
            eval_data["eval/max_ep_len"] = np.max(eval_ep_lens)
            eval_data["eval/num_episodes"] = len(eval_scores)
            eval_data["eval/time"] = eval_end_time - eval_start_time


            # Add VMF-specific metrics to logging
            data.update({
                "vmf/kappa": rle_network.current_kappa,
                "vmf/success_rate": len(rle_network.success_memory) / rle_network.success_memory.maxlen,
                "vmf/memory_size": len(rle_network.success_memory),
                "vmf/avg_success_return": np.mean(list(rle_network.returns_memory)) if rle_network.returns_memory else 0,
            })


            if args.track:
                wandb.log(eval_data, step=global_step)
                trigger_sync()

        data = {}
        data["charts/iterations"] = update
        data["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
        data["losses/ext_value_loss"] = ext_v_loss.item()
        data["losses/int_value_loss"] = int_v_loss.item()
        data["losses/policy_loss"] = pg_loss.item()
        data["losses/entropy"] = entropy_loss.item()
        data["losses/old_approx_kl"] = old_approx_kl.item()
        data["losses/clipfrac"] = np.mean(clipfracs)
        # data["losses/explained_ext_var"] = np.mean(explained_ext_var)
        # data["losses/explained_int_var"] = np.mean(explained_int_var)
        data["losses/approx_kl"] = approx_kl.item()
        data["losses/all_loss"] = loss.item()
        data["charts/SPS"] = int(global_step / (time.time() - start_time))

        data["rewards/rewards_mean"] = rewards.mean().item() #avg over the rewards in the 128 steps for the 32 envs 
        data["rewards/rewards_max"] = rewards.max().item()
        data["rewards/rewards_min"] = rewards.min().item()
        data["rewards/intrinsic_rewards_mean"] = rle_rewards.mean().item()
        data["rewards/intrinsic_rewards_max"] = rle_rewards.max().item()
        data["rewards/intrinsic_rewards_min"] = rle_rewards.min().item()

        # Log raw game rewards before normalization TO DEBUG
        data["rewards/raw_game_rewards_mean"] = raw_rewards.mean().item()
        data["rewards/raw_game_rewards_max"] = raw_rewards.max().item()
        data["rewards/raw_game_rewards_min"] = raw_rewards.min().item()

        #data["rewards/rewards_for_plot"] = rewards_for_plot.mean().item()
        

        # Log the number of envs with positive extrinsic rewards (rewards has shape (num_steps, num_envs))
        data["rewards/num_envs_with_pos_rews"] = torch.sum(rewards.sum(dim=0) > 0).item()

        # Log advantages and intrinsic advantages
        data["returns/advantages"] = b_advantages.mean().item()
        data["returns/ext_advantages"] = b_ext_advantages.mean().item()
        data["returns/int_advantages"] = b_int_advantages.mean().item()
        data["returns/ret_ext"] = b_ext_returns.mean().item()
        data["returns/ret_int"] = b_int_returns.mean().item()
        data["returns/values_ext"] = b_ext_values.mean().item()
        data["returns/values_int"] = b_int_values.mean().item()

        data["charts/traj_len"] = np.mean(avg_ep_lens)
        data["charts/max_traj_len"] = np.max(avg_ep_lens, initial=0)
        data["charts/min_traj_len"] = np.min(avg_ep_lens, initial=0)
        data["charts/time_per_it"] = it_end_time - it_start_time
        data["charts/game_score"] = float(np.mean(list(avg_returns))) if avg_returns else 0
        data["charts/max_game_score"] = np.max(avg_returns, initial=0)
        data["charts/min_game_score"] = np.min(avg_returns, initial=0)

        # Also log the histogram of the returns every 100 iterations
        if update % 100 == 0:
            if args.track:
                data["charts/returns_hist"] = wandb.Histogram(avg_returns)


        if args.track:
            wandb.log(data, step=global_step)
            trigger_sync()


    envs.close()
    # writer.close()
    


# Define the path in Google Drive
save_path = "/content/drive/MyDrive/rle/saved_models"

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Save the RLE network
save_file = f"{save_path}/{run_name}.pt"
torch.save(rle_network.state_dict(), save_file)

# Confirmation message
print(f"Saved into {save_file}")

# This file implements the PPO algorithm with RLE as the exploration method
# The RLE intrinsic reward is based on features from the value function
# The policy network is a CNN

import argparse
import random
import time
from collections import deque
from distutils.util import strtobool
from itertools import combinations

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

# Create log directory if it doesn't exist
log_dir = "/content/drive/MyDrive/rle/logs"
os.makedirs(log_dir, exist_ok=True)

# Setup logging with created directory
log_filename = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.WARNING,
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

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

#### THIS COMES FROM THE PAPER OF THE FRENCH UNI #####
def random_VMF(mu, kappa, size=None):
    """
    Von Mises-Fisher distribution sampler
    
    Parameters:
    - mu: mean direction
    - kappa: concentration parameter
        κ=0: uniform on sphere (maximum exploration)
        κ>0: concentrated around mu (more exploitation)
    - size: number of samples or batch shape
    
    Returns:
    - Samples from VMF distribution on unit sphere
    """
    n = 1 if size is None else np.product(size)
    shape = () if size is None else tuple(np.ravel(size))
    
    # mu needs to be in the unit sphere:
    mu = np.asarray(mu)
    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape  # dimension
    
    # if k=0, we equally likely distribution over the border of the sphere
    if kappa == 0:
        z = np.random.normal(0, 1, (n, d))
        return (z / np.linalg.norm(z, axis=1, keepdims=True)).reshape((*shape, d))
    
    # Step 1: Get samples perpendicular to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]  # perp to mu
    z /= np.linalg.norm(z, axis=1, keepdims=True)  # and norm again
    
    # Step 2: Sample angles using rejection sampling
    cos = _random_VMF_cos(d, kappa, n)
    sin = np.sqrt(1 - cos**2)
    
    # Step 3: Combine to get points on sphere
    x = z * sin[:, None] + cos[:, None] * mu[None, :]
    return x.reshape((*shape, d))

def _random_VMF_cos(d: int, kappa: float, n: int):
    """Generate cosine samples using Wood's rejection sampling method"""
    b = (d - 1) / (2 * kappa + (4 * kappa**2 + (d - 1)**2)**0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = np.random.beta((d - 1)/2, (d - 1)/2, size=m)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        accept = test >= -np.random.exponential(size=m)
        out.append(t[accept])
        found += len(out[-1])
    return np.concatenate(out)[:n]

##########

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
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - (infos["terminated"] | infos["TimeLimit.truncated"])
        self.episode_lengths *= 1 - (infos["terminated"] | infos["TimeLimit.truncated"])
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


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

    def get_value(self, x, reward, goal):
        obs_hidden = self.network(x / 255.0)

        # goal is the goal vector
        # it is a tensor of shape (num_envs, feature_size)
        goal_hidden = self.goal_encoder(goal)
        hidden = obs_hidden + goal_hidden

        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class RLEModel(nn.Module):
    def __init__(self, input_shape, feature_size, output_size, num_actions, num_envs, z_layer_init, device):
        super().__init__()
        self.input_size = input_shape
        self.output_size = output_size
        self.device = device
        self.feature_size = feature_size
        self.num_envs = num_envs  
        
        # RLE network architecture 
        self.rle_net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.last_layer = z_layer_init(nn.Linear(448, self.feature_size))

        # VMF sampling parameters
        self.base_kappa = 20.0
        self.current_kappa = 0.0
        self.kappa_momentum = 0.99  # Smoothing parameter, we can play around with this, needs to be tune according to global steps
        self.min_kappa = 0.0
        self.max_kappa = self.base_kappa
        
        # Success tracking for the weighting of kappa
        self.min_samples_for_exploitation = 10  #TO PLAY AROUND
        self.success_threshold_percentile = 50
        self.success_memory = deque(maxlen=100)
        self.returns_memory = deque(maxlen=100)
        
        # Running statistics for kappa
        self.running_avg_return = 0
        self.running_std_return = 0
        self.return_momentum = 0.99  #TO PLAY AROUND
        
        self.episode_count = 0 # Also for weighting
        
        # Init goal
        self.goals = self.sample_goals()
        
        # To manage switching as them
        self.num_steps_left = torch.randint(1, args.switch_steps, (num_envs,)).to(device)
        self.switch_goals_mask = torch.zeros(num_envs).to(device)
        
        self.rle_rms = RunningMeanStd(shape=(1, self.feature_size))
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

        # To avoid dividing by zero if std too little.
        self.min_std = 1e-8  # Add this

    def get_current_direction(self):
        # Here we sample the e1 if we do not have a sufficient number of past good z vectors
        if len(self.success_memory) < self.min_samples_for_exploitation:
            mu = torch.zeros(self.feature_size, device=self.device)
            mu[0] = 1.0
            return mu

        returns = torch.tensor(list(self.returns_memory), device=self.device)
        z_vectors = torch.stack([z.to(self.device) for z in self.success_memory])
        
        # We will weight the mus according to which one did better!
        weights = torch.softmax((returns - returns.mean()) / returns.std(), dim=0) 

        # Weighted average of successful directions
        mu = (weights.unsqueeze(1) * z_vectors).sum(0)
        mu = mu / mu.norm()

        logging.info(f"Computing direction - Returns shape: {returns.shape}, Z vectors shape: {z_vectors.shape}")
        logging.info(f"Mean direction stats - mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")

        return mu

    def sample_goals(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_envs
            
        # get current mu
        mu = self.get_current_direction()
        
        logging.info(f"Sampling goals - Current mu mean: {mu.mean().item()}, std: {mu.std().item()}")
        
        # and sample with it.
        samples = random_VMF(
            mu=mu.cpu().numpy(),  # Convert to CPU for numpy
            kappa=self.current_kappa,
            size=num_envs
        )
        
        # Create tensor
        samples_tensor = torch.from_numpy(samples).float().to(self.device)
        logging.info(f"Sampled goals stats - mean: {samples_tensor.mean().item()}, std: {samples_tensor.std().item()}")
        return samples_tensor

    def update_sampling_history(self, env_idx, episode_return):
        """Using success history we weight the kappa"""
        z = self.goals[env_idx]
        self.episode_count += 1
        
        logging.info(f"Update sampling history - Episode return: {episode_return}")
        logging.info(f"Current memory size: {len(self.success_memory)}")
        logging.info(f"Current kappa: {self.current_kappa}")
        
        if self.episode_count == 1:
            self.running_avg_return = episode_return
            self.running_std_return = max(episode_return / 10, self.min_std) 
        else:
            delta = episode_return - self.running_avg_return
            self.running_avg_return = self.running_avg_return * self.return_momentum + episode_return * (1 - self.return_momentum)
            self.running_std_return = max(
                self.running_std_return * self.return_momentum + abs(delta) * (1 - self.return_momentum),
                self.min_std
            )
        
        # We are still in the "learning" phase
        if len(self.success_memory) < self.min_samples_for_exploitation:
            logging.info(f" initial episode (bootstrap phase)")
            self.success_memory.append(z.detach().cpu())
            self.returns_memory.append(float(episode_return))
            logging.info(f"z vector to bootstrap. memory size: {len(self.success_memory)}")
            return
        
        # Dynamic thresholding
        if len(self.returns_memory) > 0:
            #we get the min return value according to the pass threshold percentile
            threshold = np.percentile(list(self.returns_memory), self.success_threshold_percentile)
            logging.info(f"Success threshold: {threshold}")

            # If return higher than threshold, then z is good, add it to success memory
            # NOTICE, THIS IDEA IS NOT CLEARLY GOOD SINCE DIFF Zs CAN BE GOOD IN DIFFERENT PARTS OF THE GAME, FOR THAT WE NEED DYNAMIC Z SAMPLING
            if episode_return > threshold:
                self.success_memory.append(z.detach().cpu())
                self.returns_memory.append(float(episode_return))
                
                # Here are the kappa weights
                quality_factor = min(max(0.0, (episode_return - self.running_avg_return)) / (self.running_std_return + 1e-8), 2.0) # so more weight to quality
                memory_factor = len(self.success_memory) / self.success_memory.maxlen # more weight to "experience"
                consistency_factor = min(self.episode_count / 1000, 1.0)  # and warm up idea 1000 SHOULD BE PASS AS HYPERPARAMETER #TO PLAY AROUND

                target_kappa = self.base_kappa * quality_factor * memory_factor * consistency_factor 
                target_kappa = min(max(target_kappa, self.min_kappa), self.max_kappa)

                # SMOOTHING KAPPA
                # TODO: uncomment
                # self.current_kappa = (self.kappa_momentum * self.current_kappa +
                                    #(1 - self.kappa_momentum) * target_kappa)
                ## Jonas
                def average_pairwise_cosine_similarity(vectors):
                    vectors = np.array(vectors)

                    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

                    num_vectors = len(normalized_vectors)
                    if num_vectors < 2:
                        return None

                    cosine_similarities = [
                        np.dot(normalized_vectors[i], normalized_vectors[j])
                        for i, j in combinations(range(num_vectors), 2)
                    ]

                    # Calculate the average cosine similarity
                    return np.mean(cosine_similarities)

                if len(self.success_memory) == self.success_memory.maxlen:
                    avg_pairwise_cosine_similarity = average_pairwise_cosine_similarity(self.success_memory)

                    t = (avg_pairwise_cosine_similarity--1)/(1--1)

                    #e = 3
                    #t_exp = t ** e

                    self.current_kappa = (1 - t) * 0 + t * 30

                ## End of Jonas

                
                logging.info(f"Added new z vector. New memory size: {len(self.success_memory)}")
                logging.info(f"Quality factor: {quality_factor:.3f}")
                logging.info(f"Memory factor: {memory_factor:.3f}")
                logging.info(f"Consistency factor: {consistency_factor:.3f}")
                logging.info(f"Target kappa: {target_kappa:.3f}")
                logging.info(f"New kappa: {self.current_kappa:.3f}")
                logging.info(f"Running avg return: {self.running_avg_return:.1f}")
                logging.info(f"Running std return: {self.running_std_return:.1f}")
            else:
                logging.info(f"Episode return {episode_return} below threshold {threshold}, not storing")


    def step(self, next_done: torch.Tensor, next_ep_done: torch.Tensor, returns=None):
        """
        Handle goal switching and update success history
        """
        self.switch_goals_mask = torch.zeros(args.num_envs).to(self.device)
        self.switch_goals_mask[next_done.bool()] = 1.0
        self.num_steps_left -= 1
        self.switch_goals_mask[self.num_steps_left == 0] = 1.0
        
        # once episode finish, pass the return.
        for env_idx, (done, info_done) in enumerate(zip(next_done, next_ep_done)):
            if info_done and returns is not None:  # Episode actually completed
                episode_return = returns[env_idx]
                self.update_sampling_history(env_idx, episode_return)
        
        # new goals sampling
        new_goals = self.sample_goals()
        self.goals = self.goals * (1 - self.switch_goals_mask.unsqueeze(1)) + new_goals * self.switch_goals_mask.unsqueeze(1)
        
        # and we reset steps for the new samples , since we had this hyperparameter of how much steps to sample again besides paper....
        self.num_steps_left[self.switch_goals_mask.bool()] = args.switch_steps
        
        return self.switch_goals_mask

    def compute_rle_feat(self, obs, goals=None):
        if goals is None:
            goals = self.goals
            
        with torch.no_grad():
            raw_rle_feat = self.last_layer(self.rle_net(obs))
            rle_feat = (raw_rle_feat - self.rle_feat_mean) / (self.rle_feat_std + 1e-5)
            reward = (rle_feat * goals).sum(axis=1) / torch.norm(rle_feat, dim=1)
        
        return reward, raw_rle_feat, rle_feat

    def compute_reward(self, obs, next_obs, goals=None):
        return self.compute_rle_feat(next_obs, goals=goals)

    def update_rms(self, b_rle_feats):
        self.rle_rms.update(b_rle_feats)
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def log_memory_state(self, global_step):
        if len(self.success_memory) > 0:
            z_vectors = torch.stack(list(self.success_memory))
            returns = torch.tensor(list(self.returns_memory))
            
            logging.info(f"\nMemory State at step {global_step}:")
            logging.info(f"Memory size: {len(self.success_memory)}/{self.success_memory.maxlen}")
            logging.info(f"Mean return: {returns.mean().item():.2f}")
            logging.info(f"Max return: {returns.max().item():.2f}")
            logging.info(f"Z vectors mean: {z_vectors.mean().item():.4f}")
            logging.info(f"Z vectors std: {z_vectors.std().item():.4f}")
            logging.info(f"Current kappa: {self.current_kappa:.4f}")

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
    rle_network = RLEModel(envs.single_observation_space.shape, args.feature_size, rle_output_size, num_actions, args.num_envs,
                           z_layer_init=create_layer_init_from_spec(args.z_layer_init),
                           device=device).to(device)
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
    next_obss = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
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

    next_raw_rle_feat = []

    # Update rms of the rle network
    if args.norm_rle_features:
        print("Start to initialize rle features normalization parameter.....")
        for step in range(args.num_steps * args.num_iterations_feat_norm_init):
            acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
            s, r, d, _ = envs.step(acs)
            rle_reward, raw_rle_feat, rle_feat = rle_network.compute_rle_feat(torch.Tensor(s).to(device).clone().float())
            next_raw_rle_feat += raw_rle_feat.detach().cpu().numpy().tolist()

            if len(next_raw_rle_feat) % (args.num_steps * args.num_envs) == 0:
                next_raw_rle_feat = np.stack(next_raw_rle_feat)
                rle_network.update_rms(next_raw_rle_feat)
                next_raw_rle_feat = []
        print(f"End of initializing... finished in {time.time() - start_time}")

    video_recorder = VideoRecorder(local_dir=args.local_dir, use_wandb=args.track & args.capture_video)
    video_record_conditioner = VideoStepConditioner(global_step_interval=int(args.capture_video_interval))
    is_early_stop = False

    for update in range(1, num_updates + 1):
        prev_rewards[0] = rle_rewards[-1] * args.int_coef  # last step of prev rollout
        it_start_time = time.time()

        # Annealing the rate if instructed to do so.
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

            # Compute the obs before stepping
            rle_obs = next_obs.clone().float()

            # Compute mean and std over all the envs of norm of rle_obs for logging
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
            next_obss[step] = next_obs
            rle_next_obs = next_obs.clone().float()
            rle_reward, raw_next_rle_feat, next_rle_feat = rle_network.compute_reward(rle_obs, rle_next_obs)
            rle_rewards[step] = rle_reward.data.clone() 
            raw_rle_feats[step] = raw_next_rle_feat
            rle_feats[step] = next_rle_feat

            int_reward_info = {
                "int_rewards": rle_rewards[step, 0].cpu().numpy(),
            }

            if args.capture_video:
                video_recorder.record(obs[step][0, 3, ...].cpu().numpy().copy(),
                                      info["reward"][0],
                                      int_reward_info,
                                      global_step=global_step)

            # update prev rewards
            if step < args.num_steps - 1:
                prev_rewards[step + 1] = rle_rewards[step].clone() * args.int_coef 

            for idx, d in enumerate(done):
                if info["terminated"][idx] or info["TimeLimit.truncated"][idx]:
                    episode_return = info["r"][idx]
                    avg_returns.append(info["r"][idx])
                    avg_ep_lens.append(info["l"][idx])

                    # Log memory state every 10 episodes
                    if len(avg_returns) % 10 == 0:
                        logging.info("\n=== Memory State ===")
                        logging.info(f"Memory size: {len(rle_network.success_memory)}")
                        if len(rle_network.success_memory) > 0:
                            returns = torch.tensor(list(rle_network.returns_memory))
                            logging.info(f"Returns stats - mean: {returns.mean():.2f}, min: {returns.min():.2f}, max: {returns.max():.2f}")
                            logging.info(f"Current kappa: {rle_network.current_kappa:.4f}")
                            logging.info(f"Success rate: {len(rle_network.success_memory) / rle_network.success_memory.maxlen:.4f}")
                        logging.info("==================\n")

                    # STATISTICS UPDATE OF VMF
                    rle_network.update_sampling_history(idx, episode_return)
                    
                    if global_step % 10000 == 0:  
                        rle_network.log_memory_state(global_step)

                    if args.track:
                        wandb.log({
                            "charts/episode_return": episode_return,
                            "charts/vmf_kappa": rle_network.current_kappa,  # Log current concentration parameter
                            "charts/success_rate": len(rle_network.success_memory) / rle_network.success_memory.maxlen
                        }, step=global_step)

                    if args.capture_video and idx == 0:
                        if video_record_conditioner(info["r"][idx], global_step):
                            video_recorder.flush(global_step=global_step, caption=f"{info['r'][idx]}")
                            print(f"Logged a video with len={info['l'][idx]} and return={info['r'][idx]}")
                        else:
                            video_recorder.reset()
                            print(f"Env idx={idx}: len={info['l'][idx]} and return={info['r'][idx]} is reset.")
                        trigger_sync()
            #STEP   
            next_ep_done = info["terminated"] | info["TimeLimit.truncated"]
            rle_network.step(next_done, next_ep_done, returns=info["r"])


        #here they do the weird normalization of them. 
        not_dones = (1.0 - dones).cpu().data.numpy()
        rewards_for_plot = rewards.clone()
        
        rewards_cpu = rewards.cpu().data.numpy()
        rle_rewards_cpu = rle_rewards.cpu().data.numpy()
        raw_rewards = rewards.clone() #added for debugging 
        if args.normalize_ext_rewards:
            ext_reward_per_env = np.array(
                [ext_discounted_reward.update(rewards_cpu[i], not_dones[i]) for i in range(args.num_steps)]
            )
            ext_reward_rms.update(ext_reward_per_env.flatten())
            rewards /= np.sqrt(ext_reward_rms.var)

        # Normalize random rewards
        rle_not_dones = (1.0 - rle_dones).cpu().data.numpy()
        if True:
            rle_reward_per_env = np.array(
                [int_discounted_reward.update(rle_rewards_cpu[i], rle_not_dones[i]) for i in range(args.num_steps)]
            )
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
        b_next_obs = next_obss.reshape((-1,) + envs.single_observation_space.shape)
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
        b_raw_rle_feats = raw_rle_feats.reshape((-1, rle_feature_size))
        b_prev_rewards = prev_rewards.reshape(-1)

        # Update rms of the rle network
        if args.norm_rle_features:
            rle_network.update_rms(b_raw_rle_feats.cpu().numpy())

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

                for param, target_param in zip(agent.network.parameters(), rle_network.rle_net.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

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
            eval_goal = rle_network.sample_goals(args.num_eval_envs).to(device)

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
                eval_obs, eval_done = torch.Tensor(eval_obs).to(device), torch.Tensor(eval_done).to(device)

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

                new_goals = rle_network.sample_goals(args.num_eval_envs).to(device)
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

        data["rewards/rewards_for_plot"] = rewards_for_plot.mean().item()
        

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
        data["charts/game_score"] = np.mean(avg_returns)
        data["charts/max_game_score"] = np.max(avg_returns, initial=0)
        data["charts/min_game_score"] = np.min(avg_returns, initial=0)

        def angle_between_vectors(a, b):
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            angle_rad = np.arccos(dot_product / (norm_a * norm_b))

            return np.degrees(angle_rad)


        e1 = np.zeros(shape=(rle_network.feature_size))
        e1[0] = 1.0

        angle_between_mu_e1 = angle_between_vectors(e1, rle_network.get_current_direction().cpu().numpy())

        wandb.log({"charts/e1_mu_angle": angle_between_mu_e1}, step=global_step)

        # Also log the histogram of the returns every 100 iterations
        if update % 100 == 0:
            if args.track:
                data["charts/returns_hist"] = wandb.Histogram(avg_returns)

                # Every N updates, log the distribution of successful directions
        if update % 100 == 0 and len(rle_network.success_memory) > 0:
            successful_directions = torch.stack(list(rle_network.success_memory))
            if args.track:
                # Log distribution of successful directions
                for dim in range(rle_network.feature_size):
                    wandb.log({
                        f"vmf/direction_dim_{dim}_hist": wandb.Histogram(successful_directions[:, dim].cpu().numpy())
                    }, step=global_step)
                
                # Log mean direction
                mean_direction = rle_network.get_current_direction().cpu().numpy()
                wandb.log({
                    "vmf/mean_direction": wandb.Histogram(mean_direction)
                }, step=global_step)

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

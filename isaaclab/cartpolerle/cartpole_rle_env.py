
#https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html
import torch
import numpy as np
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.direct.cartpole.cartpole_env import CartpoleEnv, CartpoleEnvCfg
from .networks import FeatureNetwork
import logging
from gym.wrappers.normalize import RunningMeanStd
import os
import csv
from copy import deepcopy

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



@configclass
class CartpoleRLEEnvCfg(CartpoleEnvCfg):
    feature_size: int = 32
    int_coef: float = 0.01 
    ext_coef: float = 1.0 
    tau: float = 0.005
    z_reset_freq: int = 16
    cfg_entry_point: str = "skrl_ppo_cfg_rle"
    num_envs: int = 4096

    
class CartpoleRLEEnv(CartpoleEnv):
    cfg: CartpoleRLEEnvCfg
    
    def __init__(self, cfg: CartpoleRLEEnvCfg, render_mode: str | None = None, **kwargs):
        self.base_obs_size = 4
        cfg.observation_space += cfg.feature_size
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize coefficients
        self._int_coef = self.cfg.int_coef
        self._ext_coef = self.cfg.ext_coef
        
        # Initialize feature network
        self.feature_network = FeatureNetwork(
            obs_shape=self.base_obs_size,
            feature_size=self.cfg.feature_size,
            tau=self.cfg.tau,
            device=self.device
        ).to(self.device)
        
        # Initialize tracking variables
        self.latent_vectors = self.feature_network.sample_z(self.num_envs)
        self.steps_since_z_reset = torch.zeros(self.num_envs, device=self.device)
        self.episode_features = []
        self.policy_net = None
        
        # Initialize reward statistics
        self.reward_forward_filter_int = RewardForwardFilter(gamma=0.99)
        self.reward_forward_filter_ext = RewardForwardFilter(gamma=0.99)
        self.ext_reward_rms = RunningMeanStd()
        self.int_reward_rms = RunningMeanStd()
        
        # Initialize step counting and reward tracking
        self.step_count = torch.zeros(self.num_envs, device=self.device)
        self.episode_extrinsic_reward = None
        self.episode_intrinsic_reward = None
        self.episode_int_rewards = None
        
        # Call reset_episode_rewards to properly initialize reward tensors
        self.reset_episode_rewards()
        
        # Initialize return history for logging
        self.return_history = []
        self.window_size = 100

        # Initialize CSV file for storing extrinsic returns
        self.csv_file_path = "extrinsic_returns_rle_running_avg.csv"
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Episode", "Extrinsic Return"])

    def reset_episode_rewards(self):
        """Initialize/reset episode rewards with proper shapes"""
        batch_size = self.num_envs
        self.episode_extrinsic_reward = torch.zeros(batch_size, device=self.device).unsqueeze(1)
        self.episode_intrinsic_reward = torch.zeros(batch_size, device=self.device).unsqueeze(1)
        self.episode_int_rewards = torch.zeros(batch_size, device=self.device).unsqueeze(1)  # For tracking cumulative intrinsic rewards
    def _get_running_average(self):
        if not self.return_history:
            return 0.0
        window = self.return_history[-self.window_size:]
        return sum(window) / len(window)



    def step(self, action):
            self.step_count += 1
            next_obs, reward, terminated, truncated, info = super().step(action)
            self._update_latent_vectors(terminated | truncated)  # Combine termination signals
            return next_obs, reward, terminated, truncated, info

    def _get_observations(self) -> dict:
        """by printing we can realize that 
        self.joint_pos and self.joint_vel contains all joint positions and velocities
        then, indexing by 0 get us the cart position and velocity whereas the 1 get us the pole position and velocity"""
        # base observations (pole angle, pole vel, cart pos, cart vel)
        base_obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        #the unsqueeze help us to : base_obs  # Shape: [num_envs, 4]  # 4 = [pole_angle, pole_vel, cart_pos, cart_vel]
        # now for policy (and value action network) we need the combined observation
        combined_obs = torch.cat([base_obs, self.latent_vectors], dim=1)
        return {
            "policy": combined_obs, #notice that it is how is called by the train.py in "C:\Users\ASUS\Documents\isaaclab\IsaacLab\source\standalone\workflows\rl_games\train.py"
            "base_obs": base_obs  #but we keep them identify for the feature network
        }

    def _get_rewards(self) -> torch.Tensor:
        """Get rewards with proper shapes"""
        # Get base CartPole reward and reshape
        batch_size = self.num_envs
        extrinsic_reward = super()._get_rewards().view(batch_size, 1)

        # Get base observation for RLE
        base_obs = self._get_observations()["base_obs"]
        
        # Ensure observation is properly shaped (flattened)
        base_obs = base_obs.reshape(batch_size, -1)
        
        # Compute intrinsic rewards
        intrinsic_reward, features = self.feature_network.compute_reward(base_obs, self.latent_vectors)
        intrinsic_reward = intrinsic_reward.view(batch_size, 1)
        
        # Update tracking with properly shaped rewards
        self.episode_extrinsic_reward += extrinsic_reward
        self.episode_intrinsic_reward += intrinsic_reward
        self.episode_features.append(features)
        
        # Store reward info
        self.latest_rewards_info = {
            "extrinsic_reward": extrinsic_reward,
            "intrinsic_reward": intrinsic_reward,
            "features": features,
            "true_reward": extrinsic_reward.clone()
        }
        
        # Return combined reward
        return self._ext_coef * extrinsic_reward + self._int_coef * intrinsic_reward
    
    def _reset_idx(self, env_ids):
        """Modified reset to properly handle latent vectors"""
        super()._reset_idx(env_ids)
        
        if env_ids is not None:
            # Calculate statistics
            extrinsic_returns = self.episode_extrinsic_reward[env_ids]
            intrinsic_returns = self.episode_intrinsic_reward[env_ids]
            steps = self.step_count[env_ids]
            mean_ext_return = extrinsic_returns.mean().item()
            mean_int_return = intrinsic_returns.mean().item()
            mean_steps = steps.mean().item()
            
            self.return_history.append(mean_ext_return)
            running_avg = self._get_running_average()
            
            print(f"\n{'-'*50}")
            print(f"Episode Summary (Reset for {len(env_ids)} environments):")
            print(f"  Extrinsic Return: {mean_ext_return:>8.2f}  (Running Avg: {running_avg:>8.2f})")
            print(f"  Intrinsic Return: {mean_int_return:>8.2f}")
            print(f"  Episode Length:   {mean_steps:>8.2f} steps")
            print(f"{'-'*50}")
            with open(self.csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([len(self.return_history), running_avg])

            # Sample new latent vectors for reset environments
            new_z = self.feature_network.sample_z(len(env_ids))
            self.latent_vectors[env_ids] = new_z
            self.steps_since_z_reset[env_ids] = 0

            info = {
                "episode": {
                    "r": mean_ext_return,
                    "l": mean_steps,
                    "extrinsic_return": mean_ext_return,
                    "intrinsic_return": mean_int_return,
                    "running_avg": running_avg
                }
            }
                
            self.episode_extrinsic_reward[env_ids] = 0
            self.episode_intrinsic_reward[env_ids] = 0
            self.step_count[env_ids] = 0
            
            if len(self.episode_features) > 0 and self.policy_net is not None:
                episode_features = torch.cat(self.episode_features, dim=0)
                self.feature_network.update_stats(episode_features)
                self.feature_network.update_from_policy_net(self.policy_net)
                self.episode_features = []
        else:
            info = {}
                
        return info

    def set_policy_net(self, policy_net):
        """Called by trainer to provide access to policy network"""
        self.policy_net = policy_net
        if hasattr(policy_net, "rle_net"):
            self.feature_network = policy_net.rle_net

    
    def _update_latent_vectors(self, dones):
        """Update latent vectors based on episode termination and time steps"""
        self.steps_since_z_reset += 1
        
        #  switching mask
        switch_mask = torch.zeros(self.num_envs, device=self.device)
        #  on episode end
        switch_mask[dones.bool()] = 1.0
        #  on interval
        switch_mask[self.steps_since_z_reset >= self.cfg.z_reset_freq] = 1.0
        
        if switch_mask.any():
            # new vectors only for those that need switching
            new_z = self.feature_network.sample_z(switch_mask.sum().int().item())
            # using mask
            self.latent_vectors = self.latent_vectors * (1 - switch_mask.unsqueeze(1)) + \
                                torch.cat([new_z, torch.zeros_like(self.latent_vectors[~switch_mask.bool()])]) * switch_mask.unsqueeze(1)
            self.steps_since_z_reset[switch_mask.bool()] = 0
# File: omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/cartpole/networks/rle_networks.py
import torch
import torch.nn as nn
import numpy as np
from gym.wrappers.normalize import RunningMeanStd

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FeatureNetwork(nn.Module):
    def __init__(self, obs_shape, feature_size=32, tau=0.005, device='cuda'):
        super().__init__()
        self.device = device
        self.feature_size = feature_size
        self.tau = tau
        
        self.input_size = obs_shape if isinstance(obs_shape, int) else np.prod(obs_shape)
        
        self.rle_net = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 512)), 
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
        )
        self.feature_layer = layer_init(nn.Linear(256, feature_size))
        
        self.rms = RunningMeanStd(shape=(1, feature_size))
        self.feat_mean = torch.zeros(feature_size, device=device)
        self.feat_std = torch.ones(feature_size, device=device)
        
        self.history_size = 1000
        self.feature_history = []
        
        self.reward_scale = 0.1
        
        # Current goals
        self.latent_vectors = None
        self.switch_steps = 16  
        self.steps_since_reset = None

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        shared_features = self.rle_net(x)
        features = self.feature_layer(shared_features)
        return features

    def normalize_features(self, features):
        if len(self.feature_history) > 0:
            hist_tensor = torch.cat(self.feature_history, dim=0)
            self.feat_mean = hist_tensor.mean(dim=0)
            self.feat_std = torch.clamp(hist_tensor.std(dim=0), min=1e-6)
        
        return (features - self.feat_mean) / (self.feat_std + 1e-8)

    def compute_reward(self, state, z):
        with torch.no_grad():
            if len(state.shape) > 2:
                state = state.reshape(state.shape[0], -1)
                
            raw_features = self.forward(state)
            norm_features = self.normalize_features(raw_features)
            
            z_normalized = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            
            reward = torch.sum(norm_features * z_normalized, dim=1)
            reward = torch.clamp(reward, min=0.0) * self.reward_scale
            
        return reward, raw_features

    def update_stats(self, features):
        self.rms.update(features.detach().cpu().numpy())
        self.feat_mean = torch.from_numpy(self.rms.mean).float().to(self.device)
        self.feat_std = torch.from_numpy(np.sqrt(self.rms.var)).float().to(self.device)

    def sample_z(self, batch_size=1):
        z = torch.randn(batch_size, self.feature_size, device=self.device)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        return z

    def update_from_policy_net(self, policy_net, tau=None):
        if tau is None:
            tau = self.tau
            
        if hasattr(policy_net, 'critic_base'):
            for param, target_param in zip(policy_net.critic_base.parameters(), self.rle_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
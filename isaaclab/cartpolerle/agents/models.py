import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RLEPolicyModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, rle_net=None, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)

        self.goal_embed_size = 16
        obs_size = np.array(observation_space.shape).prod()
        action_size = np.prod(action_space.shape)


        self.rle_net = rle_net
        feature_size = kwargs.get('feature_size', 32) if rle_net is None else rle_net.feature_size

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(feature_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.goal_embed_size), std=0.1),
            nn.Tanh(),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size + self.goal_embed_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_size), std=0.01),
        )
    
        self.actor_logstd = nn.Sequential(
            layer_init(nn.Linear(self.goal_embed_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_size), std=1.0),
        )

    def act(self, inputs, role):
        """Override act method to use GaussianMixin's act"""
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        x = inputs["states"]
        batch_size = x.shape[0]

        if self.rle_net is None:
            raise ValueError("RLE network not provided")
            
        # If latent vectors don't match batch size, regenerate them
        if not hasattr(self.rle_net, 'latent_vectors') or \
        self.rle_net.latent_vectors.shape[0] != batch_size:
            self.rle_net.latent_vectors = self.rle_net.sample_z(batch_size).to(self.device)

        goals = self.rle_net.latent_vectors[:batch_size]  # Ensure correct batch size
        if goals.device != x.device:
            goals = goals.to(x.device)

        # Encode goals
        goal_embed = self.goal_encoder(goals)
        
        # Compute action distribution parameters
        action_mean = self.actor_mean(torch.cat([x, goal_embed], dim=1))
        action_logstd = self.actor_logstd(goal_embed)

        return action_mean, action_logstd, {}


class RLEValueModel(Model):
    def __init__(self, observation_space, action_space, device, rle_net=None, **kwargs):
        super().__init__(observation_space, action_space, device)

        self.goal_embed_size = 16
        obs_size = np.array(observation_space.shape).prod()
        
        # Store reference to RLE network
        self.rle_net = rle_net
        feature_size = kwargs.get('feature_size', 32) if rle_net is None else rle_net.feature_size

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(feature_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.goal_embed_size), std=0.1),
            nn.Tanh(),
        )

        # Combined state-goal encoder
        self.goal_state_encoder = nn.Sequential(
            layer_init(nn.Linear(256 + self.goal_embed_size, 256)),
            nn.Tanh(),
        )

        # Base critic network
        self.critic_base = nn.Sequential(
            layer_init(nn.Linear(obs_size, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
        )

        # Separate heads for intrinsic and extrinsic values
        self.critic_ext = layer_init(nn.Linear(256, 1), std=1.0)
        self.critic_int = layer_init(nn.Linear(256, 1), std=1.0)

    def act(self, inputs, role):
        value, outputs = self.compute(inputs, role)
        return value, None, outputs

    def compute(self, inputs, role):
        x = inputs["states"]
        batch_size = x.shape[0]
        
        if self.rle_net is None:
            raise ValueError("RLE network not provided")
            
        # If latent vectors don't match batch size, regenerate them
        if not hasattr(self.rle_net, 'latent_vectors') or \
        self.rle_net.latent_vectors.shape[0] != batch_size:
            self.rle_net.latent_vectors = self.rle_net.sample_z(batch_size).to(self.device)

        goals = self.rle_net.latent_vectors[:batch_size]  # Ensure correct batch size
        if goals.device != x.device:
            goals = goals.to(x.device)

        goal_embed = self.goal_encoder(goals)
        hidden = self.critic_base(x)
        
        # Combine state and goal features
        hidden = self.goal_state_encoder(torch.cat([hidden, goal_embed], dim=1))

        return self.critic_ext(hidden), {
            "critic_ext": self.critic_ext(hidden),
            "critic_int": self.critic_int(hidden)
        }
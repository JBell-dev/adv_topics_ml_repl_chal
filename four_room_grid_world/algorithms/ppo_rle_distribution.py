from abc import abstractmethod, ABC

import torch
from torch.distributions import VonMises, Exponential


class RLEGoalSampler(ABC):
    @abstractmethod
    def sample(self, num_envs: int, rle_feature_size: int, device: str) -> torch.Tensor:
        raise RuntimeError("Sampler subclass must implement sample method")


class RLEGoalSamplerCreator:
    @staticmethod
    def create_from_name(sampler_name: str) -> RLEGoalSampler:
        if sampler_name == "standard_normal":
            return StandardNormalSampler()
        elif sampler_name == "standard_uniform":
            return StandardUniformSampler()
        elif sampler_name == "von_mises":
            return VonMisesSampler()
        elif sampler_name == "exponential":
            return ExponentialSampler()
        raise RuntimeError("A sampler with this name does not exist")


class StandardNormalSampler(RLEGoalSampler):
    def sample(self, num_envs: int, rle_feature_size: int, device: str) -> torch.Tensor:
        return torch.randn((num_envs, rle_feature_size), device=device).float()


class StandardUniformSampler(RLEGoalSampler):
    def sample(self, num_envs: int, rle_feature_size: int, device: str) -> torch.Tensor:
        return torch.rand((num_envs, rle_feature_size), device=device).float()


class VonMisesSampler(RLEGoalSampler):
    def __init__(self):
        self.dist = VonMises(0.0, 0.3)

    def sample(self, num_envs: int, rle_feature_size: int, device: str) -> torch.Tensor:
        return self.dist.sample(sample_shape=(num_envs, rle_feature_size)).to(device).float()


class ExponentialSampler(RLEGoalSampler):
    def __init__(self):
        self.dist = Exponential(0.3)

    def sample(self, num_envs: int, rle_feature_size: int, device: str) -> torch.Tensor:
        return self.dist.sample(sample_shape=(num_envs, rle_feature_size)).to(device).float()

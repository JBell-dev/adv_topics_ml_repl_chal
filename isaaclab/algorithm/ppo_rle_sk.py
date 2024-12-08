from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR


# [start-config-dict-torch]
RLEPPO_SK_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor
    "int_value_loss_scale": 0.5,    # intrinsic value loss scaling factor
    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)
    
    # Add RLE specific parameters
    "ext_coef": 1.0,    # Coefficient for extrinsic rewards
    "int_coef": 0.1,    # Coefficient for intrinsic rewards
    "feature_size": 32,  # Size of feature vector
    "tau": 0.005,       # Soft update coefficient

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]


class RLEPPO_SK(Agent):
    def __init__(self,
                models: Mapping[str, Model],
                memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                device: Optional[Union[str, torch.device]] = None,
                cfg: Optional[dict] = None) -> None:
        """Initialize RLEPPO_SK with dual reward coefficients"""
        print("PPO init")
        print(models)
        _cfg = copy.deepcopy(RLEPPO_SK_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                        memory=memory,
                        observation_space=observation_space,
                        action_space=action_space,
                        device=device,
                        cfg=_cfg)

        self._ext_coef = self.cfg.get("ext_coef", 1.0)  
        self._int_coef = self.cfg.get("int_coef", 0.01)  
        self._int_value_loss_scale = self.cfg.get("int_value_loss_scale", 0.5)  
        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                  lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent with additional tensors for RLE"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)  # extrinsic rewards
            self.memory.create_tensor(name="rewards_int", size=1, dtype=torch.float32)  # intrinsic rewards
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values_ext", size=1, dtype=torch.float32)  # extrinsic values
            self.memory.create_tensor(name="values_int", size=1, dtype=torch.float32)  # intrinsic values
            self.memory.create_tensor(name="returns_ext", size=1, dtype=torch.float32)  # extrinsic returns
            self.memory.create_tensor(name="returns_int", size=1, dtype=torch.float32)  # intrinsic returns
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values_ext", "values_int", 
                                "returns_ext", "returns_int", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")
        self._current_log_prob = log_prob

        return actions, log_prob, outputs

    def record_transition(self,
                     states: torch.Tensor,
                     actions: torch.Tensor,
                     rewards: torch.Tensor,
                     next_states: torch.Tensor,
                     terminated: torch.Tensor,
                     truncated: torch.Tensor,
                     infos: Any,
                     timestep: int,
                     timesteps: int) -> None:
        """Record transitions with both intrinsic and extrinsic rewards"""
        if self.memory is not None:
            self._current_next_states = next_states

            # (should match number of environments)
            batch_size = states.shape[0]

            # Reward shaping for extrinsic rewards
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # Compute values for both heads
            value_output = self.value.act({"states": self._state_preprocessor(states)}, role="value")[2]
            values_ext = self._value_preprocessor(value_output["critic_ext"], inverse=True)
            values_int = self._value_preprocessor(value_output["critic_int"], inverse=True)

            # Extract and format intrinsic rewards
            if isinstance(infos, (list, tuple)):
                rewards_int = torch.tensor(
                    [info.get("intrinsic_reward", 0.0) for info in infos],
                    device=self.device,
                    dtype=torch.float32
                ).view(batch_size, 1)
            else:
                rewards_int = torch.tensor(
                    [infos.get("intrinsic_reward", 0.0)],
                    device=self.device,
                    dtype=torch.float32
                ).view(1, 1).expand(batch_size, 1)

            #  [batch_size, 1]
            rewards = rewards.view(batch_size, 1)
            terminated = terminated.view(batch_size, 1)
            truncated = truncated.view(batch_size, 1)
            self._current_log_prob = self._current_log_prob.view(batch_size, 1)
            values_ext = values_ext.view(batch_size, 1)
            values_int = values_int.view(batch_size, 1)

            if self._time_limit_bootstrap:
                rewards = rewards + self._discount_factor * values_ext * truncated
                rewards_int = rewards_int + self._discount_factor * values_int * truncated

            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                rewards_int=rewards_int,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values_ext=values_ext,
                values_int=values_int
            )

            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    rewards_int=rewards_int,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values_ext=values_ext,
                    values_int=values_int
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Modified update step to handle dual rewards"""
        def compute_gae_dual(rewards_ext, rewards_int, dones, values_ext, values_int, 
                            next_values_ext, next_values_int, discount_factor=0.99, 
                            lambda_coefficient=0.95):
            """Compute GAE separately for intrinsic and extrinsic rewards"""
            advantage_ext = 0
            advantage_int = 0
            advantages_ext = torch.zeros_like(rewards_ext)
            advantages_int = torch.zeros_like(rewards_int)
            not_dones = dones.logical_not()
            memory_size = rewards_ext.shape[0]

            # compute advantages for both reward streams
            for i in reversed(range(memory_size)):
                next_value_ext = values_ext[i + 1] if i < memory_size - 1 else last_values_ext
                next_value_int = values_int[i + 1] if i < memory_size - 1 else last_values_int

                # extrinsic advantage
                advantage_ext = rewards_ext[i] - values_ext[i] + \
                            discount_factor * not_dones[i] * (next_value_ext + lambda_coefficient * advantage_ext)
                advantages_ext[i] = advantage_ext

                # intrinsic advantage  
                advantage_int = rewards_int[i] - values_int[i] + \
                            discount_factor * not_dones[i] * (next_value_int + lambda_coefficient * advantage_int)
                advantages_int[i] = advantage_int

            # returns computation
            returns_ext = advantages_ext + values_ext
            returns_int = advantages_int + values_int

            # normalize advantages separately
            advantages_ext = (advantages_ext - advantages_ext.mean()) / (advantages_ext.std() + 1e-8)
            advantages_int = (advantages_int - advantages_int.mean()) / (advantages_int.std() + 1e-8)

            # combine advantages with optional weighting
            combined_advantages = self._ext_coef * advantages_ext + self._int_coef * advantages_int

            return returns_ext, returns_int, combined_advantages

        # compute returns and advantages
        with torch.no_grad():
            self.value.train(False)
            last_values_output = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states)}, 
                role="value"
            )[2]
            last_values_ext = self._value_preprocessor(last_values_output["critic_ext"], inverse=True)
            last_values_int = self._value_preprocessor(last_values_output["critic_int"], inverse=True)
            self.value.train(True)

        values_ext = self.memory.get_tensor_by_name("values_ext")
        values_int = self.memory.get_tensor_by_name("values_int")
        returns_ext, returns_int, combined_advantages = compute_gae_dual(
            rewards_ext=self.memory.get_tensor_by_name("rewards"),
            rewards_int=self.memory.get_tensor_by_name("rewards_int"),
            dones=self.memory.get_tensor_by_name("terminated"),
            values_ext=values_ext,
            values_int=values_int,
            next_values_ext=last_values_ext,
            next_values_int=last_values_int,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda
        )

        # update memory
        self.memory.set_tensor_by_name("values_ext", self._value_preprocessor(values_ext, train=True))
        self.memory.set_tensor_by_name("values_int", self._value_preprocessor(values_int, train=True))
        self.memory.set_tensor_by_name("returns_ext", self._value_preprocessor(returns_ext, train=True))
        self.memory.set_tensor_by_name("returns_int", self._value_preprocessor(returns_int, train=True))
        self.memory.set_tensor_by_name("advantages", combined_advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_value_loss_ext = 0
        cumulative_value_loss_int = 0
        cumulative_entropy_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for sampled_states, sampled_actions, sampled_log_prob, sampled_values_ext, \
                sampled_values_int, sampled_returns_ext, sampled_returns_int, sampled_advantages \
                in sampled_batches:

                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                # compute policy loss
                _, next_log_prob, _ = self.policy.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, 
                    role="policy"
                )

                # compute approximate KL divergence
                with torch.no_grad():
                    ratio = next_log_prob - sampled_log_prob
                    kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                    kl_divergences.append(kl_divergence)

                # early stopping with KL divergence
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value losses for both heads
                predicted_values = self.value.act({"states": sampled_states}, role="value")[2]

                if self._clip_predicted_values:
                    predicted_values_ext = sampled_values_ext + torch.clip(
                        predicted_values["critic_ext"] - sampled_values_ext,
                        min=-self._value_clip,
                        max=self._value_clip
                    )
                    predicted_values_int = sampled_values_int + torch.clip(
                        predicted_values["critic_int"] - sampled_values_int,
                        min=-self._value_clip,
                        max=self._value_clip
                    )
                else:
                    predicted_values_ext = predicted_values["critic_ext"]
                    predicted_values_int = predicted_values["critic_int"]

                value_loss_ext = F.mse_loss(sampled_returns_ext, predicted_values_ext)
                value_loss_int = F.mse_loss(sampled_returns_int, predicted_values_int)
                #value_loss = self._value_loss_scale * (value_loss_ext + value_loss_int)
                value_loss = self._value_loss_scale * value_loss_ext + self._int_value_loss_scale * value_loss_int
                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()),
                            self._grad_norm_clip
                        )
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss_ext += value_loss_ext.item()
                cumulative_value_loss_int += value_loss_int.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss (ext)", cumulative_value_loss_ext / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss (int)", cumulative_value_loss_int / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # Update RLE network from critic
        if hasattr(self.policy, 'rle_net') and self.policy.rle_net is not None:
            self.policy.rle_net.update_from_policy_net(self.policy)
from typing import Any, Mapping, Type, Union
import copy

from skrl import logger
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG, RLEPPO_SK, RLEPPO_SK_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer, Trainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model


class RLERunner:
    def __init__(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> None:
        self._env = env
        self._cfg = cfg
        set_seed(self._cfg.get("seed", None))
        
        self._class_mapping = {
            "gaussianmixin": gaussian_model,
            "deterministicmixin": deterministic_model,
            "shared": shared_model,
            "randommemory": RandomMemory,
            "ppo": RLEPPO_SK,
            "ippo": IPPO,
            "mappo": MAPPO,
            "sequentialtrainer": SequentialTrainer,
            "kladaptivelr": KLAdaptiveLR,
            "runningstandardscaler": RunningStandardScaler
        }

        # we handle the reshaping later as in their code
        self._cfg["agent"]["rewards_shaper"] = None
        
        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        
        # Set up RLE network 
        if hasattr(self._env.unwrapped, 'set_policy_net'):
            self._env.unwrapped.set_policy_net(self._models["agent"]["policy"])
            
        #  agent and trainer
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    def _generate_models(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:

        from omni.isaac.lab_tasks.direct.cartpolesktest.agents.models import RLEPolicyModel, RLEValueModel
        from omni.isaac.lab_tasks.direct.cartpolesktest.networks.rle_networks import FeatureNetwork
        
        self._class_mapping.update({
            "rlepolicymodel": RLEPolicyModel,
            "rlevaluemodel": RLEValueModel
        })
        
        device = env.device
        observation_space = env.observation_space if not isinstance(env, MultiAgentEnvWrapper) else env.observation_spaces["agent"]
        action_space = env.action_space if not isinstance(env, MultiAgentEnvWrapper) else env.action_spaces["agent"]
        
        #  feature network
        feature_network = FeatureNetwork(
            obs_shape=env.unwrapped.base_obs_size, 
            feature_size=cfg["agent"].get("feature_size", 32),
            tau=cfg["agent"].get("tau", 0.005),
            device=device
        ).to(device)

        models = {"agent": {}}
        
        model_kwargs = dict(cfg["models"]["policy"])
        del model_kwargs["class"]
        processed_kwargs = self._process_cfg(model_kwargs)
        
        models["agent"]["policy"] = RLEPolicyModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            rle_net=feature_network, 
            **processed_kwargs
        )

        model_kwargs = dict(cfg["models"]["value"])
        del model_kwargs["class"]
        processed_kwargs = self._process_cfg(model_kwargs)
        
        models["agent"]["value"] = RLEValueModel(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            rle_net=feature_network, 
            **processed_kwargs
        )

        if hasattr(env.unwrapped, 'feature_network'):
            env.unwrapped.feature_network = feature_network

        #  latent vectors for the batch size
        if hasattr(env.unwrapped, 'num_envs'):
            feature_network.latent_vectors = feature_network.sample_z(env.unwrapped.num_envs)
            
        return models

    def _generate_agent(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Model]]) -> Agent:
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        if not "memory" in cfg:
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        try:
            memory_class = self._class(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._class("RandomMemory")
        
        memories = {}
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # Agent setup
        try:
            agent_class = self._class(cfg["agent"]["class"])
            del cfg["agent"]["class"]
        except KeyError:
            agent_class = self._class("ppo")
        
        if agent_class in [RLEPPO_SK]:
            agent_id = possible_agents[0]
            agent_cfg = RLEPPO_SK_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            
            # Fix learning rate scheduler
            if isinstance(agent_cfg["learning_rate_scheduler"], str):
                if agent_cfg["learning_rate_scheduler"].lower() == "kladaptivelr":
                    agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
                else:
                    agent_cfg["learning_rate_scheduler"] = None
                    
            # Fix preprocessors
            if isinstance(agent_cfg["state_preprocessor"], str):
                if agent_cfg["state_preprocessor"].lower() == "runningstandardscaler":
                    agent_cfg["state_preprocessor"] = RunningStandardScaler
                else:
                    agent_cfg["state_preprocessor"] = None
                    
            if isinstance(agent_cfg["value_preprocessor"], str):
                if agent_cfg["value_preprocessor"].lower() == "runningstandardscaler":
                    agent_cfg["value_preprocessor"] = RunningStandardScaler
                else:
                    agent_cfg["value_preprocessor"] = None
            
            agent_cfg["state_preprocessor_kwargs"].update({
                "size": observation_spaces[agent_id],
                "device": device
            })
            agent_cfg["value_preprocessor_kwargs"].update({
                "size": 1,
                "device": device
            })
            
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        else:
            # Multi-agent setup implementation...
            pass

        return agent_class(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], agent: Agent) -> Trainer:
        """Generate trainer instance"""
        try:
            trainer_class = self._class(cfg["trainer"]["class"])
            del cfg["trainer"]["class"]
        except KeyError:
            trainer_class = self._class("SequentialTrainer")
        return trainer_class(env=env, agents=agent, cfg=cfg["trainer"])

    def _class(self, value: str) -> Type:
        """Get skrl component class from string identifier"""
        if value.lower() in self._class_mapping:
            return self._class_mapping[value.lower()]
        raise ValueError(f"Unknown class '{value}' in runner cfg")

    def _process_cfg(self, cfg: dict) -> dict:
        """Process configuration dictionary"""
        if not isinstance(cfg, dict):
            return cfg

        processed_cfg = {}
        for key, value in cfg.items():
            if key in ["class", "rle_net"]:  
                continue
            if isinstance(value, dict):
                processed_cfg[key] = self._process_cfg(value)
            else:
                processed_cfg[key] = value
        return processed_cfg

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    def run(self, mode: str = "train") -> None:
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
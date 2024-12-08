# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from pathlib import Path
from . import agents
from .cartpole_rle_env import CartpoleRLEEnv, CartpoleRLEEnvCfg
from .cartpole_env import CartpoleEnv, CartpoleEnvCfg
from .cartpole_env_rndr import CartpoleRLEEnv_Noisy, CartpoleRLEEnv_NoisyCfg


gym.register(
    id="Isaac-Cartpole-rle",
    entry_point="omni.isaac.lab_tasks.direct.cartpolerle.cartpole_rle_env:CartpoleRLEEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleRLEEnvCfg,
        "skrl_cfg_entry_point": str(Path(agents.SKLR_CFG_ROOT) / "ppo_cfg_rle.yaml"),
    },
)

gym.register(
    id="Isaac-Cartpole-ppo",
    entry_point="omni.isaac.lab_tasks.direct.cartpolerle:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.direct.cartpolerle:CartpoleEnvCfg",
        "skrl_cfg_entry_point": str(Path(agents.SKLR_CFG_ROOT) / "skrl_ppo_cfg.yaml"),
    },
)

gym.register(
    id="Isaac-Cartpole-noisy",
    entry_point="omni.isaac.lab_tasks.direct.cartpolerle:CartpoleRLEEnv_Noisy",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "omni.isaac.lab_tasks.direct.cartpolerle:CartpoleRLEEnv_NoisyCfg",
        "skrl_cfg_entry_point": str(Path(agents.SKLR_CFG_ROOT) / "skrl_ppo_cfg_noisy.yaml"),
    },
)
# Adopted form ppo_rle atari

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
import matplotlib.pyplot as plt
import functools

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

from four_room_grid_world.algorithms.ppo_rle_distribution import RLEGoalSamplerCreator
from four_room_grid_world.util.plot_util import plot_heatmap, create_plot_env, add_room_layout_to_plot, \
    plot_trajectories, calculate_states_entropy, is_last_step_in_last_epoch, visit_count_dict_to_list
from four_room_grid_world.env_gymnasium.StateVisitCountWrapper import StateVisitCountWrapper
import four_room_grid_world.env_gymnasium.registration  # Do not remove this import
from four_room_grid_world.env_gymnasium.FourRoomGridWorld import FourRoomGridWorld

ENV_SIZE = 50

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
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="RLE",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="ID of GPU to use")
    parser.add_argument("--tag", type=str, default="PPO_RLE",
                        help="the tag used in wandb")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="advtop/FourRoomGridWorld-v0",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2_500_000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
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
    parser.add_argument("--normalize-ext-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
                        const=True,
                        help="if toggled, extrinsic rewards will be normalized")
    parser.add_argument("--reward-free", type=str, default="True",
                        help="whether to use the version of the four room environment that does not have any rewards")
    parser.add_argument("--tags", nargs="*", type=str, default=["PPO_RLE"],
                        help="a list of tags for wanddb")
    parser.add_argument("--max-episode-steps", type=int, default=1_000,
                        help="maximum number of steps per episode")

    # RLE arguments
    parser.add_argument("--switch-steps", type=int, default=128,
                        help="number of timesteps to switch the RLE network")
    parser.add_argument("--norm-rle-features", type=lambda x: bool(strtobool(x)), default=True,
                        nargs="?", const=True, help="if toggled, rle features will be normalized")
    parser.add_argument("--int-coef", type=float, default=0.1,
                        help="coefficient of extrinsic reward")
    parser.add_argument("--ext-coef", type=float, default=1.0,
                        help="coefficient of intrinsic reward")
    parser.add_argument("--int-gamma", type=float, default=0.99,
                        help="Intrinsic reward discount rate")
    parser.add_argument("--feature-size", type=int, default=4,
                        help="Size of the feature vector output by the rle network")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="The parameter for soft updating the rle network")
    parser.add_argument("--save-rle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, save the rle network at the end of training")
    parser.add_argument("--num-iterations-feat-norm-init", type=int, default=1,
                        help="number of iterations to initialize the feature normalization parameters")
    parser.add_argument("--goal-distribution", type=str, default="standard_normal",
                        help="the distribution that is used by the RLENetwork to sample goals (see ppo_rle_distribution.py)")

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


def make_env(env_id, idx, run_name):
    def thunk():
        env = gym.make(env_id, max_episode_steps=args.max_episode_steps, size=ENV_SIZE, is_reward_free=args.reward_free)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


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
        state_dims = np.array(envs.single_observation_space.shape).prod()

        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dims, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(rle_net.feature_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(64, 64), std=0.1), nn.Tanh())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(64, 64), std=0.01),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(64, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(64, 1), std=0.01)

    def get_action_and_value(self, x, reward, goal, action=None, deterministic=False):
        obs_hidden = self.network(x)

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
        obs_hidden = self.network(x)  # Replaced  x / 255.0 with x

        # goal is the goal vector
        # it is a tensor of shape (num_envs, feature_size)
        goal_hidden = self.goal_encoder(goal)
        hidden = obs_hidden + goal_hidden

        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class RLEModel(nn.Module):
    def __init__(self, input_size, feature_size, output_size, num_actions, num_envs, z_layer_init, device,
                 goal_sampler):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.goal_sampler = goal_sampler

        self.feature_size = feature_size

        # rle network phi(s) with similar architecture to value network
        self.rle_net = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.last_layer = z_layer_init(nn.Linear(64, self.feature_size))

        # Input to the last layer is the feature of the current state, shape = (num_envs, feature_size)
        self.current_ep = 0
        self.num_envs = num_envs
        self.goals = self.sample_goals()

        # num steps left for this goal per environment: switch_steps
        # resets whenever it is 0 or a life ends
        # initially randomize num_steps_left for each environment
        self.num_steps_left = torch.randint(1, args.switch_steps, (num_envs,)).to(device)

        self.switch_goals_mask = torch.zeros(num_envs).to(device)

        # Maintain statistics for the rle network to be used for normalization
        self.rle_rms = RunningMeanStd(shape=(1, self.feature_size))
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def sample_goals(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_envs
        goals = self.goal_sampler.sample(num_envs, self.feature_size, self.device)

        # normalize the goals
        goals = goals / torch.norm(goals, dim=1, keepdim=True)
        return goals

    def step(self, next_done: torch.Tensor):  # Todo: removed next_ep_done: torch.Tensor since it was not used
        """
        next_done: termination indicator
        """
        # switch_goals_mask = 0 if the goal is not updated, 1 if the goal is updated
        # switch_goals_mask is a tensor of shape (num_envs,)
        # sample new goals for the environments that need to update their goals
        self.switch_goals_mask = torch.zeros(args.num_envs).to(device)
        self.switch_goals_mask[next_done.bool()] = 1.0
        self.num_steps_left -= 1
        self.switch_goals_mask[self.num_steps_left == 0] = 1.0

        # update the goals
        new_goals = self.sample_goals()
        self.goals = self.goals * (
                    1 - self.switch_goals_mask.unsqueeze(1)) + new_goals * self.switch_goals_mask.unsqueeze(1)

        # update the num_steps_left
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

    def update_rms(self, b_rle_feats):
        # Update the rle rms
        self.rle_rms.update(b_rle_feats)
        # Update the mean and std tensors
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def compute_reward(self, obs, next_obs, goals=None):
        return self.compute_rle_feat(next_obs, goals=goals)

    def forward(self, obs, next_obs):
        pass


def plot_reward_function(feature_network, x_wall_gap_offset, y_wall_gap_offset, global_step, wandb, save_dir,
                         number_reward_functions=10):
    assert number_reward_functions == 10, "Can only plot exactly 10 reward functions"

    reward_functions = np.zeros((ENV_SIZE + 1, ENV_SIZE + 1, number_reward_functions))

    for i in range(number_reward_functions):
        x = np.arange(ENV_SIZE + 1)
        y = np.arange(ENV_SIZE + 1)
        xx, yy = np.meshgrid(x, y)
        grid_cells = np.vstack([xx.ravel(), yy.ravel()]).T

        z = feature_network.sample_goals(1)
        for x, y in grid_cells:
            obs = torch.Tensor([[x, y]])  # Include batch dimension
            reward_functions[y, x, i], _, _ = feature_network.compute_reward(obs, obs, z)

    # Plotting the 10 reward functions in a grid (2 rows x 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(reward_functions[:, :, i], cmap='viridis', vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        add_room_layout_to_plot(ax, ENV_SIZE, x_wall_gap_offset, y_wall_gap_offset)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label('Reward')

    plt.suptitle(f"Intrinsic Reward Functions at Step {global_step:,}")



def get_trajectories_RLE(env, agent, device, rle_network, number_trajectories):
    trajectories = []

    for i in range(number_trajectories):
        obs, _ = env.reset()
        trajectory = [obs]
        obs = torch.Tensor(obs).to(device).unsqueeze(0)

        z = rle_network.sample_goals(1)

        while True:
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs, 0, z)
                action = action.item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                trajectory.append(next_obs)
                obs = torch.Tensor(next_obs).unsqueeze(0).to(device)

            if terminated or truncated:
                break

        trajectories.append(trajectory)

    return trajectories


def sample_and_log_z_values(number_z_values, rle_network, global_step):
    z_values = rle_network.sample_goals(number_z_values)
    wandb.log({f"z_values_{global_step}": z_values.tolist()})
    return z_values


def log_trajectories(z_values, global_step, env):
    number_trajectories = len(z_values)
    trajectories = []

    for i in range(number_trajectories):
        obs, _ = env.reset()
        trajectory = [obs]
        obs = torch.Tensor(obs).to(device).unsqueeze(0)

        z = z_values[i]

        while True:
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(obs, 0, z)
                action = action.item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                trajectory.append(next_obs)
                obs = torch.Tensor(next_obs).unsqueeze(0).to(device)

            if terminated or truncated:
                break

        trajectories.append(trajectory)

    wandb.log({f"trajectories_{global_step}": trajectories})

def log_reward_functions(z_values, global_step, rle_network):
    number_reward_functions = len(z_values)

    reward_functions = np.zeros((ENV_SIZE + 1, ENV_SIZE + 1, number_reward_functions))

    for i in range(number_reward_functions):
        x = np.arange(ENV_SIZE + 1)
        y = np.arange(ENV_SIZE + 1)
        xx, yy = np.meshgrid(x, y)
        grid_cells = np.vstack([xx.ravel(), yy.ravel()]).T

        z = z_values[i]
        for x, y in grid_cells:
            obs = torch.Tensor([[x, y]])  # Include batch dimension
            reward_functions[x, y, i], _, _ = rle_network.compute_reward(obs, obs, z)

    wandb.log({f"reward_functions_{global_step}": reward_functions.tolist()})


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


if __name__ == "__main__":
    fig = plt.figure()  # create a figure object for plotting rle statistics
    args = parse_args()

    if args.reward_free == "True":
        args.reward_free = True
    elif args.reward_free == "False":
        args.reward_free = False
    else:
        raise RuntimeError("Invalid reward-free parameter")

    if args.reward_free:
        args.tags.append("REWARD_FREE")
    else:
        args.tags.append("NOT_REWARD_FREE")

    if args.goal_distribution:
        args.tags.append(args.goal_distribution)

    os.makedirs(args.local_dir, exist_ok=True)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
            tags=args.tags,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device: ", device)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, run_name) for i in range(args.num_envs)],
    )
    envs = StateVisitCountWrapper(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    plot_env = create_plot_env(args.env_id, ENV_SIZE)

    rle_output_size = args.feature_size  # NOTE: this is not used
    num_actions = envs.single_action_space.n

    goal_sampler = RLEGoalSamplerCreator.create_from_name(args.goal_distribution)
    rle_network = RLEModel(envs.single_observation_space.shape, args.feature_size, rle_output_size, num_actions,
                           args.num_envs,
                           z_layer_init=create_layer_init_from_spec(args.z_layer_init),
                           device=device, goal_sampler=goal_sampler).to(device)
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    next_raw_rle_feat = []

    # Update rms of the rle network
    if args.norm_rle_features:
        print("Start to initialize rle features normalization parameter.....")
        for step in range(args.num_steps * args.num_iterations_feat_norm_init):
            acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
            s, r, d, t, _ = envs.step(acs)
            rle_reward, raw_rle_feat, rle_feat = rle_network.compute_rle_feat(
                torch.Tensor(s).to(device).clone().float())
            next_raw_rle_feat += raw_rle_feat.detach().cpu().numpy().tolist()

            if len(next_raw_rle_feat) % (args.num_steps * args.num_envs) == 0:
                next_raw_rle_feat = np.stack(next_raw_rle_feat)
                rle_network.update_rms(next_raw_rle_feat)
                next_raw_rle_feat = []
        print(f"End of initializing... finished in {time.time() - start_time}")

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
            rle_dones[
                step] = rle_network.switch_goals_mask  # rle_done is True if the goal is switched in the previous step
            goals[step] = rle_network.goals

            # Compute the obs before stepping
            rle_obs = next_obs.clone().float()

            # Compute mean and std over all the envs of norm of rle_obs for logging
            # rle_obs_norm = rle_obs.pow(2).sum(dim=(1, 2, 3)).sqrt() TODO: uncomment

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
            next_obs, reward, done, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(done, truncations)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_obss[step] = next_obs
            rle_next_obs = next_obs.clone().float()
            rle_reward, raw_next_rle_feat, next_rle_feat = rle_network.compute_reward(rle_obs, rle_next_obs)
            rle_rewards[step] = rle_reward.data
            raw_rle_feats[step] = raw_next_rle_feat
            rle_feats[step] = next_rle_feat

            int_reward_info = {
                "int_rewards": rle_rewards[step, 0].cpu().numpy(),
            }

            # update prev rewards
            if step < args.num_steps - 1:
                prev_rewards[step + 1] = rle_rewards[step] * args.int_coef

            if global_step == 500_000 or is_last_step_in_last_epoch(update, num_updates, step, args.num_steps):
                plot_heatmap(infos, global_step, ENV_SIZE, f"runs/{run_name}")
                if args.track:
                    wandb.log({"state_visit_heatmap": wandb.Image(plt.gcf())}, global_step)

                plot_reward_function(rle_network, plot_env.x_wall_gap_offset, plot_env.y_wall_gap_offset, global_step, f"runs/{run_name}", 10)
                if args.track:
                    wandb.log({"reward_functions ": wandb.Image(plt.gcf())}, global_step)

            if global_step == 500_000 or global_step == 1_500_000 or is_last_step_in_last_epoch(update, num_updates,
                                                                                                step, args.num_steps):
                trajectories = get_trajectories_RLE(plot_env, agent, device, rle_network, 5)
                plot_trajectories(global_step, trajectories, ENV_SIZE, plot_env.x_wall_gap_offset,
                                  plot_env.y_wall_gap_offset, f"runs/{run_name}")
                if args.track:
                    wandb.log({"trajectories": wandb.Image(plt.gcf())}, global_step)
                    z_values = sample_and_log_z_values(10, rle_network, global_step)
                    log_trajectories(z_values, global_step, plot_env)
                    log_reward_functions(z_values, global_step, rle_network)

            if is_last_step_in_last_epoch(update, num_updates, step, args.num_steps):
                if args.track:
                    wandb.log({"visit_counts": visit_count_dict_to_list(infos["visit_counts"], ENV_SIZE)})

            for idx, d in enumerate(next_done):
                if d:
                    episodic_return = infos["final_info"][idx]["episode"]["r"].item()
                    episode_length = infos["final_info"][idx]["episode"]["l"].item()
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)
                    avg_returns.append(episodic_return)
                    avg_ep_lens.append(episode_length)
                    if args.track:
                        wandb.log({"charts/episodic_return": episodic_return,
                                   "charts/episodic_length": episode_length},
                                  step=global_step)

            rle_network.step(next_done)

        state_visit_entropy = calculate_states_entropy(infos, global_step, ENV_SIZE)
        if args.track:
            wandb.log({"charts/state_visit_entropy": state_visit_entropy}, step=global_step)

        not_dones = (1.0 - dones).cpu().data.numpy()
        rewards_cpu = rewards.cpu().data.numpy()
        rle_rewards_cpu = rle_rewards.cpu().data.numpy()
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

        data["rewards/rewards_mean"] = rewards.mean().item()
        data["rewards/rewards_max"] = rewards.max().item()
        data["rewards/rewards_min"] = rewards.min().item()
        data["rewards/intrinsic_rewards_mean"] = rle_rewards.mean().item()
        data["rewards/intrinsic_rewards_max"] = rle_rewards.max().item()
        data["rewards/intrinsic_rewards_min"] = rle_rewards.min().item()

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

        # Write dictionary to Tensorboard
        for key, value in data.items():
            if isinstance(value, dict):
                # If a nested dictionary, log each sub-key separately
                for sub_key, sub_value in value.items():
                    writer.add_scalar(f"{key}/{sub_key}", sub_value, global_step)
            else:
                writer.add_scalar(key, value, global_step)

        # Make sure to flush the data to the disk
        writer.flush()

        # Also log the histogram of the returns every 100 iterations
        if update % 100 == 0:
            if args.track:
                data["charts/returns_hist"] = wandb.Histogram(avg_returns)

        if args.track:
            wandb.log(data, step=global_step)
            trigger_sync()

    envs.close()
    writer.close()
    if args.save_rle:
        # make directory if it does not exist
        if not os.path.exists("saved_rle_networks"):
            os.makedirs("saved_rle_networks")
        # save the rle network
        torch.save(rle_network.state_dict(), f"saved_rle_networks/{run_name}.pt")

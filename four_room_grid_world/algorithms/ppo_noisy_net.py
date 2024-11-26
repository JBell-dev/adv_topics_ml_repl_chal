# Adopted from ppo_noisy_net atari
import argparse
import os
import pickle
import random
import time
from collections import deque
from distutils.util import strtobool
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers.normalize import RunningMeanStd
from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from four_room_grid_world.env_gymnasium.StateVisitCountWrapper import StateVisitCountWrapper
from four_room_grid_world.util.plot_util import plot_heatmap, get_trajectories, plot_trajectories, create_plot_env, \
    calculate_states_entropy

from four_room_grid_world.env_gymnasium.registration import register  # DO NOT REMOTE THIS IMPORT


ENV_SIZE = 50


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
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
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
    parser.add_argument("--ent-coef", type=float, default=0.0,  # Removed for noisy net
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument("--sticky-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, sticky action will be used")
    parser.add_argument("--normalize-ext-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?",
                        const=True,
                        help="if toggled, extrinsic rewards will be normalized")


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NoisyLinear(nn.Linear):
    # This NoisyLinear module is taken from: https://github.com/Kaixhin/NoisyNet-A3C/
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        # µ^w and µ^b reuse self.weight and self.bias
        self.sigma_init = sigma_init
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))  # σ^w
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))  # σ^b

        epsilon_weight = torch.zeros(out_features, in_features)
        epsilon_bias = torch.zeros(out_features)
        self.register_buffer('epsilon_weight', epsilon_weight)
        self.register_buffer('epsilon_bias', epsilon_bias)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
            nn.init.uniform_(self.weight, -np.sqrt(3 / self.in_features), np.sqrt(3 / self.in_features))
            nn.init.uniform_(self.bias, -np.sqrt(3 / self.in_features), np.sqrt(3 / self.in_features))
            nn.init.constant_(self.sigma_weight, self.sigma_init)
            nn.init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        weight = self.weight + self.sigma_weight * self.epsilon_weight
        bias = self.bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, weight, bias)

    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features, device=self.device)
        self.epsilon_bias = torch.randn(self.out_features, device=self.device)

    def remove_noise(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features, device=self.device)
        self.epsilon_bias = torch.zeros(self.out_features, device=self.device)

    @property
    def device(self):
        return self.weight.device


class NoisySequential(nn.Sequential):
    def sample_noise(self):
        for module in self:
            if hasattr(module, "sample_noise"):
                module.sample_noise()

    def remove_noise(self):
        for module in self:
            if hasattr(module, "remove_noise"):
                module.remove_noise()


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = NoisySequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64), std=0.017),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), std=0.017),
            nn.Tanh(),
            layer_init(NoisyLinear(64, 1), std=0.017),
        )
        self.actor = NoisySequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64), std=0.017),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64), std=0.017),
            nn.Tanh(),
            layer_init(NoisyLinear(64, envs.single_action_space.n), std=0.017),
        )

    def get_action_and_value(self, x, action=None, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(x)
        )

    def get_value(self, x):
        return self.critic(x)

    def sample_noise(self):
        self.actor.sample_noise()
        self.critic.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()
        self.critic.remove_noise()


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


def make_env(env_id, idx, run_name):
    def thunk():
        env = gym.make(env_id, max_episode_steps=1_000, size=ENV_SIZE)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
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
            tags=["PPO_NOISY_NET"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, run_name) for i in range(args.num_envs)],
    )
    envs.num_envs = args.num_envs

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    envs = StateVisitCountWrapper(envs)
    plot_env = create_plot_env(args.env_id, ENV_SIZE)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5
    )
    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(args.gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=128)
    avg_ep_lens = deque(maxlen=128)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        it_start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.sample_noise()  # Pick a new noise vector (until next optimisation step)

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = np.logical_or(done, truncations)
            next_done = torch.Tensor(next_done).to(device)

            if global_step == 500_000 or global_step == 2_400_000:
                plot_heatmap(infos, global_step, ENV_SIZE, f"runs/{run_name}")
                wandb.log({"State visit heatmap": wandb.Image(plt.gcf())}, global_step)

            if global_step == 500_000 or global_step == 1_500_000 or global_step == 2_400_000:  # TODO Added by me
                trajectories = get_trajectories(plot_env, agent, device)
                plot_trajectories(global_step, trajectories, ENV_SIZE, plot_env.x_wall_gap_offset, plot_env.y_wall_gap_offset, f"runs/{run_name}")

            if global_step == 2_400_000:
                with open(f"runs/{run_name}/ppo_noisy_net_visit_counts.pkl", "wb") as file:
                    pickle.dump(infos["visit_counts"], file)

            for idx, d in enumerate(next_done):
                if d:
                    episodic_return = infos["final_info"][idx]["episode"]["r"].item()
                    episode_length = infos["final_info"][idx]["episode"]["l"].item()
                    avg_returns.append(infos["final_info"][idx]["episode"]["r"].item())
                    avg_ep_lens.append(episode_length)

                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_length, global_step)

        state_visit_entropy = calculate_states_entropy(infos, global_step, ENV_SIZE)
        if args.track:
            wandb.log({"charts/state_visit_entropy": state_visit_entropy}, step=global_step)

        not_dones = (1.0 - dones).cpu().data.numpy()
        rewards_cpu = rewards.cpu().data.numpy()
        if args.normalize_ext_rewards:
            reward_per_env = np.array(
                [discounted_reward.update(rewards_cpu[i], not_dones[i]) for i in range(args.num_steps)]
            )
            reward_rms.update(reward_per_env.flatten())
            rewards /= np.sqrt(reward_rms.var)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        it_end_time = time.time()

        print("SPS:", int(global_step / (time.time() - start_time)))

        data = {}
        data["charts/iterations"] = update
        data["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
        data["losses/value_loss"] = v_loss.item()
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

        # Log the number of envs with positive extrinsic rewards (rewards has shape (num_steps, num_envs))
        data["rewards/num_envs_with_pos_rews"] = torch.sum(rewards.sum(dim=0) > 0).item()

        # Log advantages and intrinsic advantages
        data["returns/advantages"] = b_advantages.mean().item()
        data["returns/ret_ext"] = b_returns.mean().item()
        data["returns/values_ext"] = b_values.mean().item()

        data["charts/traj_len"] = np.mean(avg_ep_lens)
        data["charts/max_traj_len"] = np.max(avg_ep_lens, initial=0)
        data["charts/min_traj_len"] = np.min(avg_ep_lens, initial=0)
        data["charts/time_per_it"] = it_end_time - it_start_time
        data["charts/game_score"] = np.mean(avg_returns)
        data["charts/max_game_score"] = np.max(avg_returns, initial=0)
        data["charts/min_game_score"] = np.min(avg_returns, initial=0)

        if args.track:
            wandb.log(data, step=global_step)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

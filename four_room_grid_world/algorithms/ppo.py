# RUN WITH THESE ARGUMENTS: --seed 0 --env_id advtop/FourRoomGridWorld-v0 --total_timesteps 2500000 --learning_rate 0.001 --num_envs 32

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
# Add these imports at the top
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from four_room_grid_world.env_gymnasium.StateVisitCountWrapper import StateVisitCountWrapper
from four_room_grid_world.util.plot_util import plot_heatmap, plot_trajectories

from four_room_grid_world.env_gymnasium.registration import register  # DO NOT REMOTE THIS IMPORT

ENV_SIZE = 50

def record_episode(env, agent, device, max_steps=200, filename="episode.gif"):
    frames = []
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to(device)

    for step in range(max_steps):
        frames.append(env.render())

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
            action = action.cpu().item()

        obs, _, terminated, truncated, _ = env.step(action)
        obs = torch.Tensor(obs).to(device)

        if terminated or truncated:
            break

    imageio.mimsave(filename, frames, duration=1000 / 30)


def record_episode_with_probs(env, agent, device, max_steps=200, filename="episode.gif", iteration=0):
    frames = []
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to(device)

    # Store action probabilities for plotting
    all_probs = []
    accumulated_reward = 0

    for step in range(max_steps):
        frames.append(env.render())

        with torch.no_grad():
            # Get action and probabilities
            logits = agent.actor(obs)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(logits=logits).sample()

            # Store probabilities
            all_probs.append(probs.cpu().numpy())

            action = action.cpu().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        accumulated_reward += reward
        obs = torch.Tensor(obs).to(device)

        if terminated or truncated:
            break

    # Save the GIF
    imageio.mimsave(filename, frames, duration=1000 / 30)

    # Plot probability histogram
    if len(all_probs) > 0:
        all_probs = np.array(all_probs)
        # Calculate the average probability for each action across all steps
        action_counts = all_probs.sum(axis=0)  # Sum probabilities for each action

        plt.figure(figsize=(10, 5))
        num_actions = len(action_counts)
        plt.bar(range(num_actions), action_counts / len(all_probs))
        plt.title(f'Action Distribution - Iteration {iteration}\nAccumulated Reward: {accumulated_reward:.2f}')
        plt.xlabel('Action')
        plt.ylabel('Average Probability')
        plt.xticks(range(num_actions))
        plt.grid(True)

        # Save the plot
        plot_filename = filename.replace('.gif', '_probs.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Episode {iteration} - Accumulated Reward: {accumulated_reward:.2f}")


@dataclass
class Args:
    """
    Configuration class that defines all hyperparameters and settings:
    - exp_name: Experiment name for logging
    - env_id: The environment to train on
    - total_timesteps: Total number of environment steps to train for
    - learning_rate: Learning rate for the optimizer
    - num_envs: Number of parallel environments
    ... and many other PPO-specific parameters
    """
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    """
    Factory function that creates and wraps environments:
    - Sets up video recording if enabled
    - Wraps environment to record episode statistics
    - Configures environment parameters like max steps
    """

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=None, size=ENV_SIZE)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=None, size=ENV_SIZE)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Neural network architecture with two main components:
    
    1. Actor (Policy) Network:
       - Takes state as input
       - Outputs action probabilities
       - Architecture: Input -> 64 -> 64 -> Action Space Size
    
    2. Critic (Value) Network:
       - Takes state as input
       - Outputs value estimate of the state
       - Architecture: Input -> 64 -> 64 -> 1
    """

    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def get_trajectories(env, agent, device, max_steps=1000):
    trajectories = []

    for i in range(5):
        obs, _ = env.reset()
        obs_list = [obs]
        obs = torch.Tensor(obs).to(device)

        for step in range(max_steps):
            with torch.no_grad():
                logits = agent.actor(obs)
                action = torch.distributions.Categorical(logits=logits).sample()
                action = action.cpu().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            obs_list.append(obs)
            obs = torch.Tensor(obs).to(device)

            if terminated or truncated:
                break

        trajectories.append(obs_list)

    return trajectories


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    envs = StateVisitCountWrapper(envs)  # TODO Added by me

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    record_env = gym.make(args.env_id, render_mode="rgb_array", max_episode_steps=None, size=ENV_SIZE)

    # ALGO Logic: Storage setup -> experience collection
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if global_step == 500_000 or global_step == 2_400_000:
                plot_heatmap(infos, global_step, ENV_SIZE)

            if global_step == 500_000 or global_step == 1_500_000 or global_step == 2_400_000:  # TODO Added by me
                trajectories = get_trajectories(record_env, agent, device, max_steps=200)
                plot_trajectories(global_step, trajectories, ENV_SIZE, record_env.x_wall_gap_offset, record_env.y_wall_gap_offset)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # HERE I ADDED THIS TO CHECK WHETHER THE AGENT IS LEARNING
        if iteration % 100 == 0:  # Save GIF and plot every 100 iterations
            gif_path = f"gifs_ppo/episode_{iteration}.gif"
            os.makedirs("gifs_ppo", exist_ok=True)
            record_episode_with_probs(
                env=record_env,
                agent=agent,
                device=device,
                filename=gif_path,
                iteration=iteration
            )
            print(f"Saved episode recording and probability plot to {gif_path}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
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
                """
                PPO's key feature: Clipped objective function
                - Prevents too large policy updates
                - ratio: new_policy / old_policy
                - clipping prevents ratio from going too far from 1
                """

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    """
                    Value function is trained to predict expected returns
                    Can also be clipped to prevent large updates
                    """
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

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes -> log to tensorboard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    record_env.close()

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import gymnasium as gym

def plot_heatmap(infos, global_step, env_size):
    # Uncomment only for debugging
    # sorted_dict = dict(sorted(infos["visit_counts"].items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)

    states = np.zeros(shape=(env_size + 1, env_size + 1))
    for (x, y), count in infos["visit_counts"].items():
        states[y, x] = count  # imshow expects row, column
    # Apply a logarithmic color scale
    plt.imshow(states, cmap='viridis', norm=LogNorm(vmin=1, vmax=states.max() or 1))
    plt.colorbar()
    plt.title(f"State visit count at time step {global_step:,}")
    plt.show()


def create_plot_env(env_id, env_size):
    return gym.make(env_id, max_episode_steps=1_000, size=env_size)


def get_trajectories(env, agent, device):
    trajectories = []

    for i in range(5):
        obs, _ = env.reset()
        obs_list = [obs]
        obs = torch.Tensor(obs).to(device)

        while True:
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


def plot_trajectories(global_step, trajectories, env_size, x_wall_gap_offset, y_wall_gap_offset):
    assert len(trajectories) == 5, "Currently only supports plotting five trajectories"

    # Define colors and markers for different trajectories
    colors = ['blue', 'orange', 'green', 'purple', 'red']
    markers = ['o', 's', '^', 'D', 'P']

    plt.figure(figsize=(8, 8))

    # Plot each trajectory
    for i, trajectory in enumerate(trajectories):
        x, y = zip(*trajectory)  # Separate x and y coordinates
        plt.scatter(x, y, c=colors[i], marker=markers[i], label=f'Trajectory {i + 1}', s=20)

    centre = env_size // 2

    # Fill between for the horizontal lines (left, middle, right)
    plt.fill_between([0 - 1 / 2, centre - x_wall_gap_offset - 1 / 2], centre + 1 / 2, centre - 1 / 2, color='black',
                     lw=0)  # Left segment
    plt.fill_between([centre - x_wall_gap_offset + 1 / 2, centre + x_wall_gap_offset - 1 / 2], centre + 1 / 2,
                     centre - 1 / 2,
                     color='black', lw=0)  # Middle segment
    plt.fill_between([centre + x_wall_gap_offset + 1 / 2, env_size + 1 / 2], centre + 1 / 2, centre - 1 / 2,
                     color='black',
                     lw=0)  # Right segment

    # Fill between for the vertical lines (upper, middle, lower)
    plt.fill_betweenx([0 - 1 / 2, centre - y_wall_gap_offset - 1 / 2], centre + 1 / 2, centre - 1 / 2, color='black',
                      lw=1)  # Upper segment
    plt.fill_betweenx([centre - y_wall_gap_offset + 1 / 2, centre + y_wall_gap_offset - 1 / 2], centre + 1 / 2,
                      centre - 1 / 2,
                      color='black', lw=0)  # Middle segment
    plt.fill_betweenx([centre + y_wall_gap_offset + 1 / 2, env_size + 1 / 2], centre + 1 / 2, centre - 1 / 2,
                      color='black',
                      lw=0)  # Lower segment

    # Adjust plot settings
    plt.xlim(0 - 1 / 2, env_size + 1 - 1 + 1 / 2)
    plt.ylim(env_size + 1 - 1 + 1 / 2, 0 - 1 / 2)  # Reverse order to invert y-axis
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Five trajectories at time step {global_step:,}")
    plt.legend()
    plt.show()

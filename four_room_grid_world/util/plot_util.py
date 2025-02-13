import cv2
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont


def plot_heatmap(infos, global_step, env_size, save_dir=None):
    # Uncomment only for debugging
    # sorted_dict = dict(sorted(infos["visit_counts"].items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)

    plt.clf()  # Otherwise sometimes the trajectories are plotted here

    states = np.zeros(shape=(env_size + 1, env_size + 1))
    for (x, y), count in infos["visit_counts"].items():
        states[y, x] = count  # imshow expects row, column
    # Apply a logarithmic color scale
    plt.imshow(states, cmap='viridis', norm=LogNorm(vmin=1, vmax=states.max() or 1))
    plt.colorbar()
    plt.title(f"State visit count at time step {global_step:,}")

    return plt


def is_last_step_in_last_epoch(epoch, number_epochs, step, number_steps):
    # Number steps is zero-indexed
    return epoch == number_epochs and step == number_steps - 1


def calculate_states_entropy(infos, global_step, env_size):
    states = np.zeros(shape=(env_size + 1, env_size + 1))
    for (x, y), count in infos["visit_counts"].items():
        states[y, x] = count / global_step  # y is vertical

    linearized_states = states.flatten()
    linearized_states = np.clip(linearized_states, 1e-20, 1.0)  # Avoid division by zero
    entropy = -np.sum(linearized_states * np.log(linearized_states))
    return entropy


def create_plot_env(env_id, env_size, is_reward_free):
    return gym.make(env_id, max_episode_steps=1_000, size=env_size, is_reward_free=is_reward_free,
                    render_mode="rgb_array")


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


def add_room_layout_to_plot(p, env_size, x_wall_gap_offset, y_wall_gap_offset):
    centre = env_size // 2

    # Fill between for the horizontal lines (left, middle, right)
    p.fill_between([0 - 1 / 2, centre - x_wall_gap_offset - 1 / 2], centre + 1 / 2, centre - 1 / 2, color='black',
                   lw=0)  # Left segment
    p.fill_between([centre - x_wall_gap_offset + 1 / 2, centre + x_wall_gap_offset - 1 / 2], centre + 1 / 2,
                   centre - 1 / 2,
                   color='black', lw=0)  # Middle segment
    p.fill_between([centre + x_wall_gap_offset + 1 / 2, env_size + 1 / 2], centre + 1 / 2, centre - 1 / 2,
                   color='black',
                   lw=0)  # Right segment

    # Fill between for the vertical lines (upper, middle, lower)
    p.fill_betweenx([0 - 1 / 2, centre - y_wall_gap_offset - 1 / 2], centre + 1 / 2, centre - 1 / 2, color='black',
                    lw=0)  # Upper segment
    p.fill_betweenx([centre - y_wall_gap_offset + 1 / 2, centre + y_wall_gap_offset - 1 / 2], centre + 1 / 2,
                    centre - 1 / 2,
                    color='black', lw=0)  # Middle segment
    p.fill_betweenx([centre + y_wall_gap_offset + 1 / 2, env_size + 1 / 2], centre + 1 / 2, centre - 1 / 2,
                    color='black',
                    lw=0)  # Lower segment


def plot_trajectories(global_step, trajectories, env_size, x_wall_gap_offset, y_wall_gap_offset, save_dir):
    assert len(trajectories) == 5, "Currently only supports plotting five trajectories"

    # Define colors and markers for different trajectories
    colors = ['blue', 'orange', 'green', 'purple', 'red']
    markers = ['o', 's', '^', 'D', 'P']

    plt.figure(figsize=(8, 8))

    # Plot each trajectory
    for i, trajectory in enumerate(trajectories):
        x, y = zip(*trajectory)  # Separate x and y coordinates
        plt.scatter(x, y, c=colors[i], marker=markers[i], label=f'Trajectory {i + 1}', s=20)

    add_room_layout_to_plot(plt, env_size, x_wall_gap_offset, y_wall_gap_offset)

    # Adjust plot settings
    plt.xlim(0 - 1 / 2, env_size + 1 - 1 + 1 / 2)
    plt.ylim(env_size + 1 - 1 + 1 / 2, 0 - 1 / 2)  # Reverse order to invert y-axis
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Five trajectories at time step {global_step:,}")
    plt.legend()


def visit_count_dict_to_list(visit_count_dict, env_size):
    visit_array = np.zeros((env_size + 1, env_size + 1), dtype=int)
    for (x, y), count in visit_count_dict.items():
        visit_array[x, y] = count

    return visit_array.tolist()


def create_demo(rle_network, agent, plot_env, save_dir):
    font_size = 70
    number_trajectories = 5

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default(font_size)  # Fallback to default font

    gif_path = f"{save_dir}/grid_world_demo.gif"
    with imageio.get_writer(gif_path, mode="I", duration=1 / 30) as writer:
        for i in range(number_trajectories):
            obs, _ = plot_env.reset()
            obs = torch.Tensor(obs).unsqueeze(0)

            z = rle_network.sample_goals(1)

            while True:
                frame = plot_env.render()

                # Convert the frame to a PIL Image
                pil_frame = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_frame)

                # Add z_i text to the bottom-right corner
                text = f"z_{i}"
                width = draw.textlength(text, font=font)
                height = font_size

                position = (pil_frame.width - width - 50, pil_frame.height - height - 50)
                draw.text(position, text, fill="red", font=font)

                # Append the modified frame to the GIF
                writer.append_data(pil_frame)

                with torch.no_grad():
                    action, _, _, _, _ = agent.get_action_and_value(obs, 0, z)
                    action = action.item()

                    next_obs, reward, terminated, truncated, _ = plot_env.step(action)
                    obs = torch.Tensor(next_obs).unsqueeze(0)

                    if terminated or truncated:
                        break

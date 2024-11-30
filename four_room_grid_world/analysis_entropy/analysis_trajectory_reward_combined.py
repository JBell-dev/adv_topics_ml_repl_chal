import numpy as np
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from four_room_grid_world.util.plot_util import add_room_layout_to_plot

START = "500000"
MID = "1500000"
END = "2498560"

# Select for which point in time the plot should be generated.
# Can select between START, MID and END.
SELECTED = END

RUN_NAME = "/random-latent-exploration/RLE/runs/jlats3hg"
x_wall_gap_offset = 10
y_wall_gap_offset = 5
env_size = 50

api = wandb.Api()
run = api.run(RUN_NAME)

summary = run.summary

reward_functions = summary["reward_functions_" + SELECTED]
z_values = summary["z_values_" + SELECTED]
trajectories = summary["trajectories_" + SELECTED]

reward_functions = np.array(reward_functions)
trajectories = np.array(trajectories)


def plot_combined(trajectories, reward_functions, env_size, x_wall_gap_offset, y_wall_gap_offset):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
    axes_flat = axes.flat

    colors = [(1, 100/255, 1), (1, 0, 1), (0, 0, 0)]  # Magenta bright to Magenta dark to Black
    magenta_to_black = LinearSegmentedColormap.from_list("MagentaToBlack", colors, N=256)

    colors = np.linspace(0, 1, len(trajectories[0]))

    for i, trajectory in zip(range(4), trajectories[0:4]):
        reward_function = reward_functions[:, :, i]
        # Need to swap axis since imshow has vertical x-axis and horizontal y-axis
        reward_function = np.swapaxes(reward_function, 0, 1)

        ax = axes_flat[i]
        ax.scatter(trajectory[:, 0], trajectory[:, 1], c=colors, cmap=magenta_to_black, marker="o", s=10)
        add_room_layout_to_plot(ax, env_size, x_wall_gap_offset, y_wall_gap_offset)
        im = ax.imshow(reward_function, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_xlim(0 - 1 / 2, env_size + 1 - 1 + 1 / 2)
        ax.set_ylim(env_size + 1 - 1 + 1 / 2, 0 - 1 / 2)  # Reverse order to invert y-axis

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
    cbar.set_label("Intrinsic reward")
    plt.show()


plot_combined(trajectories, reward_functions, env_size, x_wall_gap_offset, y_wall_gap_offset)
